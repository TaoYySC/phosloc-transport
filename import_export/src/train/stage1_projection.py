import copy
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Sampler

from src.models.losses import projection_training_loss
from src.models.projection_mlp import ProjectionMLP
from src.utils import setup_seed


class _EmbeddingDataset(Dataset):
    def __init__(self, x, y, sample_weights=None):
        self.x = torch.tensor(np.asarray(x, dtype=np.float32))
        self.y = torch.tensor(np.asarray(y, dtype=np.int64))
        if sample_weights is None:
            self.sample_weights = torch.ones(len(y), dtype=torch.float32)
        else:
            self.sample_weights = torch.tensor(
                np.asarray(sample_weights, dtype=np.float32)
            )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.sample_weights[idx]


class BalancedClassBatchSampler(Sampler):
    """Sample batches with at least min_per_class from each class when possible."""

    def __init__(self, labels, batch_size=32, min_per_class=2, seed=42):
        self.labels = np.asarray(labels).astype(int)
        self.batch_size = int(batch_size)
        self.min_per_class = int(min_per_class)
        self.seed = int(seed)

        self.pos_idx = np.flatnonzero(self.labels == 1).tolist()
        self.neg_idx = np.flatnonzero(self.labels == 0).tolist()
        self.n_samples = len(self.labels)

        if len(self.pos_idx) < self.min_per_class or len(self.neg_idx) < self.min_per_class:
            self.fallback = True
        else:
            self.fallback = False

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        if self.fallback:
            order = rng.permutation(self.n_samples)
            for start in range(0, self.n_samples, self.batch_size):
                yield order[start : start + self.batch_size].tolist()
            return

        pos = rng.permutation(self.pos_idx).tolist()
        neg = rng.permutation(self.neg_idx).tolist()
        pos_ptr = 0
        neg_ptr = 0
        emitted = 0

        while emitted < self.n_samples:
            batch = []
            for _ in range(self.min_per_class):
                if pos_ptr >= len(pos):
                    pos = rng.permutation(self.pos_idx).tolist()
                    pos_ptr = 0
                batch.append(pos[pos_ptr])
                pos_ptr += 1
            for _ in range(self.min_per_class):
                if neg_ptr >= len(neg):
                    neg = rng.permutation(self.neg_idx).tolist()
                    neg_ptr = 0
                batch.append(neg[neg_ptr])
                neg_ptr += 1

            remaining = self.batch_size - len(batch)
            pools = pos[pos_ptr:] + neg[neg_ptr:]
            if len(pools) < remaining:
                pools = self.pos_idx + self.neg_idx
            if remaining > 0 and pools:
                extra = rng.choice(pools, size=min(remaining, len(pools)), replace=False)
                batch.extend(extra.tolist())

            rng.shuffle(batch)
            emitted += len(batch)
            yield batch

    def __len__(self):
        return max(1, int(np.ceil(self.n_samples / self.batch_size)))


def build_inner_train_val_split(
    df_train,
    group_col="Cluster_ID",
    seed=42,
    val_fraction=0.2,
):
    groups = df_train[group_col].astype(str).values
    idx = np.arange(len(df_train))
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=val_fraction,
        random_state=seed,
    )
    inner_train_idx, inner_val_idx = next(splitter.split(idx, groups=groups))
    return inner_train_idx, inner_val_idx


def fit_stage1_scaler(x_train):
    scaler = StandardScaler()
    scaler.fit(np.asarray(x_train, dtype=np.float32))
    return scaler


def transform_stage1_embedding_with_projection(model, scaler, x):
    x_scaled = scaler.transform(np.asarray(x, dtype=np.float32))
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        z = model.forward_projection(x_tensor).cpu().numpy()
    return z.astype(np.float32)


def _resolve_class_weights(y, class_weight):
    y = np.asarray(y).astype(int)
    n = len(y)
    if class_weight is None:
        return None
    if class_weight == "balanced":
        weights = np.zeros(n, dtype=np.float32)
        for label in (0, 1):
            mask = y == label
            count = int(mask.sum())
            if count <= 0:
                raise ValueError("balanced class_weight requires both classes.")
            weights[mask] = float(n) / (2.0 * count)
        return weights
    if isinstance(class_weight, dict):
        weights = np.ones(n, dtype=np.float32)
        for label, value in class_weight.items():
            weights[y == int(label)] = float(value)
        return weights
    raise ValueError(f"Unsupported class_weight: {class_weight!r}")


def _safe_auroc(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_prob))


def train_projection_model(
    x_train,
    y_train,
    x_val=None,
    y_val=None,
    sample_weights=None,
    input_dim=None,
    hidden_dim=64,
    projection_dim=16,
    dropout=0.3,
    alpha=0.05,
    temperature=0.1,
    batch_size=32,
    lr=3e-4,
    weight_decay=1e-3,
    max_epochs=200,
    early_stopping_patience=20,
    selection_metric="val_auroc",
    class_weight="balanced",
    seed=42,
):
    setup_seed(seed, deterministic=True)

    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=int).reshape(-1)
    if input_dim is None:
        input_dim = x_train.shape[1]

    if sample_weights is None:
        sample_weights = _resolve_class_weights(y_train, class_weight)

    model = ProjectionMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        projection_dim=projection_dim,
        dropout=dropout,
        num_classes=2,
    )

    dataset = _EmbeddingDataset(x_train, y_train, sample_weights=sample_weights)
    sampler = BalancedClassBatchSampler(
        y_train,
        batch_size=batch_size,
        min_per_class=2,
        seed=seed,
    )
    loader = DataLoader(dataset, batch_sampler=sampler)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )

    has_val = x_val is not None and y_val is not None and len(x_val) > 0
    if has_val:
        x_val = np.asarray(x_val, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=int).reshape(-1)

    best_state = copy.deepcopy(model.state_dict())
    best_score = -np.inf
    best_epoch = 0
    epochs_without_improve = 0
    history_rows = []

    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        epoch_total = 0.0
        epoch_ce = 0.0
        epoch_supcon = 0.0
        n_batches = 0

        for xb, yb, wb in loader:
            optimizer.zero_grad()
            projections = model.forward_projection(xb)
            logits = model.classifier(projections)
            loss, ce_loss, supcon_loss = projection_training_loss(
                logits=logits,
                projections=projections,
                labels=yb,
                sample_weights=wb,
                alpha=alpha,
                temperature=temperature,
            )
            loss.backward()
            optimizer.step()

            epoch_total += float(loss.item())
            epoch_ce += float(ce_loss.item())
            epoch_supcon += float(supcon_loss.item())
            n_batches += 1

        row = {
            "epoch": epoch,
            "train_loss": epoch_total / max(n_batches, 1),
            "train_ce_loss": epoch_ce / max(n_batches, 1),
            "train_supcon_loss": epoch_supcon / max(n_batches, 1),
        }

        if has_val:
            model.eval()
            with torch.no_grad():
                val_proj = model.forward_projection(torch.tensor(x_val, dtype=torch.float32))
                val_logits = model.classifier(val_proj)
                val_prob = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
            val_auroc = _safe_auroc(y_val, val_prob)
            val_loss = float(
                F.cross_entropy(val_logits, torch.tensor(y_val, dtype=torch.long)).item()
            )
            row["val_auroc"] = val_auroc
            row["val_loss"] = val_loss

            if selection_metric == "val_loss":
                score = -val_loss
            else:
                score = val_auroc if not np.isnan(val_auroc) else -np.inf

            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
        else:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        history_rows.append(row)

        if has_val and epochs_without_improve >= int(early_stopping_patience):
            break

    model.load_state_dict(best_state)
    history_df = pd.DataFrame(history_rows)
    meta = {
        "best_epoch": int(best_epoch),
        "best_score": float(best_score) if has_val else np.nan,
        "selection_metric": selection_metric,
        "alpha": float(alpha),
        "temperature": float(temperature),
        "hidden_dim": int(hidden_dim),
        "projection_dim": int(projection_dim),
        "dropout": float(dropout),
    }
    return model, history_df, meta
