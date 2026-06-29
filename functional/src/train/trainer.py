import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from src.data.torch_dataset import MultiInputDataset
from src.models.losses import build_binary_loss
from src.evaluate.metrics import find_best_threshold, compute_metrics


def multi_input_collate(batch):
    out = {}
    keys = batch[0].keys()

    for k in keys:
        vals = [b[k] for b in batch]

        if k == "y":
            out[k] = torch.stack(vals, dim=0)
        elif hasattr(vals[0], "edge_index") and hasattr(vals[0], "x"):
            out[k] = Batch.from_data_list(vals)
        else:
            out[k] = torch.stack(vals, dim=0)

    return out


class Trainer:
    def __init__(
        self,
        model,
        input_keys=None,
        device="cuda",
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=128,
        num_epochs=50,
        pos_weight=None,
        loss_type="bce",
        class_prior=0.1,
        u_loss_weight=1.0,
        non_negative=True,
        positive_loss_weight=1.0,
        early_stopping=True,
        early_stopping_metric="val_auprc",
        early_stopping_patience=8,
        early_stopping_min_delta=0.0,
        val_eval_interval=1,
        seed=42,
    ):
        self.model = model.to(device)
        self.input_keys = input_keys
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.seed = seed

        self.early_stopping = early_stopping
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.val_eval_interval = max(1, int(val_eval_interval))
        self.loss_type = str(loss_type).lower()

        self.criterion = build_binary_loss(
            loss_type=loss_type,
            pos_weight=pos_weight,
            class_prior=class_prior,
            u_loss_weight=u_loss_weight,
            non_negative=non_negative,
            positive_loss_weight=positive_loss_weight,
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    def _make_loader(self, feature_dict, y, shuffle):
        ds = MultiInputDataset(feature_dict=feature_dict, y=y)

        g = torch.Generator()
        g.manual_seed(self.seed)

        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
            generator=g,
            collate_fn=multi_input_collate,
        )

    def _move_batch_to_device(self, batch):
        out = {}
        for k, v in batch.items():
            if hasattr(v, "to"):
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out

    def _select_model_inputs(self, batch):
        keys = self.input_keys
        if keys is None:
            keys = [k for k in batch.keys() if k != "y"]
        return {k: batch[k] for k in keys if k in batch}

    def _run_one_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        total_n = 0

        for batch in loader:
            batch = self._move_batch_to_device(batch)
            y = batch["y"]

            model_inputs = self._select_model_inputs(batch)
            logits = self.model(model_inputs)
            if logits.dim() > 1 and logits.size(-1) == 1:
                logits = logits.squeeze(-1)
            loss = self.criterion(logits, y)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            bs = y.size(0)
            total_loss += loss.item() * bs
            total_n += bs

        return total_loss / max(total_n, 1)

    @torch.no_grad()
    def predict_proba(self, feature_dict, y):
        loader = self._make_loader(feature_dict, y, shuffle=False)
        self.model.eval()

        probs = []
        ys = []

        for batch in loader:
            batch = self._move_batch_to_device(batch)
            y_b = batch["y"]

            model_inputs = self._select_model_inputs(batch)
            logits = self.model(model_inputs)
            if logits.dim() > 1 and logits.size(-1) == 1:
                logits = logits.squeeze(-1)
            p = torch.sigmoid(logits)

            probs.append(p.cpu().numpy())
            ys.append(y_b.cpu().numpy())

        return np.concatenate(ys), np.concatenate(probs)

    @torch.no_grad()
    def extract_classifier_repr(self, feature_dict, y, n_repr_layers=2):
        from src.models.classifier_repr import forward_with_repr

        loader = self._make_loader(feature_dict, y, shuffle=False)
        self.model.eval()

        repr_buffers = {}
        probs = []
        ys = []

        for batch in loader:
            batch = self._move_batch_to_device(batch)
            y_b = batch["y"]
            model_inputs = self._select_model_inputs(batch)
            outputs = forward_with_repr(self.model, model_inputs, n_repr_layers=n_repr_layers)

            logits = outputs["logits"]
            if logits.dim() > 1 and logits.size(-1) == 1:
                logits = logits.squeeze(-1)
            p = torch.sigmoid(logits)

            for key, value in outputs.items():
                if key == "logits":
                    continue
                if key not in repr_buffers:
                    repr_buffers[key] = []
                repr_buffers[key].append(value.cpu().numpy())

            probs.append(p.cpu().numpy())
            ys.append(y_b.cpu().numpy())

        result = {
            key: np.concatenate(values, axis=0)
            for key, values in repr_buffers.items()
        }
        result["y"] = np.concatenate(ys)
        result["prob"] = np.concatenate(probs)
        return result

    def _get_monitored_value(self, val_metrics):
        if self.early_stopping_metric == "val_auprc":
            return float(val_metrics["auprc"])
        if self.early_stopping_metric == "val_auroc":
            return float(val_metrics["auroc"])
        if self.early_stopping_metric == "val_f1":
            return float(val_metrics["f1"])
        if self.early_stopping_metric == "val_mcc":
            return float(val_metrics["mcc"])
        raise ValueError(f"Unsupported early_stopping_metric: {self.early_stopping_metric}")

    def fit(self, train_features, y_train, val_features, y_val):
        train_loader = self._make_loader(train_features, y_train, shuffle=True)

        best_state = None
        best_threshold = 0.5
        best_epoch = 0
        best_monitor = -np.inf
        no_improve_count = 0

        history = []

        last_val_metrics = None
        last_thr = 0.5

        for epoch in range(1, self.num_epochs + 1):
            train_loss = self._run_one_epoch(train_loader)

            should_eval_val = (
                epoch == 1
                or epoch == self.num_epochs
                or epoch % self.val_eval_interval == 0
            )
            if should_eval_val:
                y_val_true, y_val_prob = self.predict_proba(val_features, y_val)
                last_thr = find_best_threshold(y_val_true, y_val_prob)
                last_val_metrics = compute_metrics(y_val_true, y_val_prob, last_thr)

            val_metrics = last_val_metrics
            thr = last_thr
            monitor_value = self._get_monitored_value(val_metrics)

            if should_eval_val:
                improved = monitor_value > (best_monitor + self.early_stopping_min_delta)
                if improved:
                    best_monitor = monitor_value
                    best_threshold = float(thr)
                    best_state = copy.deepcopy(self.model.state_dict())
                    best_epoch = epoch
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                improved = False

            history.append({
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_auroc": float(val_metrics["auroc"]),
                "val_auprc": float(val_metrics["auprc"]),
                "val_f1": float(val_metrics["f1"]),
                "val_mcc": float(val_metrics["mcc"]),
                "threshold": float(thr),
                "monitor_value": float(monitor_value),
                "best_epoch_so_far": int(best_epoch),
                "no_improve_count": int(no_improve_count),
            })

            if self.early_stopping and no_improve_count >= self.early_stopping_patience:
                print(
                    f"[INFO] Early stopping at epoch {epoch}. "
                    f"Best epoch = {best_epoch}, "
                    f"{self.early_stopping_metric} = {best_monitor:.6f}"
                )
                break

        if best_state is None:
            raise RuntimeError("Training failed to produce a valid checkpoint.")

        self.model.load_state_dict(best_state)
        return history, best_threshold