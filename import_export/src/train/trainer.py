import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from src.data.torch_dataset import MultiInputDataset
from src.models.losses import build_binary_loss


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
        seed=42,
    ):
        self.model = model.to(device)
        self.input_keys = input_keys
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.seed = seed

        self.criterion = build_binary_loss(pos_weight=pos_weight)
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
            loss = self.criterion(logits, y)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            bs = y.size(0)
            total_loss += loss.item() * bs
            total_n += bs

        return total_loss / max(total_n, 1)

    def fit(self, train_features, y_train):
        train_loader = self._make_loader(train_features, y_train, shuffle=True)

        history = []

        for epoch in range(1, self.num_epochs + 1):
            train_loss = self._run_one_epoch(train_loader)

            history.append({
                "epoch": int(epoch),
                "train_loss": float(train_loss),
            })

        return history

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
            p = torch.sigmoid(logits)

            probs.append(p.detach().cpu().numpy().reshape(-1))
            ys.append(y_b.detach().cpu().numpy().reshape(-1))

        return np.concatenate(ys), np.concatenate(probs)