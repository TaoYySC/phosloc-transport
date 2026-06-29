import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin


def _resolve_sample_weights(y, class_weight):
    y = np.asarray(y).astype(int)
    n = len(y)

    if class_weight is None:
        return np.ones(n, dtype=np.float32)

    if class_weight == "balanced":
        weights = np.zeros(n, dtype=np.float32)
        for label in (0, 1):
            mask = y == label
            count = int(mask.sum())
            if count <= 0:
                raise ValueError("class_weight=balanced requires both classes in training data.")
            weights[mask] = float(n) / (2.0 * count)
        return weights

    if isinstance(class_weight, dict):
        weights = np.ones(n, dtype=np.float32)
        for label, value in class_weight.items():
            weights[y == int(label)] = float(value)
        return weights

    raise ValueError(f"Unsupported class_weight for pairwise ranking: {class_weight!r}")


class PairwiseRankingClassifier(BaseEstimator, ClassifierMixin):
    """Linear scorer trained with pairwise ranking loss on positive-negative pairs."""

    def __init__(
        self,
        C=1.0,
        max_iter=5000,
        lr=0.05,
        tol=1e-5,
        class_weight="balanced",
        random_state=42,
    ):
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.lr = float(lr)
        self.tol = float(tol)
        self.class_weight = class_weight
        self.random_state = int(random_state)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        n_samples, n_features = X.shape
        if n_samples != len(y):
            raise ValueError("X and y must have the same number of samples.")

        pos_idx = np.flatnonzero(y == 1)
        neg_idx = np.flatnonzero(y == 0)
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            raise ValueError("Pairwise ranking requires both positive and negative samples.")

        torch.manual_seed(self.random_state)
        device = torch.device("cpu")

        x_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        sample_weights = _resolve_sample_weights(y, self.class_weight)
        pos_weights = torch.tensor(sample_weights[pos_idx], dtype=torch.float32, device=device)
        neg_weights = torch.tensor(sample_weights[neg_idx], dtype=torch.float32, device=device)

        weight = torch.zeros(n_features, dtype=torch.float32, device=device, requires_grad=True)
        bias = torch.zeros(1, dtype=torch.float32, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([weight, bias], lr=self.lr)
        reg_strength = 1.0 / max(self.C, 1e-12)

        pair_weights = pos_weights.unsqueeze(1) * neg_weights.unsqueeze(0)
        pair_weights = pair_weights / pair_weights.sum()

        prev_loss = None
        for _ in range(self.max_iter):
            scores = x_tensor @ weight + bias
            score_pos = scores[pos_idx]
            score_neg = scores[neg_idx]
            pair_diff = score_pos.unsqueeze(1) - score_neg.unsqueeze(0)
            ranking_loss = (F.softplus(-pair_diff) * pair_weights).sum()
            reg_loss = 0.5 * reg_strength * (weight.pow(2).sum() + bias.pow(2).sum())
            loss = ranking_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = float(loss.item())
            if prev_loss is not None and abs(prev_loss - loss_value) < self.tol:
                break
            prev_loss = loss_value

        self.coef_ = weight.detach().cpu().numpy().reshape(1, -1)
        self.intercept_ = bias.detach().cpu().numpy()
        self.n_features_in_ = n_features
        self.classes_ = np.array([0, 1], dtype=int)
        self.loss_type_ = "pairwise_ranking"
        self.n_iter_ = self.max_iter
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float32)
        return (X @ self.coef_.T + self.intercept_).reshape(-1)

    def predict_proba(self, X):
        scores = self.decision_function(X)
        prob_pos = 1.0 / (1.0 + np.exp(-scores))
        prob_neg = 1.0 - prob_pos
        return np.column_stack([prob_neg, prob_pos])

    def predict(self, X):
        scores = self.decision_function(X)
        return (scores >= 0.0).astype(int)
