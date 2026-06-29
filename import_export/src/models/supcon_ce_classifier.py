import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin

from src.models.losses import binary_focal_loss, supervised_contrastive_loss
from src.models.pairwise_ranking_classifier import _resolve_sample_weights


class SupConCEClassifier(BaseEstimator, ClassifierMixin):
    """Linear classifier with L = L_cls + alpha * L_SupCon (CE or Focal)."""

    def __init__(
        self,
        C=1.0,
        alpha=1.0,
        temperature=0.07,
        embed_dim=None,
        max_iter=5000,
        lr=0.05,
        tol=1e-5,
        class_weight="balanced",
        classification_loss="ce",
        focal_gamma=2.0,
        focal_alpha=0.25,
        random_state=42,
    ):
        self.C = float(C)
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.embed_dim = embed_dim
        self.max_iter = int(max_iter)
        self.lr = float(lr)
        self.tol = float(tol)
        self.class_weight = class_weight
        self.classification_loss = str(classification_loss).lower()
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = focal_alpha
        self.random_state = int(random_state)

    def _resolve_embed_dim(self, n_features, n_samples):
        if self.embed_dim is not None:
            return int(min(self.embed_dim, n_features, max(n_samples - 1, 1)))
        return int(min(64, n_features, max(n_samples - 1, 1)))

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        n_samples, n_features = X.shape
        if n_samples != len(y):
            raise ValueError("X and y must have the same number of samples.")
        if len(np.unique(y)) < 2:
            raise ValueError("SupCon+CE requires both positive and negative samples.")

        torch.manual_seed(self.random_state)
        device = torch.device("cpu")

        x_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
        sample_weights_np = _resolve_sample_weights(y, self.class_weight)
        sample_weights = torch.tensor(sample_weights_np, dtype=torch.float32, device=device)
        pos_weight = None
        if self.classification_loss == "ce":
            if self.class_weight == "balanced":
                n_pos = float((y == 1).sum())
                n_neg = float((y == 0).sum())
                if n_pos > 0:
                    pos_weight = torch.tensor(
                        [n_neg / n_pos], dtype=torch.float32, device=device
                    )
            elif isinstance(self.class_weight, dict):
                w0 = float(self.class_weight.get(0, 1.0))
                w1 = float(self.class_weight.get(1, 1.0))
                if w0 > 0:
                    pos_weight = torch.tensor(
                        [w1 / w0], dtype=torch.float32, device=device
                    )

        embed_dim = self._resolve_embed_dim(n_features, n_samples)
        proj_weight = torch.empty(n_features, embed_dim, device=device)
        torch.nn.init.xavier_uniform_(proj_weight)
        proj_weight.requires_grad_(True)
        clf_weight = torch.zeros(embed_dim, device=device, requires_grad=True)
        clf_bias = torch.zeros(1, device=device, requires_grad=True)

        optimizer = torch.optim.Adam([proj_weight, clf_weight, clf_bias], lr=self.lr)
        reg_strength = 1.0 / max(self.C, 1e-12)
        bce_loss = None
        if self.classification_loss == "ce":
            bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")

        prev_loss = None
        for _ in range(self.max_iter):
            embeddings = F.normalize(x_tensor @ proj_weight, dim=1)
            logits = embeddings @ clf_weight + clf_bias

            if self.classification_loss == "focal":
                cls_loss = binary_focal_loss(
                    logits.reshape(-1),
                    y_tensor,
                    gamma=self.focal_gamma,
                    alpha=self.focal_alpha,
                    sample_weights=sample_weights,
                )
            else:
                cls_loss = bce_loss(logits.reshape(-1), y_tensor)
            supcon = supervised_contrastive_loss(
                embeddings,
                y_tensor,
                temperature=self.temperature,
            )
            reg = 0.5 * reg_strength * (
                proj_weight.pow(2).sum()
                + clf_weight.pow(2).sum()
                + clf_bias.pow(2).sum()
            )
            loss = cls_loss + self.alpha * supcon + reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = float(loss.item())
            if prev_loss is not None and abs(prev_loss - loss_value) < self.tol:
                break
            prev_loss = loss_value

        self.proj_weight_ = proj_weight.detach().cpu().numpy()
        self.coef_ = clf_weight.detach().cpu().numpy().reshape(1, -1)
        self.intercept_ = clf_bias.detach().cpu().numpy()
        self.n_features_in_ = n_features
        self.embed_dim_ = embed_dim
        self.classes_ = np.array([0, 1], dtype=int)
        if self.classification_loss == "focal":
            self.loss_type_ = "focal_plus_supcon"
        else:
            self.loss_type_ = "ce_plus_supcon"
        return self

    def _embed(self, X):
        X = np.asarray(X, dtype=np.float32)
        z = X @ self.proj_weight_
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return z / norms

    def decision_function(self, X):
        embeddings = self._embed(X)
        return (embeddings @ self.coef_.T + self.intercept_).reshape(-1)

    def predict_proba(self, X):
        scores = self.decision_function(X)
        prob_pos = 1.0 / (1.0 + np.exp(-scores))
        prob_neg = 1.0 - prob_pos
        return np.column_stack([prob_neg, prob_pos])

    def predict(self, X):
        scores = self.decision_function(X)
        return (scores >= 0.0).astype(int)
