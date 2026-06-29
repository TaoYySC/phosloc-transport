import torch
import torch.nn as nn
import torch.nn.functional as F


def _reduce_mean(values):
    if values.numel() == 0:
        return values.new_zeros(())
    return values.mean()


def _positive_loss(logits):
    logits = logits.reshape(-1)
    return F.softplus(-logits)


def _negative_loss(logits):
    logits = logits.reshape(-1)
    return F.softplus(logits)


class NNPULoss(nn.Module):
    """Non-negative PU risk (Kiryo et al., 2017).

    Training labels:
      - 1.0 for labeled positives (P)
      - 0.0 for unlabeled samples (U)

    U samples are not treated as true negatives; they only contribute through
    the unlabeled negative risk term R_U-.
    """

    def __init__(self, class_prior=0.1, u_loss_weight=1.0, non_negative=True):
        super().__init__()
        self.class_prior = float(class_prior)
        self.u_loss_weight = float(u_loss_weight)
        self.non_negative = bool(non_negative)

    def forward(self, logits, y):
        y = y.reshape(-1)
        logits = logits.reshape(-1)

        is_positive = y > 0.5
        is_unlabeled = ~is_positive

        rp = _reduce_mean(_positive_loss(logits[is_positive]))
        rn_on_p = _reduce_mean(_negative_loss(logits[is_positive]))
        rn_on_u = _reduce_mean(_negative_loss(logits[is_unlabeled]))

        pi = self.class_prior
        u_term = self.u_loss_weight * rn_on_u
        corrected_u = u_term - pi * rn_on_p

        if self.non_negative:
            pu_risk = pi * rp + torch.clamp(corrected_u, min=0.0)
        else:
            pu_risk = pi * rp + corrected_u

        return pu_risk


class WeightedPULoss(nn.Module):
    """Weighted logistic loss for PU learning.

    P samples use standard positive/negative logistic terms with weight 1.
    U samples only contribute a down-weighted negative logistic term.
    """

    def __init__(self, u_loss_weight=0.1, positive_loss_weight=1.0):
        super().__init__()
        self.u_loss_weight = float(u_loss_weight)
        self.positive_loss_weight = float(positive_loss_weight)

    def forward(self, logits, y):
        y = y.reshape(-1)
        logits = logits.reshape(-1)

        is_positive = y > 0.5
        is_unlabeled = ~is_positive

        loss = logits.new_zeros(())
        if is_positive.any():
            pos_logits = logits[is_positive]
            loss = loss + self.positive_loss_weight * (
                _reduce_mean(_positive_loss(pos_logits))
                + _reduce_mean(_negative_loss(pos_logits))
            ) * 0.5

        if is_unlabeled.any():
            loss = loss + self.u_loss_weight * _reduce_mean(
                _negative_loss(logits[is_unlabeled])
            )

        return loss


class BCEWithLogitsLossWrapper(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        if pos_weight is None:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, y):
        return self.criterion(logits.reshape(-1), y.reshape(-1))


def build_binary_loss(
    loss_type="bce",
    pos_weight=None,
    class_prior=0.1,
    u_loss_weight=1.0,
    non_negative=True,
    positive_loss_weight=1.0,
):
    loss_type = str(loss_type).lower()

    if loss_type in {"bce", "standard", "supervised"}:
        return BCEWithLogitsLossWrapper(pos_weight=pos_weight)

    if loss_type in {"nnpu", "nn_pu"}:
        return NNPULoss(
            class_prior=class_prior,
            u_loss_weight=u_loss_weight,
            non_negative=non_negative,
        )

    if loss_type in {"upu", "u_pu"}:
        return NNPULoss(
            class_prior=class_prior,
            u_loss_weight=u_loss_weight,
            non_negative=False,
        )

    if loss_type in {"weighted_pu", "pu_weighted", "pu_bce"}:
        return WeightedPULoss(
            u_loss_weight=u_loss_weight,
            positive_loss_weight=positive_loss_weight,
        )

    raise ValueError(
        f"Unsupported loss_type: {loss_type}. "
        "Expected one of: bce, nnpu, upu, weighted_pu"
    )
