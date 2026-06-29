import torch
import torch.nn as nn
import torch.nn.functional as F


def build_binary_loss(pos_weight=None):
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def supervised_contrastive_loss(features, labels, temperature=0.07):
    """Supervised contrastive loss (Khosla et al., 2020)."""
    device = features.device
    labels = labels.reshape(-1, 1)
    n = features.shape[0]

    if n <= 1:
        return features.new_zeros(())

    similarity = torch.matmul(features, features.T) / float(temperature)
    logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
    logits = similarity - logits_max.detach()

    label_mask = torch.eq(labels, labels.T).float().to(device)
    self_mask = torch.eye(n, device=device, dtype=torch.float32)
    positive_mask = label_mask * (1.0 - self_mask)
    anchor_mask = 1.0 - self_mask

    exp_logits = torch.exp(logits) * anchor_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return features.new_zeros(())

    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_count.clamp(min=1.0)
    return -mean_log_prob_pos[valid].mean()


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, features, labels):
        return supervised_contrastive_loss(
            features,
            labels,
            temperature=self.temperature,
        )


def binary_focal_loss(
    logits,
    targets,
    gamma=2.0,
    alpha=0.25,
    sample_weights=None,
):
    """Binary focal loss on logits (Lin et al., 2017)."""
    targets = targets.reshape(-1).float()
    logits = logits.reshape(-1)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    focal_factor = (1.0 - p_t).pow(float(gamma))
    if alpha is not None:
        alpha_t = float(alpha) * targets + (1.0 - float(alpha)) * (1.0 - targets)
        focal_factor = alpha_t * focal_factor
    loss = focal_factor * bce
    if sample_weights is not None:
        loss = loss * sample_weights.reshape(-1)
    return loss.mean()


def projection_training_loss(
    logits,
    projections,
    labels,
    sample_weights=None,
    alpha=0.05,
    temperature=0.1,
):
    labels = labels.reshape(-1).long()
    ce = F.cross_entropy(logits, labels, reduction="none")
    if sample_weights is not None:
        ce = ce * sample_weights.reshape(-1)
    ce_loss = ce.mean()

    supcon_loss = supervised_contrastive_loss(
        projections,
        labels.float(),
        temperature=temperature,
    )
    total = ce_loss + float(alpha) * supcon_loss
    return total, ce_loss.detach(), supcon_loss.detach()
