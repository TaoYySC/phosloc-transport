import torch
import torch.nn as nn


def run_classifier_with_repr(classifier, z, n_repr_layers=2):
    """Run classifier head and return logits plus last *n* ReLU activations."""
    repr_layers = []
    x = z

    for module in classifier:
        x = module(x)
        if isinstance(module, nn.ReLU):
            repr_layers.append(x)

    selected = repr_layers[-n_repr_layers:]
    start_idx = len(repr_layers) - len(selected) + 1

    out = {"logits": x.squeeze(-1) if x.dim() > 1 and x.size(-1) == 1 else x}
    for offset, activation in enumerate(selected):
        out[f"classifier_l{start_idx + offset}"] = activation

    return out


def forward_with_repr(model, features, n_repr_layers=2):
    if hasattr(model, "forward_with_repr"):
        return model.forward_with_repr(features, n_repr_layers=n_repr_layers)

    if hasattr(model, "encode") and hasattr(model, "classifier"):
        z = model.encode(features)
        return run_classifier_with_repr(model.classifier, z, n_repr_layers=n_repr_layers)

    raise TypeError(
        f"{type(model).__name__} does not support classifier representation extraction."
    )
