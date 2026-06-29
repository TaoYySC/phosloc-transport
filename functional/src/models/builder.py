from src.models.dual_branch_cnn import DualBranchCNNClassifier
from src.models.tabular_dual_branch_cnn import TabularDualBranchCNNClassifier


def build_model(model_name, model_cfg, input_dims):
    if model_name == "dual_cnn":
        return DualBranchCNNClassifier(
            onehot_input_dim=input_dims["onehot"][-1],
            esm_input_dim=input_dims["esm"][-1],
            onehot_channels=tuple(model_cfg.get("onehot_channels", [64, 128])),
            esm_channels=tuple(model_cfg.get("esm_channels", [128, 256])),
            kernel_size=model_cfg.get("kernel_size", 3),
            onehot_proj_dim=model_cfg.get("onehot_proj_dim", 64),
            esm_proj_dim=model_cfg.get("esm_proj_dim", 128),
            fusion_hidden_dims=tuple(model_cfg.get("fusion_hidden_dims", [128, 64])),
            dropout=model_cfg.get("dropout", 0.3),
        )

    if model_name == "tabular_dual_cnn":
        return TabularDualBranchCNNClassifier(
            tabular_input_dim=input_dims["tabular"][-1],
            onehot_input_dim=input_dims["onehot"][-1],
            esm_input_dim=input_dims["esm"][-1],
            onehot_channels=tuple(model_cfg.get("onehot_channels", [64, 128])),
            esm_channels=tuple(model_cfg.get("esm_channels", [128, 256])),
            kernel_size=model_cfg.get("kernel_size", 3),
            onehot_proj_dim=model_cfg.get("onehot_proj_dim", 64),
            esm_proj_dim=model_cfg.get("esm_proj_dim", 128),
            tab_hidden_dims=tuple(model_cfg.get("tab_hidden_dims", [128, 64])),
            fusion_hidden_dims=tuple(model_cfg.get("fusion_hidden_dims", [128, 64])),
            dropout=model_cfg.get("dropout", 0.3),
        )

    raise ValueError(f"Unsupported model: {model_name}")