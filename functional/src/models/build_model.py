from src.models.onehot_cnn_classifier import OneHotCNNClassifier
from src.models.onehot_cnn2d_classifier import OneHotCNN2DClassifier
from src.models.esm_cnn_classifier import ESMCNNClassifier
from src.models.esm_cnn2d_classifier import ESMCNN2DClassifier
from src.models.esm_cnn2d_site_classifier import ESMCNN2DSiteClassifier
from src.models.esm_cnn2d_site_gnn_classifier import ESMCNN2DSiteGNNClassifier


def build_model(model_name, model_cfg, input_dims=None):
    if model_name == "onehot_cnn":
        return OneHotCNNClassifier(
            input_dim=model_cfg.get("input_dim", 21),
            conv_channels=tuple(model_cfg.get("conv_channels", [64, 128])),
            kernel_size=model_cfg.get("kernel_size", 3),
            encoder_output_dim=model_cfg.get("encoder_output_dim", 64),
            hidden_dim=model_cfg.get("hidden_dim", 64),
            dropout=model_cfg.get("dropout", 0.2),
        )

    if model_name == "onehot_cnn2d":
        return OneHotCNN2DClassifier(
            input_dim=model_cfg.get("input_dim", 21),
            conv_channels=tuple(model_cfg.get("conv_channels", [32, 64])),
            seq_kernel_size=model_cfg.get("seq_kernel_size", 3),
            encoder_output_dim=model_cfg.get("encoder_output_dim", 64),
            hidden_dim=model_cfg.get("hidden_dim", 64),
            dropout=model_cfg.get("dropout", 0.2),
        )

    if model_name == "esm_cnn":
        esm_input_dim = model_cfg.get("input_dim", None)
        if esm_input_dim is None and input_dims is not None and "esm" in input_dims:
            esm_input_dim = input_dims["esm"][-1]
        if esm_input_dim is None:
            esm_input_dim = 1280

        return ESMCNNClassifier(
            input_dim=esm_input_dim,
            conv_channels=tuple(model_cfg.get("conv_channels", [128, 256])),
            kernel_size=model_cfg.get("kernel_size", 3),
            encoder_output_dim=model_cfg.get("encoder_output_dim", 128),
            hidden_dim=model_cfg.get("hidden_dim", 128),
            dropout=model_cfg.get("dropout", 0.3),
        )

    if model_name == "esm_cnn2d":
        esm_input_dim = model_cfg.get("input_dim", None)
        if esm_input_dim is None and input_dims is not None and "esm" in input_dims:
            esm_input_dim = input_dims["esm"][-1]
        if esm_input_dim is None:
            esm_input_dim = 1280

        return ESMCNN2DClassifier(
            input_dim=esm_input_dim,
            proj_input_dim=model_cfg.get("proj_input_dim", 64),
            conv_channels=tuple(model_cfg.get("conv_channels", [32, 64])),
            seq_kernel_size=model_cfg.get("seq_kernel_size", 3),
            encoder_output_dim=model_cfg.get("encoder_output_dim", 64),
            hidden_dim=model_cfg.get("hidden_dim", 64),
            dropout=model_cfg.get("dropout", 0.3),
        )

    if model_name == "esm_cnn2d_site":
        esm_input_dim = model_cfg.get("input_dim", None)
        if esm_input_dim is None and input_dims is not None and "esm" in input_dims:
            esm_input_dim = input_dims["esm"][-1]
        if esm_input_dim is None:
            esm_input_dim = 1280

        return ESMCNN2DSiteClassifier(
            input_dim=esm_input_dim,
            proj_input_dim=model_cfg.get("proj_input_dim", 64),
            conv_channels=tuple(model_cfg.get("conv_channels", [32, 64])),
            seq_kernel_size=model_cfg.get("seq_kernel_size", 3),
            encoder_output_dim=model_cfg.get("encoder_output_dim", 64),
            site_hidden_dims=tuple(model_cfg.get("site_hidden_dims", [256, 128])),
            site_output_dim=model_cfg.get("site_output_dim", 64),
            fusion_hidden_dim=model_cfg.get("fusion_hidden_dim", 64),
            dropout=model_cfg.get("dropout", 0.3),
        )

    if model_name == "esm_cnn2d_site_gnn":
        esm_input_dim = model_cfg.get("input_dim", None)
        if esm_input_dim is None and input_dims is not None and "esm" in input_dims:
            esm_input_dim = input_dims["esm"][-1]
        if esm_input_dim is None:
            esm_input_dim = 1280

        graph_input_dim = model_cfg.get("gnn_node_input_dim", 23)

        return ESMCNN2DSiteGNNClassifier(
            input_dim=esm_input_dim,
            proj_input_dim=model_cfg.get("proj_input_dim", 64),
            conv_channels=tuple(model_cfg.get("conv_channels", [32, 64])),
            seq_kernel_size=model_cfg.get("seq_kernel_size", 3),
            encoder_output_dim=model_cfg.get("encoder_output_dim", 64),
            site_hidden_dims=tuple(model_cfg.get("site_hidden_dims", [256, 128])),
            site_output_dim=model_cfg.get("site_output_dim", 64),
            gnn_node_input_dim=graph_input_dim,
            gnn_hidden_dim=model_cfg.get("gnn_hidden_dim", 64),
            gnn_output_dim=model_cfg.get("gnn_output_dim", 64),
            fusion_hidden_dims=tuple(model_cfg.get("fusion_hidden_dims", [128, 64])),
            dropout=model_cfg.get("dropout", 0.3),
        )

    raise ValueError(f"Unsupported model: {model_name}")