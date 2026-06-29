from src.models.esm_cnn2d_site_classifier import ESMCNN2DSiteClassifier
from src.models.esm_cnn2d_site_gnn_classifier import ESMCNN2DSiteGNNClassifier


def build_model(model_name, model_cfg, input_dims=None):
    if model_name == "esm_cnn2d_site":
        esm_input_dim = model_cfg.get("input_dim", None)
        if esm_input_dim is None and input_dims is not None and "esm" in input_dims:
            esm_input_dim = input_dims["esm"][-1]
        if esm_input_dim is None:
            esm_input_dim = 1280

        return ESMCNN2DSiteClassifier(
            input_dim=esm_input_dim,
            proj_input_dim=model_cfg.get("proj_input_dim", 512),
            conv_channels=tuple(model_cfg.get("conv_channels", [256, 128])),
            seq_kernel_size=model_cfg.get("seq_kernel_size", 3),
            encoder_output_dim=model_cfg.get("encoder_output_dim", 64),
            site_hidden_dims=tuple(model_cfg.get("site_hidden_dims", [512, 256])),
            site_output_dim=model_cfg.get("site_output_dim", 64),
            fusion_hidden_dim=model_cfg.get("fusion_hidden_dim", 32),
            dropout=model_cfg.get("dropout", 0.3),
        )

    if model_name == "esm_cnn2d_site_gnn":
        esm_input_dim = model_cfg.get("input_dim", None)
        if esm_input_dim is None and input_dims is not None and "esm" in input_dims:
            esm_input_dim = input_dims["esm"][-1]
        if esm_input_dim is None:
            esm_input_dim = 1280

        gnn_node_input_dim = model_cfg.get("gnn_node_input_dim", None)
        if gnn_node_input_dim is None and input_dims is not None and "af_graph" in input_dims:
            gnn_node_input_dim = input_dims["af_graph"][-1]
        if gnn_node_input_dim is None:
            gnn_node_input_dim = 23

        return ESMCNN2DSiteGNNClassifier(
            input_dim=esm_input_dim,
            proj_input_dim=model_cfg.get("proj_input_dim", 512),
            conv_channels=tuple(model_cfg.get("conv_channels", [256, 128])),
            seq_kernel_size=model_cfg.get("seq_kernel_size", 3),
            encoder_output_dim=model_cfg.get("encoder_output_dim", 64),
            site_hidden_dims=tuple(model_cfg.get("site_hidden_dims", [512, 256])),
            site_output_dim=model_cfg.get("site_output_dim", 64),
            gnn_node_input_dim=gnn_node_input_dim,
            gnn_hidden_dim=model_cfg.get("gnn_hidden_dim", 64),
            gnn_output_dim=model_cfg.get("gnn_output_dim", 64),
            fusion_hidden_dim=model_cfg.get("fusion_hidden_dim", 32),
            dropout=model_cfg.get("dropout", 0.3),
        )

    raise ValueError(f"Unsupported model: {model_name}")