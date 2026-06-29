from pathlib import Path

import numpy as np
import torch

from src.features.esm_window.loader import ESMWindowLoader


class ESMWindowSiteMeanLoader:
    """Per-site features: mean-pooled window embedding + center residue embedding.

    Output shape: (N, 2 * embedding_dim) = [window_mean, site_vec].
    """

    def __init__(
        self,
        embedding_dir,
        acc_col="ACC_ID",
        pos_col="POSITION",
        window_size=41,
        embedding_dim=1280,
        file_suffix=".pt",
    ):
        self.window_loader = ESMWindowLoader(
            embedding_dir=embedding_dir,
            acc_col=acc_col,
            pos_col=pos_col,
            window_size=window_size,
            embedding_dim=embedding_dim,
            file_suffix=file_suffix,
        )
        self.acc_col = acc_col
        self.pos_col = pos_col
        self.window_size = int(window_size)
        self.embedding_dim = int(embedding_dim)
        self.feature_names_ = (
            [f"window_mean_{i}" for i in range(self.embedding_dim)]
            + [f"site_{i}" for i in range(self.embedding_dim)]
        )

    def get_feature_names_out(self):
        return list(self.feature_names_)

    def load(self, df):
        windows = self.window_loader.load(df)
        center_idx = windows.shape[1] // 2
        window_mean = windows.mean(axis=1)
        site_vec = windows[:, center_idx, :]
        return np.concatenate([window_mean, site_vec], axis=1).astype(np.float32)
