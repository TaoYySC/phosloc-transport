from pathlib import Path
import numpy as np
import torch


class ESMWindowLoader:
    def __init__(
        self,
        embedding_dir,
        acc_col="ACC_ID",
        pos_col="POSITION",
        window_size=31,
        embedding_dim=1280,
        file_suffix=".pt",
    ):
        self.embedding_dir = Path(embedding_dir)
        self.acc_col = acc_col
        self.pos_col = pos_col
        self.window_size = int(window_size)
        self.half = self.window_size // 2
        self.embedding_dim = int(embedding_dim)
        self.file_suffix = file_suffix
        self.cache = {}

    def _load_embedding(self, acc_id):
        acc_id = str(acc_id).strip()
        if acc_id in self.cache:
            return self.cache[acc_id]

        path = self.embedding_dir / f"{acc_id}{self.file_suffix}"
        data = torch.load(path, map_location="cpu")

        if isinstance(data, dict):
            if "embedding" in data:
                data = data["embedding"]
            elif "representations" in data:
                data = next(iter(data["representations"].values()))
            elif "mean_representations" in data:
                raise ValueError(f"{path} contains only mean embedding, expected per residue embedding.")
            else:
                raise ValueError(f"Unsupported embedding format in {path}")

        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        data = np.asarray(data, dtype=np.float32)

        if data.ndim != 2:
            raise ValueError(f"Expected 2D residue embedding for {acc_id}, got shape {data.shape}")
        if data.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected embedding dim {self.embedding_dim}, got {data.shape[1]} for {acc_id}")

        self.cache[acc_id] = data
        return data

    def _extract_window(self, residue_embedding, pos):
        pos = int(pos) - 1
        out = np.zeros((self.window_size, self.embedding_dim), dtype=np.float32)

        for dst_i, src_i in enumerate(range(pos - self.half, pos + self.half + 1)):
            if 0 <= src_i < residue_embedding.shape[0]:
                out[dst_i] = residue_embedding[src_i]
        return out

    def load(self, df):
        mats = []
        for acc_id, pos in zip(df[self.acc_col].tolist(), df[self.pos_col].tolist()):
            residue_embedding = self._load_embedding(acc_id)
            mats.append(self._extract_window(residue_embedding, pos))
        return np.stack(mats, axis=0)