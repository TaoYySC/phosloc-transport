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
    ):
        self.embedding_dir = Path(embedding_dir)
        self.acc_col = acc_col
        self.pos_col = pos_col
        self.window_size = int(window_size)
        self.embedding_dim = int(embedding_dim)
        self.half = self.window_size // 2
        self.cache = {}

    def _load_embedding_file(self, acc_id):
        acc_id = str(acc_id).strip()

        if acc_id in self.cache:
            return self.cache[acc_id]

        pt_path = self.embedding_dir / f"{acc_id}.pt"
        npy_path = self.embedding_dir / f"{acc_id}.npy"

        if pt_path.exists():
            obj = torch.load(pt_path, map_location="cpu")
            if isinstance(obj, dict):
                if "embedding" in obj:
                    arr = obj["embedding"]
                elif "representations" in obj:
                    arr = next(iter(obj["representations"].values()))
                else:
                    raise ValueError(f"Unsupported dict format in {pt_path}")
            else:
                arr = obj

            if isinstance(arr, torch.Tensor):
                arr = arr.detach().cpu().numpy()

        elif npy_path.exists():
            arr = np.load(npy_path)

        else:
            raise FileNotFoundError(f"Missing ESM embedding file for {acc_id}")

        arr = np.asarray(arr, dtype=np.float32)

        if arr.ndim != 2:
            raise ValueError(f"Expected 2D embedding for {acc_id}, got shape {arr.shape}")
        if arr.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dim {self.embedding_dim} for {acc_id}, got {arr.shape[1]}"
            )

        self.cache[acc_id] = arr
        return arr

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
            residue_embedding = self._load_embedding_file(acc_id)
            mats.append(self._extract_window(residue_embedding, pos))
        return np.stack(mats, axis=0)
