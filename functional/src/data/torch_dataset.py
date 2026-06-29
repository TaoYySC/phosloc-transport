import numpy as np
import torch
from torch.utils.data import Dataset


class MultiInputDataset(Dataset):
    def __init__(self, feature_dict, y):
        self.features = {}
        self.y = np.asarray(y, dtype=np.float32)

        n = len(self.y)
        for name, value in feature_dict.items():
            if value is None:
                continue

            if isinstance(value, list):
                if len(value) != n:
                    raise ValueError(f"{name} and y have different lengths.")
                self.features[name] = value
            else:
                arr = np.asarray(value, dtype=np.float32)
                if len(arr) != n:
                    raise ValueError(f"{name} and y have different lengths.")
                self.features[name] = arr

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        item = {
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
        }

        for name, val in self.features.items():
            if isinstance(val, list):
                item[name] = val[idx]
            else:
                item[name] = torch.tensor(val[idx], dtype=torch.float32)

        return item