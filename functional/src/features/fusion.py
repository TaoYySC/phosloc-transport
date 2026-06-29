import numpy as np
from sklearn.preprocessing import StandardScaler


class FeatureFusion:
    def __init__(self, scale_map=None):
        self.scale_map = scale_map or {}
        self.scalers = {}

    def _reshape_if_needed(self, x):
        if x.ndim == 2:
            return x
        return x.reshape(x.shape[0], -1)

    def fit_transform(self, feature_dict, selected_keys):
        parts = []

        for key in selected_keys:
            x = feature_dict[key]
            x = self._reshape_if_needed(x)

            if self.scale_map.get(key, False):
                scaler = StandardScaler()
                x = scaler.fit_transform(x)
                self.scalers[key] = scaler

            parts.append(x)

        if len(parts) == 0:
            raise ValueError("No feature blocks are provided.")

        return np.concatenate(parts, axis=1)

    def transform(self, feature_dict, selected_keys):
        parts = []

        for key in selected_keys:
            x = feature_dict[key]
            x = self._reshape_if_needed(x)

            if key in self.scalers:
                x = self.scalers[key].transform(x)

            parts.append(x)

        if len(parts) == 0:
            raise ValueError("No feature blocks are provided.")

        return np.concatenate(parts, axis=1)