import numpy as np
import torch

from src.features.sequence_cnn.loader import SequenceWindowLoader

class SequenceCNNBlock:
    def __init__(self, encoder, device="cuda", seq_col="FULL_SEQUENCE", pos_col="POSITION", window_size=31):
        self.encoder = encoder
        self.device = device
        self.loader = SequenceWindowLoader(
            seq_col=seq_col,
            pos_col=pos_col,
            window_size=window_size,
        )

    @torch.no_grad()
    def transform(self, df, batch_size=256):
        self.encoder.eval()
        self.encoder.to(self.device)

        x = self.loader.load(df)
        outputs = []

        for start in range(0, len(x), batch_size):
            batch = torch.tensor(x[start:start + batch_size], dtype=torch.float32, device=self.device)
            z = self.encoder(batch)
            outputs.append(z.cpu().numpy())

        return np.concatenate(outputs, axis=0)