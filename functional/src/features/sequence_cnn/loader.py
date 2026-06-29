import numpy as np

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}
UNK_IDX = len(AA_VOCAB)

class SequenceWindowLoader:
    def __init__(self, seq_col="FULL_SEQUENCE", pos_col="POSITION", window_size=31):
        self.seq_col = seq_col
        self.pos_col = pos_col
        self.window_size = int(window_size)
        self.half = self.window_size // 2
        self.num_tokens = len(AA_VOCAB) + 1

    def _extract_window(self, seq, pos):
        seq = str(seq)
        pos = int(pos) - 1

        chars = []
        for i in range(pos - self.half, pos + self.half + 1):
            if 0 <= i < len(seq):
                chars.append(seq[i].upper())
            else:
                chars.append("X")
        return chars

    def _one_hot_encode_window(self, chars):
        x = np.zeros((self.window_size, self.num_tokens), dtype=np.float32)
        for i, aa in enumerate(chars):
            idx = AA_TO_IDX.get(aa, UNK_IDX)
            x[i, idx] = 1.0
        return x

    def load(self, df):
        mats = []
        for seq, pos in zip(df[self.seq_col].tolist(), df[self.pos_col].tolist()):
            chars = self._extract_window(seq, pos)
            mats.append(self._one_hot_encode_window(chars))
        return np.stack(mats, axis=0)