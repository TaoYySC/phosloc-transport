from pathlib import Path

import numpy as np
import torch
from Bio.PDB import PDBParser
from torch_geometric.data import Data


AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}
UNK_IDX = len(AA_VOCAB)


def one_hot_aa(aa):
    x = np.zeros(len(AA_VOCAB) + 1, dtype=np.float32)
    idx = AA_TO_IDX.get(str(aa).upper(), UNK_IDX)
    x[idx] = 1.0
    return x


class AlphaFoldGraphLoader:
    def __init__(
        self,
        pdb_dir,
        acc_col="ACC_ID",
        pos_col="POSITION",
        radius=10.0,
        max_nodes=64,
        use_plddt=True,
    ):
        self.pdb_dir = Path(pdb_dir)
        self.acc_col = acc_col
        self.pos_col = pos_col
        self.radius = float(radius)
        self.max_nodes = int(max_nodes)
        self.use_plddt = bool(use_plddt)
        self.parser = PDBParser(QUIET=True)
        self.cache = {}

    def _get_pdb_path(self, acc):
        acc = str(acc).strip()
        p1 = self.pdb_dir / f"{acc}.pdb"
        p2 = self.pdb_dir / f"AF-{acc}-F1-model_v4.pdb"
        if p1.exists():
            return p1
        if p2.exists():
            return p2
        raise FileNotFoundError(f"PDB file not found for {acc}")

    def _load_structure(self, acc):
        if acc in self.cache:
            return self.cache[acc]

        path = self._get_pdb_path(acc)
        structure = self.parser.get_structure(acc, str(path))

        residues = []
        for model in structure:
            for chain in model:
                for res in chain:
                    if "CA" not in res:
                        continue
                    hetflag, resseq, icode = res.id
                    if hetflag.strip():
                        continue
                    aa = res.get_resname()
                    aa1 = self._resname_to_aa1(aa)
                    ca = res["CA"].coord.astype(np.float32)
                    plddt = float(res["CA"].bfactor)
                    residues.append(
                        {
                            "resseq": int(resseq),
                            "aa": aa1,
                            "coord": ca,
                            "plddt": plddt,
                        }
                    )

        residues = sorted(residues, key=lambda x: x["resseq"])
        self.cache[acc] = residues
        return residues

    def _resname_to_aa1(self, resname):
        table = {
            "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
            "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
            "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
            "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
        }
        return table.get(str(resname).upper(), "X")

    def _build_graph(self, acc, pos):
        residues = self._load_structure(acc)
        pos = int(pos)

        center_idx = None
        for i, r in enumerate(residues):
            if r["resseq"] == pos:
                center_idx = i
                break
        if center_idx is None:
            raise ValueError(f"Residue position {pos} not found in PDB for {acc}")

        center_coord = residues[center_idx]["coord"]

        selected = []
        for r in residues:
            dist = float(np.linalg.norm(r["coord"] - center_coord))
            if dist <= self.radius:
                selected.append((r, dist))

        selected = sorted(selected, key=lambda x: x[1])[: self.max_nodes]
        selected_residues = [x[0] for x in selected]

        x_list = []
        coords = []
        center_mask = []

        for r in selected_residues:
            feat = [one_hot_aa(r["aa"])]
            if self.use_plddt:
                feat.append(np.array([r["plddt"] / 100.0], dtype=np.float32))
            feat.append(np.array([(r["resseq"] - pos) / 50.0], dtype=np.float32))
            x_list.append(np.concatenate(feat, axis=0))
            coords.append(r["coord"])
            center_mask.append(1.0 if r["resseq"] == pos else 0.0)

        x = np.stack(x_list, axis=0)
        coords = np.stack(coords, axis=0)
        center_mask = np.asarray(center_mask, dtype=np.float32)

        edge_index = []
        edge_attr = []

        n = len(selected_residues)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dist = float(np.linalg.norm(coords[i] - coords[j]))
                seq_gap = abs(selected_residues[i]["resseq"] - selected_residues[j]["resseq"])
                if dist <= self.radius or seq_gap == 1:
                    edge_index.append([i, j])
                    edge_attr.append([dist / self.radius, float(seq_gap == 1)])

        if len(edge_index) == 0:
            edge_index = np.array([[0], [0]], dtype=np.int64)
            edge_attr = np.array([[0.0, 0.0]], dtype=np.float32)
        else:
            edge_index = np.asarray(edge_index, dtype=np.int64).T
            edge_attr = np.asarray(edge_attr, dtype=np.float32)

        data = Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            center_mask=torch.tensor(center_mask, dtype=torch.float32),
        )
        return data

    def load(self, df):
        graphs = []
        for acc, pos in zip(df[self.acc_col].tolist(), df[self.pos_col].tolist()):
            graphs.append(self._build_graph(acc, pos))
        return graphs