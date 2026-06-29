import re
from pathlib import Path

import pandas as pd


def _register_pdb_path(available, path):
    stem = path.stem
    upper_stem = stem.upper()

    if upper_stem.startswith("AF-"):
        match = re.match(r"AF-([^-]+)-F1-model", stem, flags=re.IGNORECASE)
        if match:
            available.add(match.group(1).strip())
            return
    available.add(stem.strip())


def build_pdb_available_accessions(pdb_dir):
    pdb_dir = Path(pdb_dir)
    if not pdb_dir.exists():
        raise FileNotFoundError(f"PDB directory not found: {pdb_dir}")

    available = set()
    for pattern in ("*.pdb", "*.PDB"):
        for path in pdb_dir.glob(pattern):
            _register_pdb_path(available, path)

    return available


def find_pdb_file(pdb_dir, acc):
    pdb_dir = Path(pdb_dir)
    acc = str(acc).strip()

    candidates = [
        pdb_dir / f"{acc}.pdb",
        pdb_dir / f"{acc}.PDB",
        pdb_dir / f"AF-{acc}-F1-model_v4.pdb",
        pdb_dir / f"AF-{acc}-F1-model_v3.pdb",
        pdb_dir / f"AF-{acc}-F1-model_v2.pdb",
        pdb_dir / f"AF-{acc}-F1-model_v1.pdb",
    ]

    for path in candidates:
        if path.exists():
            return path

    matches = sorted(pdb_dir.glob(f"*{acc}*.pdb"))
    if matches:
        return matches[0]

    matches = sorted(pdb_dir.glob(f"*{acc}*.PDB"))
    if matches:
        return matches[0]

    return None


def parse_pdb_residue_positions(pdb_path):
    positions = set()

    with open(pdb_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue

            resseq = line[22:26].strip()
            if not resseq:
                continue

            try:
                positions.add(int(resseq))
            except ValueError:
                continue

    return positions


def _build_acc_to_pdb_positions(pdb_dir, acc_ids):
    pdb_dir = Path(pdb_dir)
    acc_to_positions = {}

    for acc in acc_ids:
        acc = str(acc).strip()
        pdb_path = find_pdb_file(pdb_dir, acc)
        if pdb_path is None:
            acc_to_positions[acc] = None
            continue
        acc_to_positions[acc] = parse_pdb_residue_positions(pdb_path)

    return acc_to_positions


def filter_dataframe_by_pdb(
    df,
    pdb_dir,
    acc_col="ACC_ID",
    site_col="POSITION",
    label_col="LABEL",
    check_site_positions=True,
):
    df = df.copy()
    df[acc_col] = df[acc_col].astype(str).str.strip()
    df[site_col] = pd.to_numeric(df[site_col], errors="coerce")

    before_n = len(df)
    before_acc = df[acc_col].nunique()

    available_accs = build_pdb_available_accessions(pdb_dir)
    has_pdb_file = df[acc_col].isin(available_accs)

    acc_ids = sorted(df.loc[has_pdb_file, acc_col].unique())
    acc_to_positions = _build_acc_to_pdb_positions(pdb_dir, acc_ids)

    def _site_in_pdb(row):
        acc = row[acc_col]
        pos = row[site_col]
        if pd.isna(pos):
            return False

        positions = acc_to_positions.get(acc)
        if positions is None:
            return False
        return int(pos) in positions

    if check_site_positions:
        site_valid = df.apply(_site_in_pdb, axis=1)
        keep_mask = has_pdb_file & site_valid
    else:
        keep_mask = has_pdb_file

    filtered = df.loc[keep_mask].copy().reset_index(drop=True)

    after_n = len(filtered)
    after_acc = filtered[acc_col].nunique()

    removed_no_pdb_file = int((~has_pdb_file).sum())
    removed_site_missing = int((has_pdb_file & ~keep_mask).sum()) if check_site_positions else 0

    msg = (
        f"[INFO] PDB filter ({pdb_dir}): "
        f"rows before={before_n} after={after_n} removed={before_n - after_n}; "
        f"ACC_ID before={before_acc} after={after_acc}; "
        f"removed_no_pdb_file={removed_no_pdb_file} "
        f"removed_site_not_in_pdb={removed_site_missing}"
    )

    if label_col in df.columns:
        pos_before = int((df[label_col] == 1).sum())
        pos_after = int((filtered[label_col] == 1).sum())
        u_before = int((df[label_col] == 0).sum())
        u_after = int((filtered[label_col] == 0).sum())
        msg += (
            f"; P before={pos_before} after={pos_after}; "
            f"U before={u_before} after={u_after}"
        )

        if pos_after < pos_before:
            dropped_pos = df.loc[(df[label_col] == 1) & (~keep_mask), [acc_col, site_col]].head(10)
            print(
                f"[WARN] Dropped {pos_before - pos_after} positive sites failing PDB filter. "
                f"Examples:\n{dropped_pos.to_string(index=False)}"
            )

    print(msg)
    return filtered
