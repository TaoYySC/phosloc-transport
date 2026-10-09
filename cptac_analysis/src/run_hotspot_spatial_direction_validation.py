#!/usr/bin/env python3
"""Four TF-aware validations for the fixed Mixed/Predicted-candidate hotspot catalog.

Analyses (thresholds fixed; do NOT retune 15/40/3/0.6):
  1) HC Predicted nearest-neighbor distance vs STY-matched TF-aware null
  2) Predicted→Known proximity on dual-evidence TFs vs null
  3) Literature Cluster enrichment vs cluster-blind hotspots
  4) Within-hotspot Import/Export coherence + Mixed Known-anchor concordance

Does not use final hotspot membership as evidence for analyses 1–2.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = (
    REPO
    / "import_export/results"
    / "tf_phos_site_FuncTransport_Direction_summary_with_PMID_annotation_v11_147pos_d3_platt.csv"
)
DEFAULT_HOTSPOT_DIR = (
    REPO
    / "cptac_analysis/results/import_target_regulation_hotspot_v11_147pos_d3_platt"
    / "hotspots_mixed_pred_filter"
)
DEFAULT_OUT = (
    REPO
    / "cptac_analysis/results"
    / "hotspot_spatial_direction_validation_v11_147pos_d3_platt"
)
DEFAULT_CLUSTER_XLSX = (
    REPO
    / "functional/data/dataset_phos_site"
    / "TF_localization_final_four_sheets.xlsx"
)
CLUSTER_SHEETS = ("Cluster Nuclear", "Cluster Cyto")

ADJ_MAX = 15
SPAN_MAX = 40
PRED_MIN_N = 3
PRED_MEAN_MIN = 0.6
N_PERM = 10_000
DIST_THRESHOLDS = (5, 10, 15, 20, 30, 40, 50)
RNG_SEED = 20260920

C_OBS = "#4DBBD5"
C_NULL = "#9e9e9e"
C_ENR = "#c47a3a"
C_TEXT = "#333333"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_position(site: object) -> float:
    m = re.fullmatch(r"[STY](\d+)", str(site).strip(), flags=re.IGNORECASE)
    return float(m.group(1)) if m else np.nan


def parse_residue(site: object) -> str:
    s = str(site).strip().upper()
    return s[0] if s and s[0] in "STY" else ""


def direction_label(loc: object) -> Optional[str]:
    text = str(loc).strip() if pd.notna(loc) else ""
    if text == "Nuclear accumulation":
        return "Import"
    if text == "Cytoplasmic redistribution":
        return "Export"
    return None


def empirical_p_left(obs: float, null: np.ndarray) -> float:
    """P(null <= obs) with +1 correction (smaller-is-stronger statistics)."""
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if len(null) == 0 or not np.isfinite(obs):
        return np.nan
    return float((np.sum(null <= obs) + 1) / (len(null) + 1))


def empirical_p_right(obs: float, null: np.ndarray) -> float:
    """P(null >= obs) with +1 correction (larger-is-stronger statistics)."""
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if len(null) == 0 or not np.isfinite(obs):
        return np.nan
    return float((np.sum(null >= obs) + 1) / (len(null) + 1))


def nearest_neighbor_distances(positions: np.ndarray) -> np.ndarray:
    """Per-site distance to nearest other site; NaN if <2 sites."""
    pos = np.asarray(positions, dtype=float)
    n = len(pos)
    if n < 2:
        return np.full(n, np.nan)
    order = np.argsort(pos)
    sorted_pos = pos[order]
    left = np.empty(n)
    right = np.empty(n)
    left[0] = np.inf
    right[-1] = np.inf
    left[1:] = sorted_pos[1:] - sorted_pos[:-1]
    right[:-1] = sorted_pos[1:] - sorted_pos[:-1]
    nn_sorted = np.minimum(left, right)
    out = np.empty(n)
    out[order] = nn_sorted
    return out


def sty_matched_sample(
    bg_pos: np.ndarray,
    bg_res: np.ndarray,
    target_res: Sequence[str],
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """Sample |target_res| background positions, matching S/T/Y counts when possible."""
    n = len(target_res)
    if n == 0 or len(bg_pos) < n:
        return None
    counts = {"S": 0, "T": 0, "Y": 0}
    for r in target_res:
        counts[r] = counts.get(r, 0) + 1

    chosen_idx: List[int] = []
    remaining = np.ones(len(bg_pos), dtype=bool)
    for res, need in counts.items():
        if need <= 0:
            continue
        pool = np.where(remaining & (bg_res == res))[0]
        if len(pool) >= need:
            pick = rng.choice(pool, size=need, replace=False)
            chosen_idx.extend(pick.tolist())
            remaining[pick] = False
        else:
            chosen_idx.extend(pool.tolist())
            remaining[pool] = False
            short = need - len(pool)
            pool2 = np.where(remaining)[0]
            if len(pool2) < short:
                return None
            pick = rng.choice(pool2, size=short, replace=False)
            chosen_idx.extend(pick.tolist())
            remaining[pick] = False

    if len(chosen_idx) != n:
        return None
    return bg_pos[np.asarray(chosen_idx, dtype=int)]


def cluster_sorted(g: pd.DataFrame, adj_max: int, span_max: int) -> List[pd.DataFrame]:
    g = g.sort_values(["position", "site"]).reset_index(drop=True)
    if g.empty:
        return []
    clusters: List[List[int]] = []
    cur = [0]
    for i in range(1, len(g)):
        gap = int(g.at[i, "position"] - g.at[cur[-1], "position"])
        span = int(g.at[i, "position"] - g.at[cur[0], "position"])
        if gap > adj_max or span > span_max:
            clusters.append(cur)
            cur = [i]
        else:
            cur.append(i)
    clusters.append(cur)
    return [g.iloc[ix].copy() for ix in clusters]


def classify_hotspot(mem: pd.DataFrame) -> Optional[str]:
    n_known = int(mem["is_known"].sum())
    n_pred = int(mem["is_predicted"].sum())
    pred_scores = mem.loc[mem["is_predicted"], "FuncTransport_score"]
    mean_pred = float(pred_scores.mean()) if len(pred_scores) else np.nan
    if n_known >= 1 and n_pred >= 1:
        return "Mixed evidence"
    if (
        n_known == 0
        and n_pred >= PRED_MIN_N
        and pd.notna(mean_pred)
        and mean_pred >= PRED_MEAN_MIN
    ):
        return "Predicted candidate"
    return None


def build_hotspots(anchors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for acc, g in anchors.groupby("protein_acc", sort=True):
        gene = str(g["gene_name"].iloc[0])
        for mem in cluster_sorted(g, ADJ_MAX, SPAN_MAX):
            htype = classify_hotspot(mem)
            if htype is None:
                continue
            sites = mem.sort_values("position")["site"].tolist()
            start, end = int(mem["position"].min()), int(mem["position"].max())
            pred_scores = mem.loc[mem["is_predicted"], "FuncTransport_score"]
            rows.append(
                {
                    "protein_acc": acc,
                    "gene_name": gene,
                    "start": start,
                    "end": end,
                    "span": end - start,
                    "n_sites": int(len(mem)),
                    "n_known": int(mem["is_known"].sum()),
                    "n_predicted": int(mem["is_predicted"].sum()),
                    "n_cluster": int(mem["is_literature_cluster"].sum())
                    if "is_literature_cluster" in mem.columns
                    else int(mem.get("is_cluster", pd.Series(False, index=mem.index)).sum()),
                    "mean_func_score": float(pred_scores.mean()) if len(pred_scores) else np.nan,
                    "max_func_score": float(mem["FuncTransport_score"].max()),
                    "hotspot_type": htype,
                    "sites": ";".join(sites),
                    "positions": ";".join(str(int(x)) for x in sorted(mem["position"].unique())),
                }
            )
    return pd.DataFrame(rows)


def can_join_hotspot(pos: int, hotspot_positions: Sequence[int], adj_max: int, span_max: int) -> bool:
    new_pos = sorted(set(int(x) for x in hotspot_positions) | {int(pos)})
    if len(new_pos) < 2:
        return False
    span = new_pos[-1] - new_pos[0]
    if span > span_max:
        return False
    gaps = np.diff(new_pos)
    return bool(np.all(gaps <= adj_max))


def save_fig(fig: plt.Figure, out_stem: Path) -> None:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def p_to_stars(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def format_p_label(p: float) -> str:
    if not np.isfinite(p):
        return "P = n/a"
    return f"P = {p:.1e}" if p < 0.001 else f"P = {p:.3f}"


def annotate_bar_pvalue(ax, x0: float, x1: float, y_top: float, p: float) -> float:
    """Draw significance bracket above two bars; return suggested y-axis top."""
    text = f"{p_to_stars(p)}\n{format_p_label(p)}"
    h = max(0.02, abs(y_top) * 0.06) if y_top > 0.2 else 0.025
    y0 = y_top + h * 0.4
    ax.plot([x0, x0, x1, x1], [y0, y0 + h * 0.4, y0 + h * 0.4, y0], color="#333333", lw=1.0)
    ax.text((x0 + x1) / 2, y0 + h * 0.55, text, ha="center", va="bottom", fontsize=9, color="#333333")
    return y0 + h * 2.0


# ---------------------------------------------------------------------------
# Master tables
# ---------------------------------------------------------------------------

def parse_phosphosite_token(value: object) -> Optional[str]:
    s = str(value).strip()
    m = re.fullmatch(r"p?([STY])(\d+)", s, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1).upper()}{m.group(2)}"
    m = re.search(r"([STY])(\d+)", s, flags=re.IGNORECASE)
    return f"{m.group(1).upper()}{m.group(2)}" if m else None


def load_literature_cluster_from_xlsx(xlsx: Path) -> pd.DataFrame:
    """Authoritative literature multi-site Cluster set: Cluster Nuclear + Cluster Cyto."""
    frames = []
    for sheet in CLUSTER_SHEETS:
        df = pd.read_excel(xlsx, sheet_name=sheet)
        df = df.copy()
        df["cluster_sheet"] = sheet
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame(
        {
            "TF": raw["TF gene"].astype(str).str.strip(),
            "UniProt": raw["UniProt ID"].astype(str).str.strip(),
            "Phosphosite_raw": raw["Phosphosite"].astype(str).str.strip(),
            "site": raw["Phosphosite"].map(parse_phosphosite_token),
            "Site_key": raw["Site key"].astype(str).str.strip(),
            "Cluster_sites": raw["Cluster sites"].astype(str).str.strip(),
            "Direction": raw.get("Direction", pd.Series("", index=raw.index)).astype(str),
            "PMID": raw["PMID"].astype(str).str.strip(),
            "cluster_sheet": raw["cluster_sheet"],
            "Kinase": raw.get("Kinase", pd.Series("", index=raw.index)).astype(str),
        }
    )
    out = out.dropna(subset=["site"]).copy()
    out["INDEX"] = out["UniProt"] + "_" + out["site"]
    # Prefer Site_key when well-formed
    sk = out["Site_key"].str.match(r"^[A-Z0-9]+_[STY]\d+$", na=False)
    out.loc[sk, "INDEX"] = out.loc[sk, "Site_key"]
    out = out.drop_duplicates(subset=["INDEX"]).reset_index(drop=True)
    out["position"] = out["site"].map(parse_position).astype("Int64")
    out["residue"] = out["site"].map(parse_residue)
    return out


def load_site_master(
    summary: Path,
    cluster_xlsx: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(summary)
    df["protein_acc"] = df["protein_acc"].astype(str).str.strip()
    df["gene_name"] = df["gene_name"].astype(str).str.strip()
    df["site"] = df["site"].astype(str).str.strip()
    df["evidence"] = df["evidence"].astype(str).str.strip()
    df["PMID"] = df["PMID"].astype(str).str.strip()
    df["position"] = df["site"].map(parse_position)
    df["residue"] = df["site"].map(parse_residue)
    df = df.dropna(subset=["position", "residue"]).copy()
    df["position"] = df["position"].astype(int)
    df["FuncTransport_score"] = pd.to_numeric(df["FuncTransport_score"], errors="coerce")
    df["Direction_score"] = pd.to_numeric(df["Direction_score"], errors="coerce")
    df["is_known"] = ~df["PMID"].isin(["", "-", "nan", "None", "NaN"])
    df["is_predicted"] = df["evidence"].str.contains("Predicted", regex=False)
    df["is_hc_predicted"] = df["evidence"].isin({"Predicted", "Predicted;Cluster"})
    df["is_observed_background"] = True
    df["predicted_direction"] = df["Localization_annotation"].map(direction_label)
    df["INDEX"] = df["protein_acc"] + "_" + df["site"]

    cluster_path = Path(cluster_xlsx) if cluster_xlsx is not None else DEFAULT_CLUSTER_XLSX
    cluster_cat = load_literature_cluster_from_xlsx(cluster_path)
    cluster_idx = set(cluster_cat["INDEX"])
    df["is_literature_cluster"] = df["INDEX"].isin(cluster_idx)
    # Keep evidence-based flag for diagnostics
    df["is_literature_cluster_from_evidence"] = df["evidence"].str.contains("Cluster", regex=False)
    return df.reset_index(drop=True), cluster_cat


def load_fixed_hotspots(hotspot_dir: Path, site_master: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    hs = pd.read_csv(hotspot_dir / "hotspots_d15.csv")
    mem = pd.read_csv(hotspot_dir / "hotspot_members_d15.csv")
    assert len(hs) == 99, f"Expected 99 hotspots, got {len(hs)}"
    mixed = int((hs["hotspot_type"] == "Mixed evidence").sum())
    pred = int((hs["hotspot_type"] == "Predicted candidate").sum())
    assert mixed == 48 and pred == 51, f"Expected 48/51, got {mixed}/{pred}"

    catalog = pd.DataFrame(
        {
            "hotspot_id": hs["hotspot_id"],
            "TF": hs["gene_name"],
            "UniProt": hs["protein_acc"],
            "start": hs["start_position"],
            "end": hs["end_position"],
            "span": hs["span_aa"],
            "n_sites": hs["n_members"],
            "n_known": hs["n_known"],
            "n_predicted": hs["n_hc"],
            "n_cluster": hs["n_cluster"],
            "mean_func_score": hs["mean_FuncTransport_score"],
            "max_func_score": hs["max_FuncTransport_score"],
            "hotspot_type": hs["hotspot_type"],
            "sites": hs["member_sites"],
            "direction_call": hs["direction_call"],
        }
    )

    # Enrich members with master direction labels
    master_dir = site_master.set_index("INDEX")["predicted_direction"]
    mem = mem.copy()
    mem["predicted_direction"] = mem["INDEX"].map(master_dir)
    mem.loc[mem["predicted_direction"].isna(), "predicted_direction"] = mem["Localization_annotation"].map(
        direction_label
    )
    return catalog, mem


# ---------------------------------------------------------------------------
# Analysis 1
# ---------------------------------------------------------------------------

def run_analysis1(sites: pd.DataFrame, out_dir: Path, n_perm: int, rng: np.random.Generator) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = sites.loc[sites["is_hc_predicted"]].copy()
    site_rows = []
    obs_nnds: List[float] = []

    for acc, g in pred.groupby("protein_acc"):
        positions = g["position"].to_numpy(dtype=float)
        nnds = nearest_neighbor_distances(positions)
        gene = str(g["gene_name"].iloc[0])
        for i, (_, row) in enumerate(g.iterrows()):
            nnd = float(nnds[i]) if np.isfinite(nnds[i]) else np.nan
            site_rows.append(
                {
                    "INDEX": row["INDEX"],
                    "TF": gene,
                    "UniProt": acc,
                    "site": row["site"],
                    "position": int(row["position"]),
                    "residue": row["residue"],
                    "FuncTransport_score": row["FuncTransport_score"],
                    "n_pred_on_TF": int(len(g)),
                    "nearest_pred_distance": nnd,
                }
            )
            if np.isfinite(nnd):
                obs_nnds.append(nnd)

    site_df = pd.DataFrame(site_rows)
    site_df.to_csv(out_dir / "analysis1_site_level_nnd.csv", index=False)
    obs_nnds_arr = np.asarray(obs_nnds, dtype=float)
    obs_median = float(np.median(obs_nnds_arr)) if len(obs_nnds_arr) else np.nan

    # Precompute per-TF background and target residue lists
    tf_data = []
    for acc, g_pred in pred.groupby("protein_acc"):
        if len(g_pred) < 2:
            continue
        bg = sites.loc[sites["protein_acc"] == acc]
        tf_data.append(
            {
                "acc": acc,
                "target_res": g_pred["residue"].tolist(),
                "bg_pos": bg["position"].to_numpy(dtype=float),
                "bg_res": bg["residue"].to_numpy(),
                "n": len(g_pred),
            }
        )

    null_medians = np.empty(n_perm, dtype=float)
    null_frac = {d: np.empty(n_perm, dtype=float) for d in DIST_THRESHOLDS}
    # Collect pooled null NND samples for ECDF (subsample perms for memory)
    ecdf_pool: List[np.ndarray] = []
    ecdf_every = max(1, n_perm // 200)

    for p in range(n_perm):
        pooled: List[float] = []
        for td in tf_data:
            samp = sty_matched_sample(td["bg_pos"], td["bg_res"], td["target_res"], rng)
            if samp is None:
                continue
            nnds = nearest_neighbor_distances(samp)
            pooled.extend([float(x) for x in nnds if np.isfinite(x)])
        arr = np.asarray(pooled, dtype=float)
        null_medians[p] = float(np.median(arr)) if len(arr) else np.nan
        for d in DIST_THRESHOLDS:
            null_frac[d][p] = float(np.mean(arr <= d)) if len(arr) else np.nan
        if p % ecdf_every == 0 and len(arr):
            ecdf_pool.append(arr)

    null_median_mean = float(np.nanmean(null_medians))
    p_emp = empirical_p_left(obs_median, null_medians)

    obs_frac = {d: float(np.mean(obs_nnds_arr <= d)) for d in DIST_THRESHOLDS}
    enrich_rows = []
    for d in DIST_THRESHOLDS:
        null_mean = float(np.nanmean(null_frac[d]))
        enr = obs_frac[d] / null_mean if null_mean > 0 else np.nan
        p_d = empirical_p_right(obs_frac[d], null_frac[d])
        enrich_rows.append(
            {
                "distance_aa": d,
                "obs_fraction_nnd_le": obs_frac[d],
                "null_mean_fraction": null_mean,
                "enrichment": enr,
                "empirical_p_right": p_d,
            }
        )
    enrich_df = pd.DataFrame(enrich_rows)
    enrich_df.to_csv(out_dir / "analysis1_multiscale_enrichment.csv", index=False)

    summary = {
        "n_hc_predicted": int(len(pred)),
        "n_with_neighbor": int(len(obs_nnds_arr)),
        "n_TF_with_ge2_pred": int(len(tf_data)),
        "obs_median_nnd": obs_median,
        "null_mean_median_nnd": null_median_mean,
        "null_median_of_median_nnd": float(np.nanmedian(null_medians)),
        "enrichment_median_fold_decrease": (
            null_median_mean / obs_median if obs_median > 0 else np.nan
        ),
        "empirical_p_median_nnd_left": p_emp,
        "n_perm": n_perm,
        "thresholds": enrich_rows,
    }
    (out_dir / "analysis1_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame({"null_median_nnd": null_medians}).to_csv(
        out_dir / "analysis1_null_median_nnd.csv", index=False
    )

    # Figure: ECDF only (panel A style, no letter label)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    x_obs = np.sort(obs_nnds_arr)
    y_obs = np.arange(1, len(x_obs) + 1) / len(x_obs)
    ax.plot(x_obs, y_obs, color=C_OBS, lw=2.0, label="Observed HC Predicted")
    if ecdf_pool:
        grid = np.linspace(0, max(np.percentile(obs_nnds_arr, 99), 50), 200)
        null_ys = []
        for arr in ecdf_pool:
            null_ys.append(np.searchsorted(np.sort(arr), grid, side="right") / len(arr))
        mean_y = np.mean(null_ys, axis=0)
        ax.plot(grid, mean_y, color=C_NULL, lw=2.0, label="Null (mean ECDF)")
    ax.set_xlabel("Nearest-neighbor distance (aa)")
    ax.set_ylabel("ECDF")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlim(left=0)
    fig.tight_layout()
    save_fig(fig, out_dir / "figure_analysis1_nnd_ecdf_enrichment")

    return summary


# ---------------------------------------------------------------------------
# Analysis 2
# ---------------------------------------------------------------------------

def run_analysis2(sites: pd.DataFrame, catalog: pd.DataFrame, out_dir: Path, n_perm: int, rng: np.random.Generator) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Dual-evidence TFs
    by_tf = sites.groupby("protein_acc").agg(n_known=("is_known", "sum"), n_pred=("is_predicted", "sum"))
    dual = set(by_tf.index[(by_tf["n_known"] > 0) & (by_tf["n_pred"] > 0)])

    site_rows = []
    obs_dists: List[float] = []
    tf_data = []

    for acc in sorted(dual):
        g = sites.loc[sites["protein_acc"] == acc]
        known = g.loc[g["is_known"]]
        known_pos_all = known["position"].to_numpy(dtype=float)
        pred = g.loc[g["is_predicted"]]
        if len(known_pos_all) == 0 or len(pred) == 0:
            continue
        gene = str(g["gene_name"].iloc[0])
        for _, row in pred.iterrows():
            # Distance to nearest other Known (exclude self if Predicted;Cluster / dual-labeled)
            kp = known_pos_all[known_pos_all != float(row["position"])]
            if len(kp) == 0:
                continue
            d = float(np.min(np.abs(kp - float(row["position"]))))
            site_rows.append(
                {
                    "INDEX": row["INDEX"],
                    "TF": gene,
                    "UniProt": acc,
                    "site": row["site"],
                    "position": int(row["position"]),
                    "residue": row["residue"],
                    "FuncTransport_score": row["FuncTransport_score"],
                    "min_dist_to_known": d,
                }
            )
            obs_dists.append(d)
        # Keep Known fixed; sample Predicted from non-Known background
        pred_for_null = g.loc[g["is_predicted"] & ~g["is_known"]]
        if len(pred_for_null) == 0:
            pred_for_null = pred
        bg_non_known = g.loc[~g["is_known"]]
        if len(bg_non_known) < len(pred_for_null):
            continue
        known_pos = known_pos_all
        tf_data.append(
            {
                "acc": acc,
                "known_pos": known_pos,
                "target_res": pred_for_null["residue"].tolist(),
                "bg_pos": bg_non_known["position"].to_numpy(dtype=float),
                "bg_res": bg_non_known["residue"].to_numpy(),
                "n": len(pred_for_null),
            }
        )

    site_df = pd.DataFrame(site_rows)
    site_df.to_csv(out_dir / "analysis2_site_level_pred_to_known.csv", index=False)
    obs_arr = np.asarray(obs_dists, dtype=float)
    obs_median = float(np.median(obs_arr)) if len(obs_arr) else np.nan

    null_medians = np.empty(n_perm, dtype=float)
    null_frac = {d: np.empty(n_perm, dtype=float) for d in DIST_THRESHOLDS}
    ecdf_pool: List[np.ndarray] = []
    ecdf_every = max(1, n_perm // 200)

    for p in range(n_perm):
        pooled: List[float] = []
        for td in tf_data:
            samp = sty_matched_sample(td["bg_pos"], td["bg_res"], td["target_res"], rng)
            if samp is None:
                continue
            for pos in samp:
                pooled.append(float(np.min(np.abs(td["known_pos"] - pos))))
        arr = np.asarray(pooled, dtype=float)
        null_medians[p] = float(np.median(arr)) if len(arr) else np.nan
        for d in DIST_THRESHOLDS:
            null_frac[d][p] = float(np.mean(arr <= d)) if len(arr) else np.nan
        if p % ecdf_every == 0 and len(arr):
            ecdf_pool.append(arr)

    null_median_mean = float(np.nanmean(null_medians))
    p_emp = empirical_p_left(obs_median, null_medians)

    obs_frac = {d: float(np.mean(obs_arr <= d)) for d in DIST_THRESHOLDS}
    enrich_rows = []
    for d in DIST_THRESHOLDS:
        null_mean = float(np.nanmean(null_frac[d]))
        enr = obs_frac[d] / null_mean if null_mean > 0 else np.nan
        enrich_rows.append(
            {
                "distance_aa": d,
                "obs_fraction_le": obs_frac[d],
                "null_mean_fraction": null_mean,
                "enrichment": enr,
                "empirical_p_right": empirical_p_right(obs_frac[d], null_frac[d]),
            }
        )
    enrich_df = pd.DataFrame(enrich_rows)
    enrich_df.to_csv(out_dir / "analysis2_proximity_enrichment.csv", index=False)

    # Relation to 48 Mixed (descriptive only)
    mixed = catalog.loc[catalog["hotspot_type"] == "Mixed evidence"]
    mixed_pairs = 0
    for _, row in site_df.iterrows():
        hs_tf = mixed.loc[mixed["UniProt"] == row["UniProt"]]
        if hs_tf.empty:
            continue
        for _, h in hs_tf.iterrows():
            # predicted near a known that sits inside a Mixed hotspot span (descriptive)
            if int(h["start"]) - ADJ_MAX <= int(row["position"]) <= int(h["end"]) + ADJ_MAX:
                if row["min_dist_to_known"] <= ADJ_MAX:
                    mixed_pairs += 1
                    break

    summary = {
        "n_dual_TFs": int(len(tf_data)),
        "n_predicted_sites": int(len(obs_arr)),
        "obs_median_dist_to_known": obs_median,
        "null_mean_median_dist": null_median_mean,
        "null_median_of_median_dist": float(np.nanmedian(null_medians)),
        "enrichment_median_fold_decrease": (
            null_median_mean / obs_median if obs_median > 0 else np.nan
        ),
        "empirical_p_median_dist_left": p_emp,
        "n_perm": n_perm,
        "thresholds": enrich_rows,
        "n_mixed_hotspots": int(len(mixed)),
        "n_pred_sites_within_15aa_of_known_and_near_mixed_span": int(mixed_pairs),
        "note": (
            "Mixed hotspot counts are descriptive only; proximity test does not "
            "condition on the 48 Mixed hotspots."
        ),
    }
    (out_dir / "analysis2_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame({"null_median_dist": null_medians}).to_csv(
        out_dir / "analysis2_null_median_dist.csv", index=False
    )

    # Figure: ECDF only (panel A style, no letter label)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    x_obs = np.sort(obs_arr)
    y_obs = np.arange(1, len(x_obs) + 1) / len(x_obs)
    ax.plot(x_obs, y_obs, color=C_OBS, lw=2.0, label="Observed Predicted")
    if ecdf_pool:
        grid = np.linspace(0, max(np.percentile(obs_arr, 99), 50), 200)
        null_ys = [
            np.searchsorted(np.sort(arr), grid, side="right") / len(arr) for arr in ecdf_pool
        ]
        ax.plot(grid, np.mean(null_ys, axis=0), color=C_NULL, lw=2.0, label="Null (mean ECDF)")
    ax.set_xlabel("Distance to nearest Known (aa)")
    ax.set_ylabel("ECDF")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlim(left=0)
    fig.tight_layout()
    save_fig(fig, out_dir / "figure_analysis2_proximity_ecdf_enrichment")
    return summary


def dist_to_nearest_hotspot(pos: int, intervals: Sequence[Tuple[int, int]]) -> float:
    """Distance to nearest [start, end] interval; 0 if inside any interval."""
    if not intervals:
        return np.nan
    best = np.inf
    for a, b in intervals:
        if a <= pos <= b:
            return 0.0
        if pos < a:
            best = min(best, a - pos)
        else:
            best = min(best, pos - b)
    return float(best)


# ---------------------------------------------------------------------------
# Analysis 3
# ---------------------------------------------------------------------------

def run_analysis3(
    sites: pd.DataFrame,
    out_dir: Path,
    n_perm: int,
    rng: np.random.Generator,
    cluster_catalog: Optional[pd.DataFrame] = None,
    hotspot_catalog: Optional[pd.DataFrame] = None,
) -> Dict:
    """Cluster → nearest final 99-hotspot distance (literature overlap / recovery).

    Literature Cluster sites: TF_localization_final_four_sheets.xlsx
    (Cluster Nuclear + Cluster Cyto). Hotspot intervals: fixed 99-catalog.
    Note: Cluster sites can contribute as Known when building the 99, so this
    is a recovery/overlap analysis rather than a fully non-circular enrichment.
    Primary figure: Observed vs Null fraction within 15 aa (bars).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if cluster_catalog is not None:
        cluster_catalog.to_csv(out_dir / "analysis3_literature_cluster_from_xlsx.csv", index=False)
    if hotspot_catalog is None:
        raise ValueError("hotspot_catalog (fixed 99 hotspots) is required for Analysis 3")

    catalog = hotspot_catalog.copy()
    catalog.to_csv(out_dir / "analysis3_hotspot_catalog_99_used.csv", index=False)

    intervals_by_acc: Dict[str, List[Tuple[int, int]]] = {}
    for _, row in catalog.iterrows():
        intervals_by_acc.setdefault(str(row["UniProt"]), []).append(
            (int(row["start"]), int(row["end"]))
        )

    cluster_sites = sites.loc[sites["is_literature_cluster"]].copy()
    tfs_with_hs = set(intervals_by_acc.keys())
    cluster_eval = cluster_sites.loc[cluster_sites["protein_acc"].isin(tfs_with_hs)].copy()

    site_rows = []
    obs_dists: List[float] = []
    for _, row in cluster_eval.iterrows():
        acc = str(row["protein_acc"])
        pos = int(row["position"])
        d = dist_to_nearest_hotspot(pos, intervals_by_acc.get(acc, []))
        site_rows.append(
            {
                "INDEX": row["INDEX"],
                "TF": row["gene_name"],
                "UniProt": acc,
                "site": row["site"],
                "position": pos,
                "residue": row["residue"],
                "evidence": row["evidence"],
                "dist_to_nearest_hotspot99": d,
                "inside_hotspot99": bool(d == 0),
                "within_15aa": bool(np.isfinite(d) and d <= ADJ_MAX),
                "within_40aa": bool(np.isfinite(d) and d <= SPAN_MAX),
            }
        )
        if np.isfinite(d):
            obs_dists.append(d)

    site_df = pd.DataFrame(site_rows)
    site_df.to_csv(out_dir / "analysis3_cluster_to_hotspot99_distance.csv", index=False)

    obs_arr = np.asarray(obs_dists, dtype=float)
    obs_median = float(np.median(obs_arr)) if len(obs_arr) else np.nan
    obs_frac_inside = float(np.mean(obs_arr == 0)) if len(obs_arr) else np.nan
    obs_frac_15 = float(np.mean(obs_arr <= ADJ_MAX)) if len(obs_arr) else np.nan
    obs_frac_40 = float(np.mean(obs_arr <= SPAN_MAX)) if len(obs_arr) else np.nan

    cluster_by_tf = {
        acc: g["residue"].tolist()
        for acc, g in cluster_eval.groupby("protein_acc")
    }

    null_medians = np.empty(n_perm, dtype=float)
    null_frac_inside = np.empty(n_perm, dtype=float)
    null_frac_15 = np.empty(n_perm, dtype=float)
    null_frac_40 = np.empty(n_perm, dtype=float)

    for p in range(n_perm):
        pooled: List[float] = []
        for acc, res_list in cluster_by_tf.items():
            g = sites.loc[sites["protein_acc"] == acc]
            non_cl = g.loc[~g["is_literature_cluster"]]
            if len(non_cl) < len(res_list):
                continue
            samp_pos = sty_matched_sample(
                non_cl["position"].to_numpy(dtype=float),
                non_cl["residue"].to_numpy(),
                res_list,
                rng,
            )
            if samp_pos is None:
                continue
            iv = intervals_by_acc.get(acc, [])
            for pos in samp_pos:
                d = dist_to_nearest_hotspot(int(pos), iv)
                if np.isfinite(d):
                    pooled.append(d)
        arr = np.asarray(pooled, dtype=float)
        null_medians[p] = float(np.median(arr)) if len(arr) else np.nan
        null_frac_inside[p] = float(np.mean(arr == 0)) if len(arr) else np.nan
        null_frac_15[p] = float(np.mean(arr <= ADJ_MAX)) if len(arr) else np.nan
        null_frac_40[p] = float(np.mean(arr <= SPAN_MAX)) if len(arr) else np.nan

    null_median_mean = float(np.nanmean(null_medians))
    p_median_left = empirical_p_left(obs_median, null_medians)
    p_inside_right = empirical_p_right(obs_frac_inside, null_frac_inside)
    p_15_right = empirical_p_right(obs_frac_15, null_frac_15)
    p_40_right = empirical_p_right(obs_frac_40, null_frac_40)

    bg = sites.loc[sites["protein_acc"].isin(tfs_with_hs & set(cluster_eval["protein_acc"]))].copy()

    def inside_row(r) -> bool:
        d = dist_to_nearest_hotspot(int(r["position"]), intervals_by_acc.get(str(r["protein_acc"]), []))
        return bool(np.isfinite(d) and d == 0)

    bg["inside"] = [inside_row(r) for _, r in bg.iterrows()]
    cl = bg["is_literature_cluster"]
    a = int(((cl) & (bg["inside"])).sum())
    b = int(((cl) & (~bg["inside"])).sum())
    c = int(((~cl) & (bg["inside"])).sum())
    d_cnt = int(((~cl) & (~bg["inside"])).sum())
    oddsratio, fisher_p_greater = fisher_exact([[a, b], [c, d_cnt]], alternative="greater")
    _, fisher_p_less = fisher_exact([[a, b], [c, d_cnt]], alternative="less")
    _, fisher_p_two = fisher_exact([[a, b], [c, d_cnt]], alternative="two-sided")

    pd.DataFrame(
        [
            {"group": "Cluster", "inside": a, "not_inside": b},
            {"group": "non_Cluster", "inside": c, "not_inside": d_cnt},
        ]
    ).to_csv(out_dir / "analysis3_fisher_inside_table.csv", index=False)

    pd.DataFrame(
        {
            "null_median_dist": null_medians,
            "null_frac_inside": null_frac_inside,
            "null_frac_within_15": null_frac_15,
            "null_frac_within_40": null_frac_40,
        }
    ).to_csv(out_dir / "analysis3_null_distance_stats.csv", index=False)

    enrich_15 = (
        obs_frac_15 / float(np.nanmean(null_frac_15))
        if float(np.nanmean(null_frac_15)) > 0
        else np.nan
    )
    enrich_40 = (
        obs_frac_40 / float(np.nanmean(null_frac_40))
        if float(np.nanmean(null_frac_40)) > 0
        else np.nan
    )
    fold_median = null_median_mean / obs_median if obs_median and obs_median > 0 else np.nan

    summary = {
        "metric": "Cluster_to_nearest_hotspot99_distance",
        "hotspot_set": "fixed_99",
        "n_hotspots": int(len(catalog)),
        "hotspot_type_counts": catalog["hotspot_type"].value_counts().to_dict(),
        "n_cluster_sites_total": int(len(cluster_sites)),
        "n_cluster_sites_on_TFs_with_hotspot99": int(len(obs_arr)),
        "n_cluster_sites_on_TFs_with_blind_hotspot": int(len(obs_arr)),  # alias for summary md
        "n_blind_hotspots": int(len(catalog)),  # alias
        "obs_median_dist": obs_median,
        "null_mean_median_dist": null_median_mean,
        "fold_null_over_obs_median": fold_median,
        "empirical_p_median_dist_left": p_median_left,
        "obs_frac_inside": obs_frac_inside,
        "null_mean_frac_inside": float(np.nanmean(null_frac_inside)),
        "empirical_p_inside_right": p_inside_right,
        "obs_frac_within_15": obs_frac_15,
        "null_mean_frac_within_15": float(np.nanmean(null_frac_15)),
        "enrichment_within_15": enrich_15,
        "empirical_p_within_15_right": p_15_right,
        "obs_frac_within_40": obs_frac_40,
        "null_mean_frac_within_40": float(np.nanmean(null_frac_40)),
        "enrichment_within_40": enrich_40,
        "empirical_p_within_40_right": p_40_right,
        "odds_ratio_fisher_inside": float(oddsratio),
        "fisher_exact_p_greater": float(fisher_p_greater),
        "fisher_exact_p_less": float(fisher_p_less),
        "fisher_exact_p_two_sided": float(fisher_p_two),
        "fisher_counts": {"a": a, "b": b, "c": c, "d": d_cnt},
        "n_perm": n_perm,
        "cluster_source": {
            "xlsx": str(DEFAULT_CLUSTER_XLSX),
            "sheets": list(CLUSTER_SHEETS),
            "n_sites": int(sites["is_literature_cluster"].sum()),
        },
        "circularity_note": (
            "Final 99 hotspots may include Cluster sites as Known anchors; "
            "treat as recovery/overlap with the reported catalog, not a fully "
            "independent enrichment test."
        ),
        "interpretation": (
            "Distance from literature Cluster sites (Cluster Nuclear + Cluster Cyto) "
            "to nearest of the fixed 99 hotspots. Higher near-hotspot fraction vs null "
            "indicates recovery of literature multi-site regions by the catalog."
        ),
        "permutation_p_right": p_15_right,
        "permutation_p_left": p_median_left,
    }
    (out_dir / "analysis3_summary.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(3.6, 3.8))
    null_in = null_frac_inside[np.isfinite(null_frac_inside)]
    null_in_mean = float(np.mean(null_in))
    null_lo, null_hi = np.percentile(null_in, [2.5, 97.5])
    ax.bar([0], [obs_frac_inside], width=0.55, color=C_OBS, label="Observed")
    ax.bar([1], [null_in_mean], width=0.55, color=C_NULL, label="Null")
    ax.errorbar(
        [1],
        [null_in_mean],
        yerr=[[null_in_mean - null_lo], [null_hi - null_in_mean]],
        fmt="none",
        ecolor="#555555",
        capsize=4,
        lw=1.2,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Observed", "Null"])
    ax.set_ylabel("Fraction inside hotspot (99)")
    y_top = max(float(obs_frac_inside), null_hi, null_in_mean)
    ax.text(
        0.5,
        y_top * 1.05,
        p_to_stars(p_inside_right),
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="normal",
        color="#333333",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(y_top * 1.25, 0.4))
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    save_fig(fig, out_dir / "figure_analysis3_cluster_enrichment")
    return summary


# ---------------------------------------------------------------------------
# Analysis 4
# ---------------------------------------------------------------------------

def coherence_from_labels(labels: Sequence[str]) -> float:
    labs = [x for x in labels if x in {"Import", "Export"}]
    if len(labs) < 2:
        return np.nan
    n_imp = sum(1 for x in labs if x == "Import")
    n_exp = len(labs) - n_imp
    return max(n_imp, n_exp) / (n_imp + n_exp)


def run_analysis4(
    sites: pd.DataFrame,
    catalog: pd.DataFrame,
    members: pd.DataFrame,
    out_dir: Path,
    n_perm: int,
    rng: np.random.Generator,
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    mem = members.copy()
    if "predicted_direction" not in mem.columns:
        mem["predicted_direction"] = mem["Localization_annotation"].map(direction_label)

    # Hotspot-level coherence
    hs_rows = []
    for hid, g in mem.groupby("hotspot_id"):
        dirs = [d for d in g["predicted_direction"].tolist() if d in {"Import", "Export"}]
        n_imp = sum(1 for d in dirs if d == "Import")
        n_exp = sum(1 for d in dirs if d == "Export")
        coh = coherence_from_labels(dirs)
        meta = catalog.loc[catalog["hotspot_id"] == hid].iloc[0]
        hs_rows.append(
            {
                "hotspot_id": hid,
                "TF": meta["TF"],
                "UniProt": meta["UniProt"],
                "hotspot_type": meta["hotspot_type"],
                "n_sites": int(len(g)),
                "n_import": n_imp,
                "n_export": n_exp,
                "n_reliable_direction": n_imp + n_exp,
                "coherence": coh,
                "fully_coherent": bool(pd.notna(coh) and coh == 1.0),
                "eligible": bool((n_imp + n_exp) >= 2),
            }
        )
    hs_df = pd.DataFrame(hs_rows)
    hs_df.to_csv(out_dir / "analysis4_hotspot_coherence.csv", index=False)

    eligible = hs_df.loc[hs_df["eligible"]].copy()
    obs_mean = float(eligible["coherence"].mean()) if len(eligible) else np.nan
    obs_median = float(eligible["coherence"].median()) if len(eligible) else np.nan
    obs_full = float(eligible["fully_coherent"].mean()) if len(eligible) else np.nan

    # Build TF-level label pools among eligible hotspot members with reliable direction
    mem_el = mem.merge(
        eligible[["hotspot_id"]], on="hotspot_id", how="inner"
    )
    mem_el = mem_el.loc[mem_el["predicted_direction"].isin(["Import", "Export"])].copy()

    # Map INDEX -> TF
    tf_indices: Dict[str, List[str]] = {}
    tf_labels: Dict[str, List[str]] = {}
    index_to_hotspot = dict(zip(mem_el["INDEX"], mem_el["hotspot_id"]))
    for acc, g in mem_el.groupby("protein_acc"):
        tf_indices[str(acc)] = g["INDEX"].tolist()
        tf_labels[str(acc)] = g["predicted_direction"].tolist()

    # Site-level table
    mem_el.to_csv(out_dir / "analysis4_site_level_directions.csv", index=False)

    null_mean = np.empty(n_perm, dtype=float)
    null_median = np.empty(n_perm, dtype=float)
    null_full = np.empty(n_perm, dtype=float)

    # Precompute hotspot membership lists of INDEX
    hs_members = {
        hid: g["INDEX"].tolist()
        for hid, g in mem_el.groupby("hotspot_id")
    }

    for p in range(n_perm):
        # Shuffle labels within TF
        label_map: Dict[str, str] = {}
        for acc, idxs in tf_indices.items():
            labs = tf_labels[acc][:]
            rng.shuffle(labs)
            for ix, lab in zip(idxs, labs):
                label_map[ix] = lab
        cohs = []
        fulls = []
        for hid, idxs in hs_members.items():
            labs = [label_map[i] for i in idxs if i in label_map]
            c = coherence_from_labels(labs)
            if np.isfinite(c):
                cohs.append(c)
                fulls.append(1.0 if c == 1.0 else 0.0)
        null_mean[p] = float(np.mean(cohs)) if cohs else np.nan
        null_median[p] = float(np.median(cohs)) if cohs else np.nan
        null_full[p] = float(np.mean(fulls)) if fulls else np.nan

    summary = {
        "n_hotspots_total": int(len(hs_df)),
        "n_eligible_ge2_direction": int(len(eligible)),
        "obs_mean_coherence": obs_mean,
        "obs_median_coherence": obs_median,
        "obs_fully_coherent_fraction": obs_full,
        "null_mean_of_mean_coherence": float(np.nanmean(null_mean)),
        "null_mean_of_median_coherence": float(np.nanmean(null_median)),
        "null_mean_fully_coherent_fraction": float(np.nanmean(null_full)),
        "empirical_p_mean_coherence_right": empirical_p_right(obs_mean, null_mean),
        "empirical_p_median_coherence_right": empirical_p_right(obs_median, null_median),
        "empirical_p_fully_coherent_right": empirical_p_right(obs_full, null_full),
        "n_perm": n_perm,
    }

    # Mixed Known-anchor concordance
    mixed_ids = set(catalog.loc[catalog["hotspot_type"] == "Mixed evidence", "hotspot_id"])
    anchor_rows = []
    for hid in mixed_ids:
        g = mem.loc[mem["hotspot_id"] == hid].copy()
        # Known by PMID / is_known flag in members
        if "is_known" in g.columns:
            known = g.loc[g["is_known"].astype(bool)]
            pred = g.loc[~g["is_known"].astype(bool) & g["evidence"].astype(str).str.contains("Predicted", regex=False)]
        else:
            known = g.loc[g["evidence"].astype(str).eq("Known") | g["evidence"].astype(str).eq("Cluster")]
            pred = g.loc[g["evidence"].astype(str).str.contains("Predicted", regex=False)]

        known_dirs = [d for d in known["predicted_direction"].tolist() if d in {"Import", "Export"}]
        unique_known = sorted(set(known_dirs))
        if len(unique_known) == 0:
            status = "no_known_direction"
            concordant = np.nan
            n_pred_dir = 0
            n_agree = 0
        elif len(unique_known) > 1:
            status = "mixed_known_anchor"
            concordant = np.nan
            n_pred_dir = int(pred["predicted_direction"].isin(["Import", "Export"]).sum())
            n_agree = np.nan
        else:
            anchor = unique_known[0]
            pred_dirs = [d for d in pred["predicted_direction"].tolist() if d in {"Import", "Export"}]
            n_pred_dir = len(pred_dirs)
            n_agree = sum(1 for d in pred_dirs if d == anchor)
            status = "concordant_anchor"
            concordant = (n_agree / n_pred_dir) if n_pred_dir else np.nan
        meta = catalog.loc[catalog["hotspot_id"] == hid].iloc[0]
        anchor_rows.append(
            {
                "hotspot_id": hid,
                "TF": meta["TF"],
                "UniProt": meta["UniProt"],
                "n_known_with_direction": len(known_dirs),
                "known_directions": ";".join(unique_known) if unique_known else "",
                "n_predicted_with_direction": n_pred_dir,
                "n_predicted_agree_with_anchor": n_agree,
                "concordance": concordant,
                "anchor_status": status,
            }
        )
    anchor_df = pd.DataFrame(anchor_rows)
    anchor_df.to_csv(out_dir / "analysis4_mixed_known_anchor_concordance.csv", index=False)

    usable = anchor_df.loc[anchor_df["anchor_status"] == "concordant_anchor"]
    summary["n_mixed_hotspots"] = int(len(anchor_df))
    summary["n_mixed_known_anchor"] = int((anchor_df["anchor_status"] == "concordant_anchor").sum())
    summary["n_mixed_known_conflict"] = int((anchor_df["anchor_status"] == "mixed_known_anchor").sum())
    summary["n_mixed_no_known_direction"] = int((anchor_df["anchor_status"] == "no_known_direction").sum())
    if len(usable):
        # site-weighted and hotspot-mean concordance
        total_pred = int(usable["n_predicted_with_direction"].sum())
        total_agree = int(pd.to_numeric(usable["n_predicted_agree_with_anchor"], errors="coerce").fillna(0).sum())
        summary["mixed_anchor_site_concordance"] = total_agree / total_pred if total_pred else np.nan
        summary["mixed_anchor_hotspot_mean_concordance"] = float(usable["concordance"].mean())
    else:
        summary["mixed_anchor_site_concordance"] = np.nan
        summary["mixed_anchor_hotspot_mean_concordance"] = np.nan

    (out_dir / "analysis4_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(
        {
            "null_mean_coherence": null_mean,
            "null_median_coherence": null_median,
            "null_fully_coherent_fraction": null_full,
        }
    ).to_csv(out_dir / "analysis4_null_coherence.csv", index=False)

    # Single figure: fully coherent fraction Observed vs Null (bars + null CI)
    fig, ax = plt.subplots(figsize=(3.6, 3.8))
    null_fc = null_full[np.isfinite(null_full)]
    null_fc_mean = float(np.mean(null_fc))
    lo, hi = np.percentile(null_fc, [2.5, 97.5])
    ax.bar([0], [obs_full], width=0.55, color=C_OBS, label="Observed")
    ax.bar([1], [null_fc_mean], width=0.55, color=C_NULL, label="Null")
    ax.errorbar(
        [1],
        [null_fc_mean],
        yerr=[[null_fc_mean - lo], [hi - null_fc_mean]],
        fmt="none",
        ecolor="#555555",
        capsize=4,
        lw=1.2,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Observed", "Null"])
    ax.set_ylabel("Fully coherent hotspot fraction")
    p_full = empirical_p_right(obs_full, null_full)
    y_top = max(float(obs_full), hi, null_fc_mean)
    y_lim = annotate_bar_pvalue(ax, 0, 1, y_top, p_full)
    ax.set_ylim(0.7, max(1.02, y_lim))
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    save_fig(fig, out_dir / "figure_analysis4_direction_coherence")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_summary_md(out: Path, s1: Dict, s2: Dict, s3: Dict, s4: Dict) -> None:
    def support(flag: bool) -> str:
        return "支持" if flag else "不支持"

    a1_ok = s1.get("empirical_p_median_nnd_left", 1) < 0.05
    a2_ok = s2.get("empirical_p_median_dist_left", 1) < 0.05
    a3_ok = (
        s3.get("empirical_p_median_dist_left", 1) < 0.05
        or s3.get("empirical_p_within_15_right", 1) < 0.05
        or s3.get("permutation_p_right", 1) < 0.05
    )
    a3_depleted = (
        s3.get("empirical_p_median_dist_left", 1) > 0.95
        and s3.get("obs_median_dist", 0) > s3.get("null_mean_median_dist", 0)
    ) or (
        s3.get("fisher_exact_p_less", 1) < 0.05
        and s3.get("odds_ratio_fisher_inside", s3.get("odds_ratio_fisher", 1)) < 1
    )
    a4_ok = s4.get("empirical_p_mean_coherence_right", 1) < 0.05

    if a3_ok and not a3_depleted:
        a3_verdict = "支持"
    elif a3_depleted:
        a3_verdict = "不支持（距 blind hotspot 更远 / 重合更少）"
    else:
        a3_verdict = "不支持"

    lines = [
        "# Hotspot spatial / literature / direction validation",
        "",
        "Fixed catalog: **99 hotspots = 48 Mixed + 51 Predicted candidate** on 85 TFs.",
        "Thresholds unchanged: adj≤15 aa, span≤40 aa, ≥3 Predicted, mean score≥0.6.",
        "",
        "## Analysis 1 — HC Predicted spatial clustering",
        f"- Observed median NND: **{s1['obs_median_nnd']:.3f}** aa",
        f"- Null mean median NND: **{s1['null_mean_median_nnd']:.3f}** aa",
        f"- Fold (null/obs): **{s1['enrichment_median_fold_decrease']:.3f}**",
        f"- Empirical P (left): **{s1['empirical_p_median_nnd_left']:.4g}**",
        f"- Verdict: **{support(a1_ok)}** Predicted spatial clustering",
        "",
        "## Analysis 2 — Predicted proximity to Known",
        f"- Dual-evidence TFs: **{s2['n_dual_TFs']}**; Predicted sites: **{s2['n_predicted_sites']}**",
        f"- Observed median dist→Known: **{s2['obs_median_dist_to_known']:.3f}** aa",
        f"- Null mean median: **{s2['null_mean_median_dist']:.3f}** aa",
        f"- Fold (null/obs): **{s2['enrichment_median_fold_decrease']:.3f}**",
        f"- Empirical P (left): **{s2['empirical_p_median_dist_left']:.4g}**",
        f"- Descriptive overlap with Mixed spans (≤15 aa to Known): **{s2['n_pred_sites_within_15aa_of_known_and_near_mixed_span']}** sites",
        f"- Verdict: **{support(a2_ok)}** proximity to Known regulatory sites",
        "",
        "## Analysis 3 — Literature Cluster vs fixed 99 hotspots",
        f"- Hotspot set: **fixed 99** ({s3.get('hotspot_type_counts', {})})",
        f"- Cluster sites on those TFs: **{s3.get('n_cluster_sites_on_TFs_with_hotspot99', s3.get('n_cluster_sites_on_TFs_with_blind_hotspot', 'NA'))}** / {s3.get('n_cluster_sites_total', 'NA')}",
        f"- Observed median dist→hotspot: **{s3.get('obs_median_dist', float('nan')):.3f}** aa",
        f"- Null mean median: **{s3.get('null_mean_median_dist', float('nan')):.3f}** aa",
        f"- Fold (null/obs): **{s3.get('fold_null_over_obs_median', float('nan'))}**",
        f"- Empirical P (median left): **{s3.get('empirical_p_median_dist_left', float('nan')):.4g}**",
        f"- Fraction ≤15 aa: obs **{s3.get('obs_frac_within_15', float('nan')):.3f}** vs null "
        f"**{s3.get('null_mean_frac_within_15', float('nan')):.3f}** "
        f"(enrichment **{s3.get('enrichment_within_15', float('nan'))}**; "
        f"P **{s3.get('empirical_p_within_15_right', float('nan')):.4g}**)",
        f"- Note: {s3.get('circularity_note', 'uses fixed 99 catalog')}",
        f"- Verdict: **{a3_verdict}** literature multi-site recovery by the 99-hotspot catalog",
        "",
        "## Analysis 4 — Directional coherence",
        f"- Eligible hotspots (≥2 reliable directions): **{s4['n_eligible_ge2_direction']}**",
        f"- Observed mean / median coherence: **{s4['obs_mean_coherence']:.3f}** / **{s4['obs_median_coherence']:.3f}**",
        f"- Null mean of mean coherence: **{s4['null_mean_of_mean_coherence']:.3f}**",
        f"- Fully coherent fraction: obs **{s4['obs_fully_coherent_fraction']:.3f}** vs null **{s4['null_mean_fully_coherent_fraction']:.3f}**",
        f"- Empirical P (mean coherence): **{s4['empirical_p_mean_coherence_right']:.4g}**",
        f"- Mixed Known-anchor site concordance: **{s4.get('mixed_anchor_site_concordance', float('nan'))}**",
        f"- Mixed conflict / usable / no-dir: "
        f"{s4['n_mixed_known_conflict']} / {s4['n_mixed_known_anchor']} / {s4['n_mixed_no_known_direction']}",
        f"- Verdict: **{support(a4_ok)}** directional coherence",
        "",
        "## Evidence-chain summary",
        "",
        "| Step | Result |",
        "|---|---|",
        f"| Predicted spatial clustering | {support(a1_ok)} |",
        f"| Proximity to Known regulatory sites | {support(a2_ok)} |",
        f"| Literature multisite support | {a3_verdict} |",
        f"| Directional coherence | {support(a4_ok)} |",
        "",
        f"Core chain (1→2→4) supported: **{support(a1_ok and a2_ok and a4_ok)}**; "
        f"full 1→2→3→4 chain: **{support(a1_ok and a2_ok and a3_ok and a4_ok)}**",
        "",
    ]
    (out / "VALIDATION_SUMMARY.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--hotspot-dir", type=Path, default=DEFAULT_HOTSPOT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cluster-xlsx", type=Path, default=DEFAULT_CLUSTER_XLSX)
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("Loading site master + literature Cluster sheets…")
    sites, cluster_cat = load_site_master(args.summary, args.cluster_xlsx)
    cluster_cat.to_csv(out / "literature_cluster_catalog_from_xlsx.csv", index=False)
    print(
        f"Literature Cluster sites from xlsx: {len(cluster_cat)} "
        f"(Nuclear={(cluster_cat['cluster_sheet']=='Cluster Nuclear').sum()}, "
        f"Cyto={(cluster_cat['cluster_sheet']=='Cluster Cyto').sum()})"
    )
    master_out = sites.rename(
        columns={
            "gene_name": "TF",
            "protein_acc": "UniProt",
            "position": "Position",
            "residue": "Residue",
            "Direction_score": "direction_score",
        }
    )
    keep_cols = [
        "TF",
        "UniProt",
        "Position",
        "Residue",
        "site",
        "PMID",
        "evidence",
        "FuncTransport_score",
        "predicted_direction",
        "direction_score",
        "is_known",
        "is_predicted",
        "is_hc_predicted",
        "is_literature_cluster",
        "is_literature_cluster_from_evidence",
        "is_observed_background",
        "INDEX",
        "Localization_annotation",
    ]
    master_out[keep_cols].to_csv(out / "site_master.csv", index=False)

    print("Loading fixed 99 hotspots…")
    catalog, members = load_fixed_hotspots(args.hotspot_dir, sites)
    catalog.to_csv(out / "hotspot_catalog_99.csv", index=False)
    members.to_csv(out / "hotspot_members_99.csv", index=False)

    print(f"Analysis 1 ({args.n_perm} perms)…")
    s1 = run_analysis1(sites, out / "analysis1_nnd", args.n_perm, rng)
    print(json.dumps({k: s1[k] for k in ["obs_median_nnd", "null_mean_median_nnd", "empirical_p_median_nnd_left"]}, indent=2))

    print(f"Analysis 2 ({args.n_perm} perms)…")
    s2 = run_analysis2(sites, catalog, out / "analysis2_proximity", args.n_perm, rng)
    print(json.dumps({k: s2[k] for k in ["obs_median_dist_to_known", "null_mean_median_dist", "empirical_p_median_dist_left"]}, indent=2))

    print(f"Analysis 3 ({args.n_perm} perms) against fixed 99 hotspots…")
    s3 = run_analysis3(
        sites,
        out / "analysis3_cluster_enrichment",
        args.n_perm,
        rng,
        cluster_catalog=cluster_cat,
        hotspot_catalog=catalog,
    )
    print(
        json.dumps(
            {
                k: s3[k]
                for k in [
                    "obs_median_dist",
                    "null_mean_median_dist",
                    "obs_frac_within_15",
                    "null_mean_frac_within_15",
                    "empirical_p_within_15_right",
                ]
            },
            indent=2,
        )
    )

    print(f"Analysis 4 ({args.n_perm} perms)…")
    s4 = run_analysis4(sites, catalog, members, out / "analysis4_coherence", args.n_perm, rng)
    print(
        json.dumps(
            {
                k: s4[k]
                for k in [
                    "obs_mean_coherence",
                    "obs_fully_coherent_fraction",
                    "empirical_p_mean_coherence_right",
                    "mixed_anchor_site_concordance",
                ]
            },
            indent=2,
        )
    )

    write_summary_md(out, s1, s2, s3, s4)
    combined = {"analysis1": s1, "analysis2": s2, "analysis3": s3, "analysis4": s4}
    (out / "all_analyses_summary.json").write_text(json.dumps(combined, indent=2))
    print(f"Done. Wrote {out}")


if __name__ == "__main__":
    main()
