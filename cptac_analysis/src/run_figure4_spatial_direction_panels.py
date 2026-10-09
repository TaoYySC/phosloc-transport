#!/usr/bin/env python3
"""Figure 4A–C + Figure S5A–C for spatial clustering / directional coherence.

Figure 4A  HC Predicted close-pair enrichment (obs/expected) at 10/15/25/40/50 aa
Figure 4B  Directional coherence of catalog hotspots vs matched pseudo-hotspots
Figure 4C  Three representative hotspot schematics (STAT3 / AR / SIX4)
Figure S5A Predicted→Known proximity ECDF (supporting)
Figure S5B Exploratory comparison: local phospho-hotspots vs literature multi-site regions
Figure S5C Sensitivity heatmap over adj×span grid

Catalog thresholds remain fixed for the main 99-hotspot analyses (adj≤15, span≤40,
≥3 Predicted, mean score≥0.6). S5B is not a validation panel: all literature Cluster
sites are held out from S5B-only hotspot construction, then tested for overlap.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

mpl.rcParams.update(
    {
        # PDF core fonts (WinAnsi) keep phrases as single text objects so
        # viewers/AI extract whole labels instead of per-character CID glyphs.
        "pdf.use14corefonts": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
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
DEFAULT_CLUSTER_XLSX = (
    REPO
    / "functional/data/dataset_phos_site"
    / "TF_localization_final_four_sheets.xlsx"
)
DEFAULT_OUT = (
    REPO
    / "cptac_analysis/results"
    / "figure4_spatial_direction_panels_v11_147pos_d3_platt"
)

CLUSTER_SHEETS = ("Cluster Nuclear", "Cluster Cyto")
ADJ_MAX = 15
SPAN_MAX = 40
PRED_MIN_N = 3
PRED_MEAN_MIN = 0.6
N_PERM = 10_000
RNG_SEED = 20260920
CLOSE_PAIR_DISTS = (10, 15, 25, 40, 50)
SENS_ADJ = (10, 15, 20)
SENS_SPAN = (30, 40, 50)

C_OBS = "#4DBBD5"
C_NULL = "#9e9e9e"
C_ENR = "#E64B35"
C_NUC = "#3C5488"
C_CYTO = "#E64B35"
C_UNRES = "#B0B0B0"
C_HS_BG = "#FFF3D6"
C_TEXT = "#333333"

EXEMPLARS = (
    {
        "gene": "STAT3",
        "uniprot": "P40763",
        "length": 770,
        "start": 686,
        "end": 708,
        "title": "STAT3 (P40763)",
        "subtitle": "Mixed evidence hotspot 686–708 · nuclear accumulation-consistent",
    },
    {
        "gene": "AR",
        "uniprot": "P10275",
        "length": 920,
        "start": 647,
        "end": 653,
        "title": "AR (P10275)",
        "subtitle": "Mixed evidence hotspot 647–653 · cytoplasmic redistribution-consistent",
    },
    {
        "gene": "SIX4",
        "uniprot": "Q9UIU6",
        "length": 781,
        "start": 282,
        "end": 308,
        "title": "SIX4 (Q9UIU6)",
        "subtitle": "Predicted-candidate hotspot 282–308 · cytoplasmic redistribution-consistent",
    },
)

# PSP-like site class colors
C_KNOWN = "#C0392B"
C_HC = "#2980B9"
C_OTHER = "#95A5A6"
C_BAR = "#2C3E50"


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


def direction_display(lab: Optional[str]) -> str:
    if lab == "Import":
        return "nuclear"
    if lab == "Export":
        return "cytoplasmic"
    return "unresolved"


def p_to_star(p: float) -> str:
    if not np.isfinite(p):
        return "ns"
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def empirical_p_right(obs: float, null: np.ndarray) -> float:
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if len(null) == 0 or not np.isfinite(obs):
        return np.nan
    return float((np.sum(null >= obs) + 1) / (len(null) + 1))


def empirical_p_left(obs: float, null: np.ndarray) -> float:
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if len(null) == 0 or not np.isfinite(obs):
        return np.nan
    return float((np.sum(null <= obs) + 1) / (len(null) + 1))


def save_fig(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def sty_matched_sample(
    bg_pos: np.ndarray,
    bg_res: np.ndarray,
    target_res: Sequence[str],
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    n = len(target_res)
    if n == 0 or len(bg_pos) < n:
        return None
    counts = {"S": 0, "T": 0, "Y": 0}
    for r in target_res:
        counts[r] = counts.get(r, 0) + 1
    chosen: List[int] = []
    remaining = np.ones(len(bg_pos), dtype=bool)
    for res, need in counts.items():
        if need <= 0:
            continue
        pool = np.where(remaining & (bg_res == res))[0]
        if len(pool) >= need:
            pick = rng.choice(pool, size=need, replace=False)
            chosen.extend(pick.tolist())
            remaining[pick] = False
        else:
            chosen.extend(pool.tolist())
            remaining[pool] = False
            short = need - len(pool)
            pool2 = np.where(remaining)[0]
            if len(pool2) < short:
                return None
            pick = rng.choice(pool2, size=short, replace=False)
            chosen.extend(pick.tolist())
            remaining[pick] = False
    if len(chosen) != n:
        return None
    return bg_pos[np.asarray(chosen, dtype=int)]


def count_close_pairs(pos: np.ndarray, d: float) -> int:
    arr = np.sort(np.asarray(pos, dtype=float))
    n = len(arr)
    if n < 2:
        return 0
    count = 0
    j = 0
    for i in range(n):
        while j < n and arr[j] - arr[i] <= d:
            j += 1
        count += j - i - 1
    return int(count)


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


def build_catalog_from_seeds(
    seeds: pd.DataFrame, adj_max: int, span_max: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    hs_rows = []
    mem_rows = []
    hid = 0
    for acc, g in seeds.groupby("protein_acc", sort=True):
        gene = str(g["gene_name"].iloc[0])
        for mem in cluster_sorted(g, adj_max, span_max):
            htype = classify_hotspot(mem)
            if htype is None:
                continue
            hid += 1
            sites = mem.sort_values("position")
            start = int(sites["position"].min())
            end = int(sites["position"].max())
            hotspot_id = f"{acc}_H{hid}_{sites['site'].iloc[0]}_{sites['site'].iloc[-1]}"
            pred_scores = sites.loc[sites["is_predicted"], "FuncTransport_score"]
            hs_rows.append(
                {
                    "hotspot_id": hotspot_id,
                    "TF": gene,
                    "UniProt": acc,
                    "start": start,
                    "end": end,
                    "span": end - start,
                    "n_sites": int(len(sites)),
                    "n_known": int(sites["is_known"].sum()),
                    "n_predicted": int(sites["is_predicted"].sum()),
                    "mean_func_score": float(pred_scores.mean()) if len(pred_scores) else np.nan,
                    "hotspot_type": htype,
                    "adj_max": adj_max,
                    "span_max": span_max,
                }
            )
            for _, row in sites.iterrows():
                mem_rows.append(
                    {
                        "hotspot_id": hotspot_id,
                        "TF": gene,
                        "UniProt": acc,
                        "INDEX": row["INDEX"],
                        "site": row["site"],
                        "position": int(row["position"]),
                        "residue": row["residue"],
                        "evidence": row["evidence"],
                        "is_known": bool(row["is_known"]),
                        "is_predicted": bool(row["is_predicted"]),
                        "FuncTransport_score": row["FuncTransport_score"],
                        "predicted_direction": row["predicted_direction"],
                        "Localization_annotation": row["Localization_annotation"],
                        "hotspot_type": htype,
                    }
                )
    return pd.DataFrame(hs_rows), pd.DataFrame(mem_rows)


def load_literature_cluster(xlsx: Path) -> pd.DataFrame:
    rows = []
    for sheet in CLUSTER_SHEETS:
        raw = pd.read_excel(xlsx, sheet_name=sheet)
        colmap = {c.lower().strip(): c for c in raw.columns}
        tf_c = colmap.get("tf gene") or colmap.get("tf") or colmap.get("gene")
        up_c = colmap.get("uniprot id") or colmap.get("uniprot")
        ps_c = colmap.get("phosphosite") or colmap.get("site")
        for _, row in raw.iterrows():
            site = str(row[ps_c]).strip()
            m = re.search(r"([STY]\d+)", site, flags=re.IGNORECASE)
            if not m:
                continue
            site_tok = m.group(1).upper()
            acc = str(row[up_c]).strip()
            rows.append(
                {
                    "TF": str(row[tf_c]).strip(),
                    "UniProt": acc,
                    "site": site_tok,
                    "INDEX": f"{acc}_{site_tok}",
                    "sheet": sheet,
                }
            )
    out = pd.DataFrame(rows).drop_duplicates("INDEX")
    return out


def load_site_master(summary: Path, cluster_xlsx: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(summary)
    df["protein_acc"] = df["protein_acc"].astype(str).str.strip()
    df["gene_name"] = df["gene_name"].astype(str).str.strip()
    df["site"] = df["site"].astype(str).str.strip()
    df["evidence"] = df["evidence"].astype(str).str.strip()
    df["PMID"] = df["PMID"].astype(str).str.strip()
    df["Localization_annotation"] = df["Localization_annotation"].astype(str)
    df["position"] = df["site"].map(parse_position)
    df["residue"] = df["site"].map(parse_residue)
    df = df.dropna(subset=["position", "residue"]).copy()
    df["position"] = df["position"].astype(int)
    df["FuncTransport_score"] = pd.to_numeric(df["FuncTransport_score"], errors="coerce")
    df["is_known"] = ~df["PMID"].isin(["", "-", "nan", "None", "NaN"])
    df["is_predicted"] = df["evidence"].str.contains("Predicted", regex=False)
    df["is_hc_predicted"] = df["evidence"].isin({"Predicted", "Predicted;Cluster"})
    df["predicted_direction"] = df["Localization_annotation"].map(direction_label)
    df["INDEX"] = df["protein_acc"] + "_" + df["site"]
    cluster_cat = load_literature_cluster(cluster_xlsx)
    df["is_literature_cluster"] = df["INDEX"].isin(set(cluster_cat["INDEX"]))
    return df.reset_index(drop=True), cluster_cat


def load_fixed_catalog(hotspot_dir: Path, sites: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    hs = pd.read_csv(hotspot_dir / "hotspots_d15.csv")
    mem = pd.read_csv(hotspot_dir / "hotspot_members_d15.csv")
    assert len(hs) == 99, f"Expected 99 hotspots, got {len(hs)}"
    site_dir = sites.set_index("INDEX")["predicted_direction"]
    site_loc = sites.set_index("INDEX")["Localization_annotation"]
    mem = mem.copy()
    mem["INDEX"] = mem["protein_acc"].astype(str) + "_" + mem["site"].astype(str)
    if "predicted_direction" not in mem.columns:
        mem["predicted_direction"] = mem["INDEX"].map(site_dir)
    if "Localization_annotation" not in mem.columns:
        mem["Localization_annotation"] = mem["INDEX"].map(site_loc)
    catalog = pd.DataFrame(
        {
            "hotspot_id": hs["hotspot_id"],
            "TF": hs["gene_name"],
            "UniProt": hs["protein_acc"],
            "start": hs["start_position"],
            "end": hs["end_position"],
            "span": hs["span_aa"],
            "n_sites": hs["n_members"],
            "hotspot_type": hs["hotspot_type"],
            "direction_call": hs.get("direction_call", pd.Series([""] * len(hs))),
        }
    )
    return catalog, mem


def coherence_from_labels(labels: Sequence[str]) -> float:
    labs = [x for x in labels if x in {"Import", "Export"}]
    if len(labs) < 2:
        return np.nan
    n_imp = sum(1 for x in labs if x == "Import")
    n_exp = len(labs) - n_imp
    return max(n_imp, n_exp) / (n_imp + n_exp)


def is_fully_coherent(labels: Sequence[str]) -> bool:
    c = coherence_from_labels(labels)
    return bool(np.isfinite(c) and c == 1.0)


# ---------------------------------------------------------------------------
# Figure 4A — close-pair enrichment
# ---------------------------------------------------------------------------

def run_figure4a(sites: pd.DataFrame, out_dir: Path, n_perm: int, rng: np.random.Generator) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Strict HC Predicted only: exclude dual-labeled Known∩HC (PMID + Predicted;Cluster)
    pred = sites.loc[sites["is_hc_predicted"] & ~sites["is_known"]].copy()
    tf_data = []
    obs_pairs = {d: 0 for d in CLOSE_PAIR_DISTS}
    for acc, g in pred.groupby("protein_acc"):
        if len(g) < 2:
            continue
        pos = g["position"].to_numpy(dtype=float)
        for d in CLOSE_PAIR_DISTS:
            obs_pairs[d] += count_close_pairs(pos, d)
        bg = sites.loc[sites["protein_acc"] == acc]
        tf_data.append(
            {
                "target_res": g["residue"].tolist(),
                "bg_pos": bg["position"].to_numpy(dtype=float),
                "bg_res": bg["residue"].to_numpy(),
                "n": len(g),
            }
        )

    null_pairs = {d: np.empty(n_perm, dtype=float) for d in CLOSE_PAIR_DISTS}
    for p in range(n_perm):
        totals = {d: 0 for d in CLOSE_PAIR_DISTS}
        for td in tf_data:
            samp = sty_matched_sample(td["bg_pos"], td["bg_res"], td["target_res"], rng)
            if samp is None:
                continue
            for d in CLOSE_PAIR_DISTS:
                totals[d] += count_close_pairs(samp, d)
        for d in CLOSE_PAIR_DISTS:
            null_pairs[d][p] = totals[d]

    rows = []
    for d in CLOSE_PAIR_DISTS:
        null_mean = float(np.nanmean(null_pairs[d]))
        enr = obs_pairs[d] / null_mean if null_mean > 0 else np.nan
        p_right = empirical_p_right(obs_pairs[d], null_pairs[d])
        rows.append(
            {
                "distance_aa": d,
                "obs_close_pairs": int(obs_pairs[d]),
                "null_mean_close_pairs": null_mean,
                "enrichment_obs_over_expected": enr,
                "empirical_p_right": p_right,
                "star": p_to_star(p_right),
            }
        )
    enrich_df = pd.DataFrame(rows)
    enrich_df.to_csv(out_dir / "figure4a_close_pair_enrichment.csv", index=False)
    pd.DataFrame({f"null_pairs_{d}": null_pairs[d] for d in CLOSE_PAIR_DISTS}).to_csv(
        out_dir / "figure4a_null_close_pairs.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    x = np.arange(len(CLOSE_PAIR_DISTS))
    y = enrich_df["enrichment_obs_over_expected"].to_numpy()
    ax.bar(x, y, width=0.65 * 2 / 3, color=C_OBS, edgecolor="none")
    ax.axhline(1.0, color=C_NULL, ls="--", lw=1.0, label="Expected (null = 1)")
    for i, row in enumerate(enrich_df.itertuples(index=False)):
        ax.text(
            i,
            float(row.enrichment_obs_over_expected) + 0.04,
            row.star,
            ha="center",
            va="bottom",
            fontsize=11,
            color=C_TEXT,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in CLOSE_PAIR_DISTS])
    ax.set_xlabel("Distance threshold (aa)")
    ax.set_ylabel("Observed / expected close-pair enrichment")
    ax.set_ylim(0, max(2.4, float(np.nanmax(y)) * 1.2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    save_fig(fig, out_dir / "figure4a_close_pair_enrichment")

    summary = {
        "n_hc_predicted_sites": int(len(pred)),
        "n_tfs_with_ge2_hc": int(len(tf_data)),
        "definition": (
            "HC Predicted only (evidence in {Predicted, Predicted;Cluster}) and not Known "
            "(no PMID); excludes dual-labeled Known∩HC sites"
        ),
        "enrichment_table": rows,
    }
    (out_dir / "figure4a_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Figure 4B — matched pseudo-hotspot coherence
# ---------------------------------------------------------------------------

def run_figure4b(
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

    # Classic eligibility: ≥2 direction-resolved sites
    min_n_resolved = 2

    hs_rows = []
    for hid, g in mem.groupby("hotspot_id"):
        dirs = [d for d in g["predicted_direction"].tolist() if d in {"Import", "Export"}]
        meta = catalog.loc[catalog["hotspot_id"] == hid].iloc[0]
        coh = coherence_from_labels(dirs)
        hs_rows.append(
            {
                "hotspot_id": hid,
                "TF": meta["TF"],
                "UniProt": meta["UniProt"],
                "start": int(meta["start"]),
                "end": int(meta["end"]),
                "span": int(meta["span"]),
                "n_sites": int(meta["n_sites"]),
                "hotspot_type": meta["hotspot_type"],
                "n_direction_resolved": len(dirs),
                "coherence": coh,  # majority-direction fraction
                "fully_coherent": bool(np.isfinite(coh) and coh == 1.0),
                "eligible": len(dirs) >= min_n_resolved,
            }
        )
    hs_df = pd.DataFrame(hs_rows)
    hs_df.to_csv(out_dir / "figure4b_hotspot_coherence.csv", index=False)
    eligible = hs_df.loc[hs_df["eligible"]].copy()
    n_eligible = int(len(eligible))
    n_coherent = int(eligible["fully_coherent"].sum())
    obs_mean_coh = float(eligible["coherence"].mean()) if n_eligible else np.nan
    obs_frac_full = float(eligible["fully_coherent"].mean()) if n_eligible else np.nan

    # Null: leave-one-hotspot-out — for each hotspot, sample matched pseudo-sets from
    # the same TF after excluding ONLY that hotspot's interval; other hotspots on the
    # same TF remain in the pool. Fallback to all sites if the leave-one-out pool is
    # too small.
    tf_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    for acc, g in sites.groupby("protein_acc"):
        g = g.sort_values("position")
        pos = g["position"].to_numpy(dtype=int)
        dirs = g["predicted_direction"].tolist()
        dir_codes = np.array(
            [1 if d == "Import" else 2 if d == "Export" else 0 for d in dirs], dtype=np.int8
        )
        tf_arrays[str(acc)] = {
            "pos": pos,
            "dir": dir_codes,
        }

    job_specs = []
    n_fallback = 0
    for _, row in eligible.iterrows():
        acc = str(row["UniProt"])
        n_sites = int(row["n_sites"])
        span = int(row["span"])
        tol = max(5, int(round(0.25 * max(span, 1))))
        arr = tf_arrays.get(acc)
        if arr is None:
            continue
        pos = arr["pos"]
        a0, b0 = int(row["start"]), int(row["end"])
        outside_self = ~((pos >= a0) & (pos <= b0))
        pool = np.where(outside_self)[0]
        used_fallback = False
        if len(pool) < n_sites:
            pool = np.arange(len(pos))
            used_fallback = True
            n_fallback += 1
        if len(pool) < n_sites:
            continue
        job_specs.append(
            {
                "acc": acc,
                "n": n_sites,
                "span": span,
                "tol": tol,
                "pool": pool,
                "hotspot_id": row["hotspot_id"],
                "fallback_all_sites": used_fallback,
            }
        )

    def sample_span_matched(spec: Dict) -> Optional[np.ndarray]:
        arr = tf_arrays[spec["acc"]]
        pos = arr["pos"]
        pool = spec["pool"]
        n = spec["n"]
        span = spec["span"]
        tol = spec["tol"]
        best_idx = None
        best_delta = None
        for _ in range(50):
            choose = rng.choice(pool, size=n, replace=False)
            s = int(pos[choose].max() - pos[choose].min())
            delta = abs(s - span)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_idx = choose
            if delta <= tol:
                return choose
        return best_idx

    def majority_coherence_from_codes(codes: np.ndarray) -> float:
        labs = codes[(codes == 1) | (codes == 2)]
        if len(labs) < min_n_resolved:
            return np.nan
        n_imp = int(np.sum(labs == 1))
        n_exp = int(np.sum(labs == 2))
        return max(n_imp, n_exp) / (n_imp + n_exp)

    null_mean_coh = np.empty(n_perm, dtype=float)
    null_frac_full = np.empty(n_perm, dtype=float)
    for p in range(n_perm):
        coh_vals = []
        full_flags = []
        for spec in job_specs:
            idx = sample_span_matched(spec)
            if idx is None:
                continue
            codes = tf_arrays[spec["acc"]]["dir"][idx]
            c = majority_coherence_from_codes(codes)
            if not np.isfinite(c):
                continue
            coh_vals.append(c)
            full_flags.append(1.0 if c == 1.0 else 0.0)
        null_mean_coh[p] = float(np.mean(coh_vals)) if coh_vals else np.nan
        null_frac_full[p] = float(np.mean(full_flags)) if full_flags else np.nan

    p_mean = empirical_p_right(obs_mean_coh, null_mean_coh)
    p_full = empirical_p_right(obs_frac_full, null_frac_full)
    null_mean_of_mean = float(np.nanmean(null_mean_coh))
    null_mean_of_full = float(np.nanmean(null_frac_full))
    pd.DataFrame(
        {
            "null_mean_majority_coherence": null_mean_coh,
            "null_fully_coherent_fraction": null_frac_full,
        }
    ).to_csv(out_dir / "figure4b_null_matched_pseudo_coherence.csv", index=False)

    # Main figure: fully coherent fraction (classic 4B display) under leave-one-out
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    vals = null_frac_full[np.isfinite(null_frac_full)]
    ax.hist(vals, bins=30, color=C_NULL, alpha=0.85, density=True, label="Matched pseudo-hotspots")
    ax.axvline(obs_frac_full, color=C_OBS, lw=2.2, label="Observed hotspots")
    ax.axvline(
        null_mean_of_full,
        color="#666666",
        lw=1.2,
        ls="--",
        label=f"Null mean={null_mean_of_full:.3f}",
    )
    ax.set_xlabel("Fully directionally coherent fraction")
    ax.set_ylabel("Density")
    ax.text(
        0.98,
        0.95,
        f"{n_coherent}/{n_eligible} hotspots were\n"
        f"directionally coherent\n(leave-one-hotspot-out)\n{p_to_star(p_full)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=C_TEXT,
    )
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    save_fig(fig, out_dir / "figure4b_directional_coherence")

    # Secondary: mean majority coherence
    fig2, ax2 = plt.subplots(figsize=(4.6, 3.8))
    vals2 = null_mean_coh[np.isfinite(null_mean_coh)]
    ax2.hist(vals2, bins=30, color=C_NULL, alpha=0.85, density=True, label="Matched pseudo-hotspots")
    ax2.axvline(obs_mean_coh, color=C_OBS, lw=2.2, label="Observed hotspots")
    ax2.axvline(
        null_mean_of_mean,
        color="#666666",
        lw=1.2,
        ls="--",
        label=f"Null mean={null_mean_of_mean:.3f}",
    )
    ax2.set_xlabel("Mean majority-direction coherence")
    ax2.set_ylabel("Density")
    ax2.text(
        0.98,
        0.95,
        f"Mean coherence={obs_mean_coh:.3f}\n"
        f"(leave-one-hotspot-out)\n{p_to_star(p_mean)}",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=C_TEXT,
    )
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    fig2.tight_layout()
    save_fig(fig2, out_dir / "figure4b_mean_majority_coherence_loo")

    summary = {
        "null_scheme": "leave_one_hotspot_out",
        "statistic_primary": "fully_coherent_fraction",
        "min_n_direction_resolved": min_n_resolved,
        "n_hotspots_total": int(len(hs_df)),
        "n_eligible": n_eligible,
        "n_eligible_ge2_direction": n_eligible,
        "n_fully_coherent": n_coherent,
        "n_jobs": int(len(job_specs)),
        "n_jobs_fallback_all_sites": int(n_fallback),
        "obs_fully_coherent_fraction": obs_frac_full,
        "null_mean_fully_coherent_fraction": null_mean_of_full,
        "empirical_p_right_fully_coherent": p_full,
        "star_fully_coherent": p_to_star(p_full),
        "obs_mean_majority_coherence": obs_mean_coh,
        "null_mean_majority_coherence": null_mean_of_mean,
        "empirical_p_right_mean_coherence": p_mean,
        "star_mean_coherence": p_to_star(p_mean),
        # backward-compatible keys
        "obs_coherent_fraction": obs_frac_full,
        "null_mean_coherent_fraction": null_mean_of_full,
        "empirical_p_right": p_full,
        "star": p_to_star(p_full),
        "annotation": (
            f"{n_coherent}/{n_eligible} hotspots were directionally coherent "
            f"(leave-one-hotspot-out)"
        ),
        "null_definition": (
            "Matched pseudo-hotspots: same TF, same n_sites, span-matched; "
            "leave-one-hotspot-out — exclude only the tested hotspot interval from the "
            "sampling pool, keeping other hotspots on the same TF; fallback to all sites "
            "if the leave-one-out pool is too small; not direction-label shuffle."
        ),
    }
    (out_dir / "figure4b_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Figure 4C — exemplars
# ---------------------------------------------------------------------------

def run_figure4c(sites: pd.DataFrame, members: pd.DataFrame, out_dir: Path) -> Dict:
    """PSP-style full-length tracks + compact hotspot inset; one file per exemplar."""
    out_dir.mkdir(parents=True, exist_ok=True)
    used = []

    for ex in EXEMPLARS:
        gene = ex["gene"]
        acc = ex["uniprot"]
        length = int(ex["length"])
        hs0, hs1 = int(ex["start"]), int(ex["end"])

        g = sites.loc[sites["protein_acc"].astype(str) == acc].copy()
        if g.empty:
            g = sites.loc[sites["gene_name"].astype(str).str.upper() == gene].copy()
        g = g.sort_values("position")

        def site_class(row: pd.Series) -> str:
            if bool(row["is_known"]):
                return "Known"
            if bool(row["is_hc_predicted"]) or bool(row["is_predicted"]):
                return "HC Predicted"
            return "Other observed"

        g["site_class"] = g.apply(site_class, axis=1)
        g["dir_lab"] = g["predicted_direction"].map(
            lambda d: "Import" if d == "Import" else "Export" if d == "Export" else "Unresolved"
        )
        # Color = direction; size/edge = evidence (links to Fig 4B coherence)
        dir_color = {"Import": C_NUC, "Export": C_CYTO, "Unresolved": C_UNRES}
        evid_z = {"Other observed": 2, "HC Predicted": 3, "Known": 4}

        def marker_style(cls: str) -> Tuple[float, str, float]:
            """Return (size, edgecolor, edge_lw) for evidence class."""
            if cls == "Known":
                return 52.0, "#1A1A1A", 1.15
            if cls == "HC Predicted":
                return 34.0, "#FFFFFF", 0.55
            return 18.0, "#FFFFFF", 0.35

        def draw_sequence_bar(ax, x0: float, x1: float, y: float = 0.0, height: float = 0.07) -> None:
            """Single rounded sequence bar (replaces double-line + gray fill)."""
            from matplotlib.patches import FancyBboxPatch

            round_r = max(0.015, height * 0.35)
            ax.add_patch(
                FancyBboxPatch(
                    (x0, y - height / 2),
                    x1 - x0,
                    height,
                    boxstyle=f"round,pad=0.0,rounding_size={round_r}",
                    linewidth=0,
                    facecolor="#B0BEC5",
                    zorder=1,
                )
            )

        def draw_lollipops(
            ax,
            rows: pd.DataFrame,
            *,
            label_all: bool,
            stem_scale: float = 1.0,
            ms_scale: float = 1.0,
            label_fs: float = 7.0,
            label_rot: float = 0,
            y0: float = 0.0,
            bar_half: float = 0.05,
            min_label_gap: float = 28.0,
        ) -> None:
            """Color=direction; size/edge=evidence; horizontal labels with stagger."""
            items = []
            for _, row in rows.iterrows():
                pos = float(row["position"])
                cls = str(row["site_class"])
                dlab = str(row["dir_lab"])
                do_label = bool(label_all or cls != "Other observed")
                items.append((pos, cls, dlab, str(row["site"]), do_label))
            items.sort(key=lambda t: t[0])

            label_levels: Dict[int, int] = {}
            labeled_idx = [i for i, it in enumerate(items) if it[4]]
            for j, i in enumerate(labeled_idx):
                if j == 0:
                    label_levels[i] = 0
                else:
                    prev_i = labeled_idx[j - 1]
                    if items[i][0] - items[prev_i][0] < min_label_gap:
                        label_levels[i] = 1 - label_levels[prev_i]
                    else:
                        label_levels[i] = 0

            for i, (pos, cls, dlab, site, do_label) in enumerate(items):
                color = dir_color[dlab]
                ms, ec, elw = marker_style(cls)
                base_h = (0.48 if cls == "Known" else 0.38 if cls == "HC Predicted" else 0.24) * stem_scale
                level = label_levels.get(i, 0)
                stem_h = y0 + base_h + (0.22 * stem_scale if level else 0.0)
                ax.plot(
                    [pos, pos],
                    [y0 + bar_half, stem_h],
                    color=color,
                    lw=1.35 if do_label else 0.85,
                    zorder=evid_z[cls],
                    solid_capstyle="butt",
                )
                ax.scatter(
                    [pos],
                    [stem_h],
                    s=ms * ms_scale,
                    c=color,
                    edgecolors=ec,
                    linewidths=elw,
                    zorder=evid_z[cls] + 0.5,
                )
                if do_label:
                    ax.text(
                        pos,
                        stem_h + 0.04 * stem_scale,
                        site,
                        ha="center",
                        va="bottom",
                        fontsize=label_fs,
                        rotation=label_rot,
                        color=C_TEXT,
                        clip_on=False,
                    )

        fig = plt.figure(figsize=(9.0, 3.35))
        # Main track; magnifier close under the sequence bar
        ax = fig.add_axes([0.08, 0.34, 0.86, 0.52])
        hs_mid_frac = (0.5 * (hs0 + hs1)) / max(length, 1)
        # Fixed inset size for all exemplars (unified cone / callout geometry)
        mag_w, mag_h = 0.36, 0.20
        mag_x = float(np.clip(hs_mid_frac - mag_w / 2, 0.08, 0.92 - mag_w))
        axm = fig.add_axes([mag_x, 0.155, mag_w, mag_h])

        # --- main track: sequence bar + hotspot band only (no dense sites inside) ---
        draw_sequence_bar(ax, 1, length, y=0.0, height=0.07)
        ax.axvspan(hs0, hs1, ymin=0.40, ymax=0.60, color=C_HS_BG, alpha=0.95, zorder=0, lw=0)

        # Sites outside hotspot only on the full-length track
        g_out = g.loc[~g["position"].between(hs0, hs1)]
        g_in = g.loc[g["position"].between(hs0, hs1)]
        draw_lollipops(
            ax,
            g_out,
            label_all=False,
            stem_scale=1.0,
            ms_scale=1.0,
            label_fs=6.5,
            label_rot=0,
            y0=0.0,
            bar_half=0.035,
            min_label_gap=max(22.0, 0.035 * length),
        )

        # Termini at bar ends: two lines stacked; block center aligned to gray bar
        ax.text(
            1,
            0.0,
            "N\n1",
            ha="right",
            va="center",
            fontsize=7.5,
            fontweight="normal",
            color=C_BAR,
            linespacing=0.95,
            clip_on=False,
        )
        ax.text(
            length,
            0.0,
            f"{length}\nC",
            ha="left",
            va="center",
            fontsize=7.5,
            fontweight="normal",
            color=C_BAR,
            linespacing=0.95,
            clip_on=False,
        )
        ax.set_xlim(-0.05 * length, length * 1.05)
        ax.set_ylim(-0.35, 1.15)
        ax.set_yticks([])
        ax.set_xticks([])
        for sp in ("left", "top", "right", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.text(
            0.0,
            1.02,
            f"{ex['title']}  ·  {ex['subtitle']}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color=C_TEXT,
            clip_on=False,
        )

        # --- compact magnifier (no title) ---
        pad = max(2, int(round(0.15 * max(hs1 - hs0, 1))))
        z0, z1 = hs0 - pad, hs1 + pad
        axm.set_facecolor("#FFFBF0")
        for sp in ("top", "right", "left", "bottom"):
            axm.spines[sp].set_color("#C4A35A")
            axm.spines[sp].set_linewidth(1.2)

        g_zoom = g.loc[g["position"].between(hs0, hs1)].sort_values("position").reset_index(drop=True)
        n = len(g_zoom)
        if n >= 3:
            # Evenly spaced slots; alternate stem heights so horizontal labels don't collide
            slot = 1.90 if n >= 8 else 1.60
            disp = 1.0 + slot * np.arange(n, dtype=float)
            x_right = float(disp[-1] + 1.0)
            draw_sequence_bar(axm, 0.45, x_right - 0.45, y=0.0, height=0.07)
            for k, row in g_zoom.iterrows():
                cls = str(row["site_class"])
                dlab = str(row["dir_lab"])
                color = dir_color[dlab]
                ms, ec, elw = marker_style(cls)
                x = float(disp[k])
                stem_h = 0.72 if (k % 2) else 0.36
                axm.plot([x, x], [0.07, stem_h], color=color, lw=1.3, zorder=evid_z[cls])
                axm.scatter(
                    [x],
                    [stem_h],
                    s=ms * 0.85,
                    c=color,
                    edgecolors=ec,
                    linewidths=elw,
                    zorder=evid_z[cls] + 0.5,
                )
                axm.text(
                    x,
                    stem_h + 0.05,
                    str(row["site"]),
                    ha="center",
                    va="bottom",
                    fontsize=6.2 if n >= 8 else 7.0,
                    rotation=0,
                    color=C_TEXT,
                    clip_on=False,
                )
            axm.set_xlim(-0.1, x_right + 0.2)
            axm.set_xticks([])
            axm.set_ylim(-0.20, 1.15)
        elif n > 0:
            draw_sequence_bar(axm, z0, z1, y=0.0, height=0.07)
            draw_lollipops(
                axm,
                g_zoom,
                label_all=True,
                stem_scale=1.0,
                ms_scale=1.1,
                label_fs=7.5,
                label_rot=0,
                y0=0.0,
            )
            axm.set_xlim(z0 - 0.6, z1 + 0.6)
            axm.set_xticks([hs0, hs1] if hs0 != hs1 else [hs0])
            axm.tick_params(axis="x", labelsize=6.5, length=2, colors="#8B6914")
            axm.set_ylim(-0.25, 1.20)
        else:
            axm.set_ylim(-0.25, 1.20)

        axm.set_yticks([])

        # Projection cone: hotspot bottom → full inset top (same inset size → unified look)
        from matplotlib.patches import Polygon

        fig.canvas.draw()
        inv = fig.transFigure.inverted()
        y_bar = -0.02
        p_tl = inv.transform(ax.transData.transform((hs0, y_bar)))
        p_tr = inv.transform(ax.transData.transform((hs1, y_bar)))
        p_bl = inv.transform(axm.transAxes.transform((0.0, 1.0)))
        p_br = inv.transform(axm.transAxes.transform((1.0, 1.0)))
        fan = Polygon(
            [p_tl, p_tr, p_br, p_bl],
            closed=True,
            transform=fig.transFigure,
            facecolor="#E8D5A3",
            edgecolor="none",
            alpha=0.30,
            zorder=0,
            clip_on=False,
        )
        fig.patches.append(fan)
        for (a, b) in ((p_tl, p_bl), (p_tr, p_br)):
            fig.add_artist(
                Line2D(
                    [a[0], b[0]],
                    [a[1], b[1]],
                    transform=fig.transFigure,
                    color="#C4A35A",
                    lw=0.7,
                    alpha=0.75,
                    clip_on=False,
                    zorder=0,
                )
            )

        # Legend: color = direction; size/edge = evidence
        legend_elems = [
            Line2D(
                [0],
                [0],
                marker="o",
                color=C_NUC,
                markerfacecolor=C_NUC,
                markeredgecolor="none",
                markersize=7,
                lw=0,
                label="Nuclear accumulation",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color=C_CYTO,
                markerfacecolor=C_CYTO,
                markeredgecolor="none",
                markersize=7,
                lw=0,
                label="Cytoplasmic redistribution",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color=C_UNRES,
                markerfacecolor=C_UNRES,
                markeredgecolor="none",
                markersize=7,
                lw=0,
                label="Unresolved",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="#888888",
                markerfacecolor="#DDDDDD",
                markeredgecolor="#1A1A1A",
                markeredgewidth=1.2,
                markersize=8.5,
                lw=0,
                label="Known",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="#888888",
                markerfacecolor="#DDDDDD",
                markeredgecolor="white",
                markeredgewidth=0.6,
                markersize=6.5,
                lw=0,
                label="Predicted",
            ),
            Patch(facecolor=C_HS_BG, edgecolor="#C4A35A", linewidth=0.8, label="Hotspot region"),
        ]
        fig.legend(
            handles=legend_elems,
            loc="lower center",
            ncol=3,
            frameon=False,
            fontsize=7.5,
            bbox_to_anchor=(0.5, 0.005),
            columnspacing=1.2,
            handletextpad=0.4,
        )

        stem = out_dir / f"figure4c_{gene}_full_length_psp"
        fig.savefig(f"{stem}.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        counts = g["site_class"].value_counts().to_dict()
        dir_counts = g_in["dir_lab"].value_counts().to_dict() if len(g_in) else {}
        used.append(
            {
                "gene": gene,
                "uniprot": acc,
                "length": length,
                "hotspot": f"{hs0}-{hs1}",
                "n_sites_total": int(len(g)),
                "n_known": int(counts.get("Known", 0)),
                "n_hc_predicted": int(counts.get("HC Predicted", 0)),
                "n_other_observed": int(counts.get("Other observed", 0)),
                "n_sites_in_hotspot_zoom": int(len(g_in)),
                "hotspot_direction_counts": {str(k): int(v) for k, v in dir_counts.items()},
                "figure": str(stem),
            }
        )

    summary = {
        "style": (
            "Full-length sequence bar with hotspot callout inset; color encodes "
            "direction (nuclear / cytoplasmic / unresolved); marker size+edge encode "
            "evidence (Known = large+black edge, Predicted = standard); sites inside "
            "hotspot shown only in the inset"
        ),
        "exemplars": used,
    }
    (out_dir / "figure4c_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Figure S5A — Predicted→Known proximity
# ---------------------------------------------------------------------------

def run_figures5a(sites: pd.DataFrame, out_dir: Path, n_perm: int, rng: np.random.Generator) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    dual = []
    site_rows = []
    for acc, g in sites.groupby("protein_acc"):
        known = g.loc[g["is_known"]]
        # In this release Predicted ≡ HC Predicted
        pred = g.loc[g["is_predicted"] & ~g["is_known"]]
        if len(known) == 0 or len(pred) == 0:
            continue
        known_pos = known["position"].to_numpy(dtype=float)
        gene = str(g["gene_name"].iloc[0])
        for _, row in pred.iterrows():
            d = float(np.min(np.abs(known_pos - float(row["position"]))))
            site_rows.append(
                {
                    "INDEX": row["INDEX"],
                    "TF": gene,
                    "UniProt": acc,
                    "site": row["site"],
                    "residue": row["residue"],
                    "position": int(row["position"]),
                    "dist_to_nearest_known": d,
                    "is_hc_predicted": bool(row["is_hc_predicted"]),
                }
            )
        bg_non_known = g.loc[~g["is_known"]]
        if len(bg_non_known) < len(pred):
            continue
        dual.append(
            {
                "known_pos": known_pos,
                "target_res": pred["residue"].tolist(),
                "bg_pos": bg_non_known["position"].to_numpy(dtype=float),
                "bg_res": bg_non_known["residue"].to_numpy(),
            }
        )

    site_df = pd.DataFrame(site_rows)
    site_df.to_csv(out_dir / "figures5a_predicted_to_known_distances.csv", index=False)
    obs = site_df["dist_to_nearest_known"].to_numpy(dtype=float)

    ecdf_pool = []
    null_medians = []
    ecdf_every = max(1, n_perm // 200)
    for p in range(n_perm):
        pooled = []
        for td in dual:
            samp = sty_matched_sample(td["bg_pos"], td["bg_res"], td["target_res"], rng)
            if samp is None:
                continue
            for x in samp:
                pooled.append(float(np.min(np.abs(td["known_pos"] - x))))
        if pooled:
            null_medians.append(float(np.median(pooled)))
            if p % ecdf_every == 0:
                ecdf_pool.append(np.asarray(pooled, dtype=float))

    obs_med = float(np.median(obs)) if len(obs) else np.nan
    p_left = empirical_p_left(obs_med, np.asarray(null_medians, dtype=float))

    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    grid = np.linspace(0, 150, 301)
    if len(obs):
        ys = np.searchsorted(np.sort(obs), grid, side="right") / len(obs)
        ax.plot(grid, ys, color=C_OBS, lw=2.0, label="HC Predicted")
    if ecdf_pool:
        null_ys = [np.searchsorted(np.sort(a), grid, side="right") / len(a) for a in ecdf_pool]
        ax.plot(grid, np.mean(null_ys, axis=0), color=C_NULL, lw=2.0, label="Null (mean ECDF)")
    ax.set_xlabel("Distance to nearest Known (aa)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.text(0.5, 0.92, p_to_star(p_left), transform=ax.transAxes, ha="center", fontsize=12, color=C_TEXT)
    fig.tight_layout()
    save_fig(fig, out_dir / "figures5a_proximity_to_known_ecdf")

    summary = {
        "n_predicted_non_known": int(len(site_df)),
        "n_hc_predicted": int(site_df["is_hc_predicted"].sum()) if len(site_df) else 0,
        "obs_median_dist_hc": obs_med,
        "null_mean_median_dist": float(np.mean(null_medians)) if null_medians else np.nan,
        "empirical_p_left": p_left,
        "note": (
            "Supporting observation; not independent validation. "
            "Only HC Predicted is plotted because Predicted ≡ HC Predicted in this release."
        ),
    }
    (out_dir / "figures5a_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Figure S5B — exploratory: local hotspots vs literature multi-site regions
# ---------------------------------------------------------------------------

def dist_to_nearest_interval(pos: int, intervals: Sequence[Tuple[int, int]]) -> float:
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


def run_figures5b(
    sites: pd.DataFrame,
    cluster_cat: pd.DataFrame,
    out_dir: Path,
    n_perm: int,
    rng: np.random.Generator,
) -> Dict:
    """Exploratory overlap of literature Cluster sites with Cluster-blind hotspots.

    Not a validation panel. Completely hold out ALL literature Cluster sites
    (n≈104), including dual-labeled Predicted;Cluster / Known∩Cluster, from
    S5B-only hotspot construction; then bring them back and ask whether they
    fall inside the resulting local phospho-hotspots more often than a
    within-TF STY-matched null.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Complete hold-out: any site whose evidence contains "Cluster"
    cluster_sites = sites.loc[
        sites["evidence"].astype(str).str.contains("Cluster", regex=False)
    ].copy()
    # Also include literature cluster catalog INDEX if present but not tagged in evidence
    if cluster_cat is not None and len(cluster_cat) and "INDEX" in cluster_cat.columns:
        lit_idx = set(cluster_cat["INDEX"].astype(str))
        extra = sites.loc[
            sites["INDEX"].astype(str).isin(lit_idx)
            & (~sites["INDEX"].astype(str).isin(set(cluster_sites["INDEX"].astype(str))))
        ]
        if len(extra):
            cluster_sites = pd.concat([cluster_sites, extra], ignore_index=True)
    cluster_idx = set(cluster_sites["INDEX"].astype(str))

    # Seeds for S5B-only catalog: Known ∪ HC Predicted, but NEVER any Cluster site
    seeds = sites.loc[
        (sites["is_known"] | sites["is_hc_predicted"])
        & (~sites["INDEX"].astype(str).isin(cluster_idx))
    ].copy()
    catalog, members = build_catalog_from_seeds(seeds, ADJ_MAX, SPAN_MAX)
    if "INDEX" not in members.columns:
        members = members.copy()
        members["INDEX"] = members["UniProt"].astype(str) + "_" + members["site"].astype(str)
    n_cluster_in_members = int(members["INDEX"].astype(str).isin(cluster_idx).sum())
    assert n_cluster_in_members == 0, (
        f"Hold-out failed: {n_cluster_in_members} Cluster sites appear as hotspot members"
    )
    catalog.to_csv(out_dir / "figures5b_cluster_blind_hotspots.csv", index=False)
    members.to_csv(out_dir / "figures5b_cluster_blind_members.csv", index=False)

    intervals_by_acc: Dict[str, List[Tuple[int, int]]] = {}
    for acc, g in catalog.groupby("UniProt"):
        intervals_by_acc[str(acc)] = list(zip(g["start"].astype(int), g["end"].astype(int)))

    test_sites = cluster_sites
    obs_inside = []
    site_rows = []
    for _, row in test_sites.iterrows():
        acc = str(row["protein_acc"])
        intervals = intervals_by_acc.get(acc, [])
        d = dist_to_nearest_interval(int(row["position"]), intervals)
        inside = bool(np.isfinite(d) and d == 0)
        obs_inside.append(inside)
        site_rows.append(
            {
                "INDEX": row["INDEX"],
                "TF": row["gene_name"],
                "UniProt": acc,
                "site": row["site"],
                "residue": row["residue"],
                "position": int(row["position"]),
                "evidence": str(row["evidence"]),
                "held_out": True,
                "dist_to_nearest_blind_hotspot": d,
                "inside_blind_hotspot": inside,
            }
        )
    site_df = pd.DataFrame(site_rows)
    site_df.to_csv(out_dir / "figures5b_cluster_site_distances.csv", index=False)
    obs_frac = float(np.mean(obs_inside)) if obs_inside else np.nan
    n_inside = int(np.sum(obs_inside))

    # Null: within-TF STY from non-Cluster phosphosites (outside blind intervals preferred)
    tf_null_jobs: List[Dict] = []
    for acc, g_cl in test_sites.groupby("protein_acc"):
        acc = str(acc)
        bg = sites.loc[
            (sites["protein_acc"].astype(str) == acc)
            & (~sites["INDEX"].astype(str).isin(cluster_idx))
        ].copy()
        intervals = intervals_by_acc.get(acc, [])
        if intervals and len(bg):
            pos = bg["position"].to_numpy(dtype=int)
            outside = np.ones(len(bg), dtype=bool)
            for a, b in intervals:
                outside &= ~((pos >= a) & (pos <= b))
            bg_out = bg.loc[outside]
            if len(bg_out) >= len(g_cl):
                bg = bg_out
        if len(bg) < len(g_cl):
            bg = sites.loc[sites["protein_acc"].astype(str) == acc]
        tf_null_jobs.append(
            {
                "acc": acc,
                "n": int(len(g_cl)),
                "target_res": g_cl["residue"].tolist(),
                "bg_pos": bg["position"].to_numpy(dtype=float),
                "bg_res": bg["residue"].to_numpy(),
                "intervals": intervals,
            }
        )

    null_frac = np.empty(n_perm, dtype=float)
    for p in range(n_perm):
        flags: List[float] = []
        for job in tf_null_jobs:
            samp = sty_matched_sample(job["bg_pos"], job["bg_res"], job["target_res"], rng)
            if samp is None:
                continue
            for pos in samp:
                d = dist_to_nearest_interval(int(pos), job["intervals"])
                flags.append(1.0 if d == 0 else 0.0)
        null_frac[p] = float(np.mean(flags)) if flags else np.nan

    p_right = empirical_p_right(obs_frac, null_frac)
    null_mean = float(np.nanmean(null_frac))
    finite = null_frac[np.isfinite(null_frac)]
    null_lo, null_hi = (
        np.percentile(finite, [2.5, 97.5]) if len(finite) else (np.nan, np.nan)
    )

    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.bar(
        [0],
        [obs_frac],
        width=0.55,
        color=C_OBS,
        edgecolor=C_OBS,
        linewidth=0,
        label="Held-out Cluster",
        zorder=3,
    )
    ax.bar(
        [1],
        [null_mean],
        width=0.55,
        color=C_NULL,
        edgecolor=C_NULL,
        linewidth=0,
        label="Null",
        zorder=3,
    )
    ax.errorbar(
        [1],
        [null_mean],
        yerr=[[max(0.0, null_mean - null_lo)], [max(0.0, null_hi - null_mean)]],
        fmt="none",
        ecolor="#555555",
        capsize=4,
        lw=1.2,
        zorder=4,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Held-out\nCluster", "Null"])
    ax.set_ylabel("Fraction of sites inside\nCluster-blind hotspots")
    y_top = max(obs_frac, null_hi if np.isfinite(null_hi) else 0.0, null_mean)
    ax.text(
        0.5,
        y_top * 1.06,
        p_to_star(p_right),
        ha="center",
        fontsize=12,
        color=C_TEXT,
    )
    ax.set_ylim(0, y_top * 1.28)
    ax.set_title(
        "Comparison between local phospho-hotspots and\n"
        "literature-supported multi-site regulatory regions",
        fontsize=9.5,
        pad=8,
        color=C_TEXT,
    )
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors="black", direction="out", length=3.5, width=1.0)
    ax.patch.set_edgecolor("none")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    save_fig(fig, out_dir / "figures5b_cluster_blind_enrichment")

    summary = {
        "panel_role": "exploratory_comparison_not_validation",
        "title": (
            "Comparison between local phospho-hotspots and "
            "literature-supported multi-site regulatory regions"
        ),
        "interpretation": (
            "Overlap between Cluster-blind local phospho-hotspots and literature "
            "multi-site Cluster regions was limited; the two may represent partially "
            "distinct regulatory architectures. Suitable for Discussion / SI, not as "
            "external validation of hotspot calling."
        ),
        "seed_definition": (
            "Known (PMID) + HC Predicted for S5B-only construction, with ALL literature "
            "Cluster sites completely held out (including Predicted;Cluster and any "
            "Known∩Cluster). Held-out sites are tested afterward for interval overlap."
        ),
        "n_seeds_used": int(len(seeds)),
        "n_cluster_held_out": int(len(cluster_sites)),
        "n_cluster_in_members_assert": n_cluster_in_members,
        "n_blind_hotspots": int(len(catalog)),
        "n_blind_tfs": int(catalog["TF"].nunique()) if len(catalog) else 0,
        "n_cluster_sites_tested": int(len(site_df)),
        "n_inside": n_inside,
        "obs_frac_inside": obs_frac,
        "null_mean_frac_inside": null_mean,
        "null_lo": float(null_lo),
        "null_hi": float(null_hi),
        "enrichment_vs_null": obs_frac / null_mean if null_mean and null_mean > 0 else np.nan,
        "empirical_p_right": p_right,
        "star": p_to_star(p_right),
        "mixed": int((catalog["hotspot_type"] == "Mixed evidence").sum()) if len(catalog) else 0,
        "predicted_candidate": int((catalog["hotspot_type"] == "Predicted candidate").sum())
        if len(catalog)
        else 0,
        "null_definition": (
            "Within-TF STY-matched: for each TF with held-out Cluster sites, redraw the "
            "same n sites with the same S/T/Y composition from that TF's non-Cluster "
            "phosphosite pool, preferentially outside Cluster-blind hotspot intervals "
            "(fallback to all non-Cluster sites if outside pool too small)."
        ),
    }
    (out_dir / "figures5b_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame({"null_frac_inside": null_frac}).to_csv(
        out_dir / "figures5b_null_frac_inside.csv", index=False
    )
    return summary


def coherent_fraction_for_catalog(members: pd.DataFrame) -> Tuple[int, int, float]:
    if members.empty:
        return 0, 0, np.nan
    n_coh = 0
    n_elig = 0
    for _, g in members.groupby("hotspot_id"):
        labs = [d for d in g["predicted_direction"].tolist() if d in {"Import", "Export"}]
        if len(labs) < 2:
            continue
        n_elig += 1
        if is_fully_coherent(labs):
            n_coh += 1
    frac = n_coh / n_elig if n_elig else np.nan
    return n_coh, n_elig, frac


def run_figures5c(sites: pd.DataFrame, out_dir: Path) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Main catalog seed rule (uses Cluster-as-Known via PMID, matching Stage-2 catalog)
    seeds = sites.loc[sites["is_known"] | sites["is_predicted"]].copy()
    rows = []
    for adj in SENS_ADJ:
        for span in SENS_SPAN:
            catalog, members = build_catalog_from_seeds(seeds, adj, span)
            n_coh, n_elig, frac = coherent_fraction_for_catalog(members)
            rows.append(
                {
                    "adj_max": adj,
                    "span_max": span,
                    "n_hotspots": int(len(catalog)),
                    "n_TF": int(catalog["TF"].nunique()) if len(catalog) else 0,
                    "n_eligible_coherence": n_elig,
                    "n_fully_coherent": n_coh,
                    "directionally_coherent_fraction": frac,
                    "is_main": adj == ADJ_MAX and span == SPAN_MAX,
                }
            )
    sens = pd.DataFrame(rows)
    sens.to_csv(out_dir / "figures5c_sensitivity_grid.csv", index=False)

    def pivot(col: str) -> np.ndarray:
        mat = np.full((len(SENS_ADJ), len(SENS_SPAN)), np.nan)
        for i, adj in enumerate(SENS_ADJ):
            for j, span in enumerate(SENS_SPAN):
                val = sens.loc[(sens.adj_max == adj) & (sens.span_max == span), col]
                mat[i, j] = float(val.iloc[0]) if len(val) else np.nan
        return mat

    # Soft single-hue tile grids (one palette per panel). Numbers primary; color secondary.
    from matplotlib.colors import LinearSegmentedColormap, to_rgb

    cmaps = {
        "n_hotspots": LinearSegmentedColormap.from_list(
            "hotspot_orange", ["#FCE8D5", "#C66A2B"]
        ),
        "n_TF": LinearSegmentedColormap.from_list(
            "tf_blue", ["#E5F0F7", "#3E78A8"]
        ),
        "directionally_coherent_fraction": LinearSegmentedColormap.from_list(
            "coherent_green", ["#E7F2E7", "#4A8A5B"]
        ),
    }
    outline = {
        "n_hotspots": "#C66A2B",
        "n_TF": "#3E78A8",
        "directionally_coherent_fraction": "#4A8A5B",
    }
    panels = [
        ("n_hotspots", "Hotspot number", False),
        ("n_TF", "TF number", False),
        ("directionally_coherent_fraction", "Coherent fraction", True),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(8.8, 3.0),
        sharey=True,
        gridspec_kw={"wspace": 0.28},
    )
    gap = 0.08  # white gutter between tiles

    for ax, (col, title, is_frac) in zip(axes, panels):
        mat = pivot(col)
        cmap = cmaps[col]
        vmin, vmax = float(np.nanmin(mat)), float(np.nanmax(mat))
        # Expand tiny ranges so fill still readable
        if np.isclose(vmin, vmax):
            vmin, vmax = vmin * 0.95, vmax * 1.05 if vmax else (0.0, 1.0)
        ax.set_xlim(-0.5, len(SENS_SPAN) - 0.5)
        ax.set_ylim(len(SENS_ADJ) - 0.5, -0.5)  # adj 10 on top
        ax.set_aspect("equal")
        ax.set_xticks(range(len(SENS_SPAN)))
        ax.set_xticklabels([str(s) for s in SENS_SPAN], fontsize=9)
        ax.set_yticks(range(len(SENS_ADJ)))
        ax.set_yticklabels([str(a) for a in SENS_ADJ], fontsize=9)
        ax.set_xlabel("Span (aa)", fontsize=9)
        ax.set_title(title, fontsize=10, pad=8, color=C_TEXT)
        for sp in ("top", "right", "bottom", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=0)

        for i, adj in enumerate(SENS_ADJ):
            for j, span in enumerate(SENS_SPAN):
                val = mat[i, j]
                t = 0.0 if vmax == vmin else (val - vmin) / (vmax - vmin)
                face = cmap(t)
                is_main = adj == ADJ_MAX and span == SPAN_MAX
                rect = plt.Rectangle(
                    (j - 0.5 + gap / 2, i - 0.5 + gap / 2),
                    1 - gap,
                    1 - gap,
                    facecolor=face,
                    edgecolor=outline[col] if is_main else "none",
                    linewidth=1.4 if is_main else 0.0,
                    zorder=1,
                )
                ax.add_patch(rect)
                txt = f"{val:.2f}" if is_frac else f"{int(val)}"
                # Light text on darker tiles for contrast
                r, g, b = to_rgb(face)
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#FFFFFF" if luminance < 0.55 else "#1F2A2E",
                    fontweight="medium" if is_main else "normal",
                    zorder=2,
                )

    axes[0].set_ylabel("Adjacent gap (aa)", fontsize=9)
    fig.suptitle(
        "Parameter sensitivity  ·  outline = main (15 / 40)",
        fontsize=10,
        color="#555555",
        y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, out_dir / "figures5c_sensitivity_heatmap")

    summary = {
        "grid": rows,
        "main_setting": {"adj_max": ADJ_MAX, "span_max": SPAN_MAX},
        "interpretation": (
            "Sensitivity of Stage-2 hotspot calling to clustering parameters. "
            "Each cell rebuilds Mixed/Predicted-candidate hotspots under a given "
            "max adjacent gap × max span, then reports catalog size (# hotspots, # TFs) "
            "and the fraction of eligible hotspots that are fully directionally coherent. "
            "Shows that the main 15×40 setting is not an isolated outlier."
        ),
    }
    (out_dir / "figures5c_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_readme(out: Path, summaries: Dict) -> None:
    s4a = summaries["4a"]
    s4b = summaries["4b"]
    s5a = summaries["s5a"]
    s5b = summaries["s5b"]
    lines = [
        "# Figure 4 + Figure S5 spatial / direction panels",
        "",
        "## Figure 4A — HC Predicted close-pair enrichment",
        f"- TFs with ≥2 HC Predicted: **{s4a['n_tfs_with_ge2_hc']}**",
    ]
    for row in s4a["enrichment_table"]:
        lines.append(
            f"- {row['distance_aa']} aa: enrichment **{row['enrichment_obs_over_expected']:.3f}** "
            f"({row['star']}; obs pairs={row['obs_close_pairs']}, null mean={row['null_mean_close_pairs']:.1f})"
        )
    lines += [
        "",
        "## Figure 4B — Directional coherence vs matched pseudo-hotspots",
        f"- {s4b['annotation']}",
        f"- Observed coherent fraction: **{s4b['obs_coherent_fraction']:.3f}**",
        f"- Null mean: **{s4b['null_mean_coherent_fraction']:.3f}** ({s4b['star']})",
        f"- Null: {s4b['null_definition']}",
        "",
        "## Figure 4C — Representative hotspots (PSP-style full-length)",
        "- Separate files: STAT3 / AR / SIX4 full-length tracks with Known, HC Predicted, other observed sites; hotspot span shaded.",
        "",
        "## Figure S5A — Predicted→Known proximity (supporting)",
        f"- HC median dist→Known: **{s5a['obs_median_dist_hc']:.3f}** vs null **{s5a['null_mean_median_dist']:.3f}** ({p_to_star(s5a['empirical_p_left'])})",
        f"- Note: {s5a['note']}",
        "",
        "## Figure S5B — Exploratory: local hotspots vs literature multi-site regions",
        f"- Role: **{s5b.get('panel_role', 'exploratory')}** (not validation)",
        f"- Title: {s5b.get('title', '')}",
        f"- Completely held out **{s5b.get('n_cluster_held_out', 'NA')}** Cluster sites from S5B-only construction "
        f"(blind catalog: **{s5b.get('n_blind_hotspots', 'NA')}** hotspots / **{s5b.get('n_blind_tfs', 'NA')}** TFs)",
        f"- Inside: **{s5b.get('n_inside', 'NA')}/{s5b.get('n_cluster_sites_tested', 'NA')}** "
        f"(obs **{s5b.get('obs_frac_inside', float('nan')):.3f}** vs null **{s5b.get('null_mean_frac_inside', float('nan')):.3f}**, {s5b.get('star', '')})",
        f"- Interpretation: {s5b.get('interpretation', '')}",
        f"- Null: {s5b.get('null_definition', '')}",
        "",
        "## Figure S5C — Sensitivity heatmap",
        "- Logic: rebuild catalog across adj∈{10,15,20} × span∈{30,40,50}; report #hotspots, #TFs, coherent fraction.",
        "- Shows main 15×40 is stable (marked with *); not a cherry-picked outlier.",
        "",
    ]
    (out / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--hotspot-dir", type=Path, default=DEFAULT_HOTSPOT_DIR)
    parser.add_argument("--cluster-xlsx", type=Path, default=DEFAULT_CLUSTER_XLSX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--skip-4a", action="store_true", help="Reuse existing Figure 4A outputs")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional subset: 4b 4c s5a s5b s5c",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    only = set(args.only) if args.only else None

    print("Loading site master…", flush=True)
    sites, cluster_cat = load_site_master(args.summary, args.cluster_xlsx)
    sites.to_csv(out / "site_master.csv", index=False)
    cluster_cat.to_csv(out / "literature_cluster_catalog.csv", index=False)

    print("Loading fixed 99-hotspot catalog…", flush=True)
    catalog, members = load_fixed_catalog(args.hotspot_dir, sites)

    def want(tag: str) -> bool:
        return only is None or tag in only

    if args.skip_4a and (out / "figure4a" / "figure4a_summary.json").exists():
        print("Skipping Figure 4A (reuse)…", flush=True)
        s4a = json.loads((out / "figure4a" / "figure4a_summary.json").read_text())
    elif want("4a"):
        print("Figure 4A…", flush=True)
        s4a = run_figure4a(sites, out / "figure4a", args.n_perm, rng)
    else:
        s4a = json.loads((out / "figure4a" / "figure4a_summary.json").read_text()) if (out / "figure4a" / "figure4a_summary.json").exists() else {}

    if want("4b"):
        print("Figure 4B…", flush=True)
        s4b = run_figure4b(sites, catalog, members, out / "figure4b", args.n_perm, rng)
    else:
        s4b = json.loads((out / "figure4b" / "figure4b_summary.json").read_text()) if (out / "figure4b" / "figure4b_summary.json").exists() else {}

    if want("4c"):
        print("Figure 4C…", flush=True)
        s4c = run_figure4c(sites, members, out / "figure4c")
    else:
        s4c = json.loads((out / "figure4c" / "figure4c_summary.json").read_text()) if (out / "figure4c" / "figure4c_summary.json").exists() else {}

    if want("s5a"):
        print("Figure S5A…", flush=True)
        s5a = run_figures5a(sites, out / "figures5a", args.n_perm, rng)
    else:
        s5a = json.loads((out / "figures5a" / "figures5a_summary.json").read_text()) if (out / "figures5a" / "figures5a_summary.json").exists() else {}

    if want("s5b"):
        print("Figure S5B…", flush=True)
        s5b = run_figures5b(sites, cluster_cat, out / "figures5b", args.n_perm, rng)
    else:
        s5b = json.loads((out / "figures5b" / "figures5b_summary.json").read_text()) if (out / "figures5b" / "figures5b_summary.json").exists() else {}

    if want("s5c"):
        print("Figure S5C…", flush=True)
        s5c = run_figures5c(sites, out / "figures5c")
    else:
        s5c = json.loads((out / "figures5c" / "figures5c_summary.json").read_text()) if (out / "figures5c" / "figures5c_summary.json").exists() else {}

    all_sum = {"4a": s4a, "4b": s4b, "4c": s4c, "s5a": s5a, "s5b": s5b, "s5c": s5c}
    (out / "all_panels_summary.json").write_text(json.dumps(all_sum, indent=2, default=str))
    if all(k in all_sum and all_sum[k] for k in ("4a", "4b", "s5a", "s5b")):
        write_readme(out, all_sum)
    print(json.dumps({"4b": s4b, "s5b": s5b}, indent=2, default=str), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
