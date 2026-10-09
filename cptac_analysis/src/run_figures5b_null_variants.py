#!/usr/bin/env python3
"""S5B null / statistic variants (sequential sensitivity).

Shared construction: hold out ALL literature Cluster sites (n≈104) from hotspot
building; Known + HC Predicted seeds only; then bring Cluster sites back.

Variants
  1 outside_pool_sty   STY null from non-Cluster sites outside blind hotspot intervals
  2 exclude_seeds_sty  STY null from non-Cluster & non-(Known∪HC) phosphosites
  3 sequence_sty       STY null from all S/T/Y residues on the protein sequence
  4 distance_stat      statistic = median dist→nearest hotspot (left-tail), STY null
                       on non-Cluster phosphosites (baseline pool)
  5 tf_with_hotspot    inside-fraction enrichment only on Cluster sites whose TF has
                       ≥1 blind hotspot; STY null on non-Cluster phosphosites

Also reports the current baseline (all non-Cluster phosphosite STY null + inside)
as variant 0 for comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_figure4_spatial_direction_panels import (  # noqa: E402
    ADJ_MAX,
    DEFAULT_CLUSTER_XLSX,
    DEFAULT_OUT,
    DEFAULT_SUMMARY,
    N_PERM,
    RNG_SEED,
    SPAN_MAX,
    build_catalog_from_seeds,
    dist_to_nearest_interval,
    empirical_p_left,
    empirical_p_right,
    load_site_master,
    p_to_star,
    save_fig,
    sty_matched_sample,
)

mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

FASTA_CANDIDATES = (
    REPO / "functional/data/fasta/uniprotkb_AND_reviewed_true_AND_model_o_2026_04_08.fasta",
    REPO / "import_export/data/fasta/uniprotkb_AND_reviewed_true_AND_model_o_2026_04_08.fasta",
)
C_OBS = "#4DBBD5"
C_NULL = "#9e9e9e"
C_TEXT = "#333333"


def load_fasta_subset(path: Path, wanted: set) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not wanted or not path.exists():
        return out
    acc = None
    chunks: List[str] = []
    with path.open() as fh:
        for line in fh:
            if line.startswith(">"):
                if acc is not None and acc in wanted:
                    out[acc] = "".join(chunks)
                    if len(out) == len(wanted):
                        break
                # header like >sp|P04637|P53_HUMAN ...
                parts = line[1:].split("|")
                acc = parts[1].strip() if len(parts) >= 2 else line[1:].split()[0].strip()
                chunks = []
            else:
                chunks.append(line.strip())
        if acc is not None and acc in wanted and acc not in out:
            out[acc] = "".join(chunks)
    return out


def sty_positions_from_seq(seq: str) -> Tuple[np.ndarray, np.ndarray]:
    pos, res = [], []
    for i, aa in enumerate(seq, start=1):
        if aa in ("S", "T", "Y"):
            pos.append(i)
            res.append(aa)
    return np.asarray(pos, dtype=float), np.asarray(res, dtype=object)


def inside_flags(positions: Sequence[float], intervals: List[Tuple[int, int]]) -> List[float]:
    flags = []
    for pos in positions:
        if not intervals or not np.isfinite(pos):
            flags.append(0.0)
            continue
        d = dist_to_nearest_interval(int(pos), intervals)
        flags.append(1.0 if d == 0 else 0.0)
    return flags


def dist_values(positions: Sequence[float], intervals: List[Tuple[int, int]]) -> List[float]:
    vals = []
    for pos in positions:
        if not intervals or not np.isfinite(pos):
            vals.append(np.nan)
            continue
        vals.append(dist_to_nearest_interval(int(pos), intervals))
    return vals


def bar_plot(obs: float, null_mean: float, null_lo: float, null_hi: float, star: str, ylabel: str, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.8, 3.8))
    ax.bar([0], [obs], width=0.55, color=C_OBS, label="Observed")
    ax.bar([1], [null_mean], width=0.55, color=C_NULL, label="Null")
    ax.errorbar(
        [1],
        [null_mean],
        yerr=[[max(0.0, null_mean - null_lo)], [max(0.0, null_hi - null_mean)]],
        fmt="none",
        ecolor="#555555",
        capsize=4,
        lw=1.2,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Observed", "Null"])
    ax.set_ylabel(ylabel)
    y_top = max(obs, null_hi, null_mean, 0.05)
    ax.text(0.5, y_top * 1.08, star, ha="center", fontsize=12, color=C_TEXT)
    ax.set_ylim(0, max(y_top * 1.3, 0.25))
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    save_fig(fig, stem)


def ecdf_dist_plot(obs_dists: np.ndarray, null_medians: np.ndarray, stem: Path, star: str) -> None:
    obs = obs_dists[np.isfinite(obs_dists)]
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    if len(obs):
        xs = np.sort(obs)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, color=C_OBS, lw=2.0, label="Observed Cluster")
    ax.axvline(np.nanmedian(obs) if len(obs) else np.nan, color=C_OBS, ls="--", lw=1.2)
    ax.axvline(float(np.nanmean(null_medians)), color=C_NULL, ls="--", lw=1.2, label="Null mean median")
    ax.set_xlabel("Distance to nearest blind hotspot (aa)")
    ax.set_ylabel("ECDF")
    ax.text(0.98, 0.05, star, transform=ax.transAxes, ha="right", va="bottom", fontsize=12)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_fig(fig, stem)


def build_jobs(
    test_sites: pd.DataFrame,
    pool_by_acc: Dict[str, pd.DataFrame],
    intervals_by_acc: Dict[str, List[Tuple[int, int]]],
) -> List[Dict]:
    jobs = []
    for acc, g_cl in test_sites.groupby("protein_acc"):
        acc = str(acc)
        bg = pool_by_acc.get(acc)
        if bg is None or len(bg) < len(g_cl):
            continue
        jobs.append(
            {
                "acc": acc,
                "n": int(len(g_cl)),
                "target_res": g_cl["residue"].tolist(),
                "bg_pos": bg["position"].to_numpy(dtype=float),
                "bg_res": bg["residue"].to_numpy(),
                "intervals": intervals_by_acc.get(acc, []),
            }
        )
    return jobs


def run_inside_variant(
    name: str,
    test_sites: pd.DataFrame,
    jobs: List[Dict],
    intervals_by_acc: Dict[str, List[Tuple[int, int]]],
    n_perm: int,
    rng: np.random.Generator,
    out_dir: Path,
    null_definition: str,
) -> Dict:
    obs_flags = []
    for _, row in test_sites.iterrows():
        acc = str(row["protein_acc"])
        intervals = intervals_by_acc.get(acc, [])
        d = dist_to_nearest_interval(int(row["position"]), intervals) if intervals else np.inf
        obs_flags.append(1.0 if d == 0 else 0.0)
    obs_frac = float(np.mean(obs_flags)) if obs_flags else np.nan

    null_frac = np.empty(n_perm, dtype=float)
    for p in range(n_perm):
        flags: List[float] = []
        for job in jobs:
            samp = sty_matched_sample(job["bg_pos"], job["bg_res"], job["target_res"], rng)
            if samp is None:
                continue
            flags.extend(inside_flags(samp, job["intervals"]))
        null_frac[p] = float(np.mean(flags)) if flags else np.nan

    p_right = empirical_p_right(obs_frac, null_frac)
    null_mean = float(np.nanmean(null_frac))
    finite = null_frac[np.isfinite(null_frac)]
    null_lo, null_hi = (
        np.percentile(finite, [2.5, 97.5]) if len(finite) else (np.nan, np.nan)
    )
    star = p_to_star(p_right)
    bar_plot(
        obs_frac,
        null_mean,
        float(null_lo),
        float(null_hi),
        star,
        "Fraction of Cluster sites\ninside blind hotspots",
        out_dir / f"variant_{name}_enrichment",
    )
    pd.DataFrame({"null_frac_inside": null_frac}).to_csv(
        out_dir / f"variant_{name}_null_frac_inside.csv", index=False
    )
    summary = {
        "variant": name,
        "statistic": "frac_inside",
        "n_cluster_tested": int(len(test_sites)),
        "n_inside": int(np.sum(obs_flags)),
        "obs": obs_frac,
        "null_mean": null_mean,
        "null_lo": float(null_lo),
        "null_hi": float(null_hi),
        "empirical_p": p_right,
        "tail": "right",
        "star": star,
        "enrichment_vs_null": obs_frac / null_mean if null_mean and null_mean > 0 else np.nan,
        "n_null_jobs": int(len(jobs)),
        "null_definition": null_definition,
    }
    (out_dir / f"variant_{name}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_distance_variant(
    name: str,
    test_sites: pd.DataFrame,
    jobs: List[Dict],
    intervals_by_acc: Dict[str, List[Tuple[int, int]]],
    n_perm: int,
    rng: np.random.Generator,
    out_dir: Path,
    null_definition: str,
) -> Dict:
    # Only sites on TFs with at least one blind hotspot contribute to median.
    usable = test_sites.loc[
        test_sites["protein_acc"].astype(str).isin(set(intervals_by_acc))
    ].copy()
    obs_dists = []
    for _, row in usable.iterrows():
        acc = str(row["protein_acc"])
        obs_dists.append(dist_to_nearest_interval(int(row["position"]), intervals_by_acc[acc]))
    obs_arr = np.asarray(obs_dists, dtype=float)
    obs_med = float(np.median(obs_arr)) if len(obs_arr) else np.nan

    # Restrict jobs to TFs that have intervals and are in usable set
    usable_acc = set(usable["protein_acc"].astype(str))
    jobs_use = [j for j in jobs if j["acc"] in usable_acc and j["intervals"]]

    null_med = np.empty(n_perm, dtype=float)
    for p in range(n_perm):
        vals: List[float] = []
        for job in jobs_use:
            samp = sty_matched_sample(job["bg_pos"], job["bg_res"], job["target_res"], rng)
            if samp is None:
                continue
            vals.extend(dist_values(samp, job["intervals"]))
        vals_a = np.asarray(vals, dtype=float)
        vals_a = vals_a[np.isfinite(vals_a)]
        null_med[p] = float(np.median(vals_a)) if len(vals_a) else np.nan

    p_left = empirical_p_left(obs_med, null_med)
    null_mean = float(np.nanmean(null_med))
    finite = null_med[np.isfinite(null_med)]
    null_lo, null_hi = (
        np.percentile(finite, [2.5, 97.5]) if len(finite) else (np.nan, np.nan)
    )
    star = p_to_star(p_left)
    ecdf_dist_plot(obs_arr, null_med, out_dir / f"variant_{name}_ecdf", star)
    # also bar on median distance (lower is better) — flip display not needed
    fig, ax = plt.subplots(figsize=(3.8, 3.8))
    ax.bar([0], [obs_med], width=0.55, color=C_OBS, label="Observed")
    ax.bar([1], [null_mean], width=0.55, color=C_NULL, label="Null")
    ax.errorbar(
        [1],
        [null_mean],
        yerr=[[max(0.0, null_mean - null_lo)], [max(0.0, null_hi - null_mean)]],
        fmt="none",
        ecolor="#555555",
        capsize=4,
        lw=1.2,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Observed", "Null"])
    ax.set_ylabel("Median distance to nearest\nblind hotspot (aa)")
    ax.text(0.5, max(obs_med, null_hi) * 1.05, star, ha="center", fontsize=12)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_fig(fig, out_dir / f"variant_{name}_median_distance")
    pd.DataFrame({"null_median_dist": null_med}).to_csv(
        out_dir / f"variant_{name}_null_median_dist.csv", index=False
    )
    summary = {
        "variant": name,
        "statistic": "median_dist",
        "n_cluster_tested": int(len(usable)),
        "n_cluster_excluded_no_hotspot_tf": int(len(test_sites) - len(usable)),
        "obs": obs_med,
        "null_mean": null_mean,
        "null_lo": float(null_lo),
        "null_hi": float(null_hi),
        "empirical_p": p_left,
        "tail": "left",
        "star": star,
        "n_null_jobs": int(len(jobs_use)),
        "null_definition": null_definition,
    }
    (out_dir / f"variant_{name}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--cluster-xlsx", type=Path, default=DEFAULT_CLUSTER_XLSX)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT / "figures5b" / "null_variants",
    )
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument(
        "--fasta",
        type=Path,
        default=next((p for p in FASTA_CANDIDATES if p.exists()), FASTA_CANDIDATES[0]),
    )
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    print("Loading sites…", flush=True)
    sites, _ = load_site_master(args.summary, args.cluster_xlsx)

    cluster_sites = sites.loc[
        sites["evidence"].astype(str).str.contains("Cluster", regex=False)
    ].copy()
    cluster_idx = set(cluster_sites["INDEX"].astype(str))
    seeds = sites.loc[
        (sites["is_known"] | sites["is_hc_predicted"])
        & (~sites["INDEX"].astype(str).isin(cluster_idx))
    ].copy()
    print(
        f"Building blind catalog (hold out {len(cluster_sites)} Cluster; "
        f"{len(seeds)} seeds)…",
        flush=True,
    )
    catalog, members = build_catalog_from_seeds(seeds, ADJ_MAX, SPAN_MAX)
    if "INDEX" not in members.columns:
        members = members.copy()
        members["INDEX"] = members["UniProt"].astype(str) + "_" + members["site"].astype(str)
    assert int(members["INDEX"].astype(str).isin(cluster_idx).sum()) == 0

    intervals_by_acc: Dict[str, List[Tuple[int, int]]] = {}
    for acc, g in catalog.groupby("UniProt"):
        intervals_by_acc[str(acc)] = list(zip(g["start"].astype(int), g["end"].astype(int)))

    catalog.to_csv(out / "shared_cluster_blind_hotspots.csv", index=False)
    members.to_csv(out / "shared_cluster_blind_members.csv", index=False)
    cluster_sites.to_csv(out / "shared_heldout_cluster_sites.csv", index=False)

    non_cluster = sites.loc[~sites["INDEX"].astype(str).isin(cluster_idx)].copy()
    seed_idx = set(seeds["INDEX"].astype(str))

    # Pools per variant
    pool0: Dict[str, pd.DataFrame] = {
        str(a): g for a, g in non_cluster.groupby("protein_acc")
    }

    n_cluster_by_acc = cluster_sites.groupby(cluster_sites["protein_acc"].astype(str)).size().to_dict()
    pool1: Dict[str, pd.DataFrame] = {}
    for acc, g in non_cluster.groupby("protein_acc"):
        acc = str(acc)
        intervals = intervals_by_acc.get(acc, [])
        need = int(n_cluster_by_acc.get(acc, 0))
        if not intervals:
            pool1[acc] = g
            continue
        pos = g["position"].to_numpy(dtype=int)
        outside = np.ones(len(g), dtype=bool)
        for a, b in intervals:
            outside &= ~((pos >= a) & (pos <= b))
        g_out = g.loc[outside].copy()
        # Align with 4B: fall back to full non-Cluster pool if outside pool too small
        pool1[acc] = g_out if len(g_out) >= max(need, 1) else g

    non_seed = non_cluster.loc[~non_cluster["INDEX"].astype(str).isin(seed_idx)].copy()
    pool2: Dict[str, pd.DataFrame] = {
        str(a): g for a, g in non_seed.groupby("protein_acc")
    }

    wanted_acc = set(cluster_sites["protein_acc"].astype(str))
    print(f"Loading FASTA STY for {len(wanted_acc)} Cluster TFs from {args.fasta}…", flush=True)
    seqs = load_fasta_subset(args.fasta, wanted_acc)
    pool3: Dict[str, pd.DataFrame] = {}
    for acc in wanted_acc:
        seq = seqs.get(acc)
        if not seq:
            continue
        pos, res = sty_positions_from_seq(seq)
        pool3[acc] = pd.DataFrame({"position": pos, "residue": res})

    hs_tfs = set(intervals_by_acc)
    test_v5 = cluster_sites.loc[cluster_sites["protein_acc"].astype(str).isin(hs_tfs)].copy()

    results: List[Dict] = []

    print("Variant 0 — baseline (non-Cluster phosphosite STY)…", flush=True)
    results.append(
        run_inside_variant(
            "0_baseline_noncluster_sty",
            cluster_sites,
            build_jobs(cluster_sites, pool0, intervals_by_acc),
            intervals_by_acc,
            args.n_perm,
            rng,
            out,
            "Within-TF STY from all non-Cluster phosphosites (current S5B null).",
        )
    )

    print("Variant 1 — outside-pool STY…", flush=True)
    results.append(
        run_inside_variant(
            "1_outside_pool_sty",
            cluster_sites,
            build_jobs(cluster_sites, pool1, intervals_by_acc),
            intervals_by_acc,
            args.n_perm,
            rng,
            out,
            "Within-TF STY from non-Cluster phosphosites outside blind hotspot intervals "
            "(fallback: if outside pool < n Cluster on that TF, job skipped).",
        )
    )

    print("Variant 2 — exclude Known∪HC seeds…", flush=True)
    results.append(
        run_inside_variant(
            "2_exclude_seeds_sty",
            cluster_sites,
            build_jobs(cluster_sites, pool2, intervals_by_acc),
            intervals_by_acc,
            args.n_perm,
            rng,
            out,
            "Within-TF STY from phosphosites that are neither Cluster nor Known∪HC seeds.",
        )
    )

    print("Variant 3 — sequence STY…", flush=True)
    results.append(
        run_inside_variant(
            "3_sequence_sty",
            cluster_sites,
            build_jobs(cluster_sites, pool3, intervals_by_acc),
            intervals_by_acc,
            args.n_perm,
            rng,
            out,
            "Within-TF STY from all S/T/Y residues on the UniProt sequence (not only observed phosphosites).",
        )
    )

    print("Variant 4 — median distance statistic…", flush=True)
    results.append(
        run_distance_variant(
            "4_distance_stat",
            cluster_sites,
            build_jobs(cluster_sites, pool0, intervals_by_acc),
            intervals_by_acc,
            args.n_perm,
            rng,
            out,
            "Statistic = median distance to nearest blind hotspot (left-tail). "
            "Null = within-TF STY from non-Cluster phosphosites. Sites on TFs without "
            "blind hotspots excluded from median.",
        )
    )

    print("Variant 5 — only TFs with blind hotspot…", flush=True)
    results.append(
        run_inside_variant(
            "5_tf_with_hotspot",
            test_v5,
            build_jobs(test_v5, pool0, intervals_by_acc),
            intervals_by_acc,
            args.n_perm,
            rng,
            out,
            "Same inside-fraction as baseline, but test set restricted to Cluster sites "
            "on TFs that have ≥1 blind hotspot; STY null from non-Cluster phosphosites.",
        )
    )

    cmp = pd.DataFrame(results)
    cmp.to_csv(out / "variants_comparison.csv", index=False)
    (out / "variants_comparison.json").write_text(json.dumps(results, indent=2, default=str))

    # Overview figure
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    labels = []
    for r in results:
        short = r["variant"].split("_", 1)[1] if "_" in r["variant"] else r["variant"]
        labels.append(short.replace("_", "\n"))
    x = np.arange(len(results))
    obs_vals = [r["obs"] for r in results]
    null_vals = [r["null_mean"] for r in results]
    # For distance variant, values are on different scale — plot separately note
    inside_idx = [i for i, r in enumerate(results) if r["statistic"] == "frac_inside"]
    dist_idx = [i for i, r in enumerate(results) if r["statistic"] == "median_dist"]
    ax.bar(x[inside_idx] - 0.15, [obs_vals[i] for i in inside_idx], width=0.3, color=C_OBS, label="Observed")
    ax.bar(x[inside_idx] + 0.15, [null_vals[i] for i in inside_idx], width=0.3, color=C_NULL, label="Null")
    for i in inside_idx:
        ax.text(i, max(obs_vals[i], null_vals[i]) + 0.01, results[i]["star"], ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Fraction inside (variants 0–3, 5)")
    ax.set_title("S5B null variants (distance variant shown in its own panel)")
    ax.legend(frameon=False, fontsize=8)
    if dist_idx:
        ax2 = ax.twinx()
        i = dist_idx[0]
        ax2.scatter([i - 0.15], [obs_vals[i]], color=C_OBS, s=40, zorder=5)
        ax2.scatter([i + 0.15], [null_vals[i]], color=C_NULL, s=40, zorder=5)
        ax2.set_ylabel("Median distance aa (variant 4)", color="#666666")
        ax2.text(i, max(obs_vals[i], null_vals[i]) * 1.05, results[i]["star"], ha="center", fontsize=9)
    fig.tight_layout()
    save_fig(fig, out / "variants_overview")

    meta = {
        "n_cluster_held_out": int(len(cluster_sites)),
        "n_seeds": int(len(seeds)),
        "n_blind_hotspots": int(len(catalog)),
        "n_blind_tfs": int(len(intervals_by_acc)),
        "n_fasta_loaded": int(len(seqs)),
        "n_perm": int(args.n_perm),
        "fasta": str(args.fasta),
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"meta": meta, "results": results}, indent=2, default=str), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
