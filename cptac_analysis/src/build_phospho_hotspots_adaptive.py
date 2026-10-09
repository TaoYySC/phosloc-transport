#!/usr/bin/env python3
"""Estimate an adaptive single-linkage gap cutoff from Known/HC adjacent gaps,
then build hotspot catalogs (same membership rules as build_phospho_hotspots.py).

Workflow:
  1) Pool adjacent gaps of Known ∪ HC anchors across all TF proteins
  2) Estimate candidate cutoffs: knee (gaps≤200), 2-component GMM, percentiles
  3) Pick a primary cutoff (prefer GMM if in [5, 50], else knee, else median)
  4) Single-linkage cluster with that cutoff (singletons kept)
  5) Write catalogs + gap diagnostics + comparison vs fixed d=10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_phospho_hotspots import (  # noqa: E402
    DEFAULT_SUMMARY,
    annotate_site_flags,
    build_hotspots_for_distance,
    collect_cptac_observed_indices,
    load_all_tf_sites,
    load_region_flags,
)

DEFAULT_OUT = (
    REPO
    / "cptac_analysis/results/import_target_regulation_hotspot_v11_147pos_d3_platt"
    / "hotspots_adaptive"
)
DEFAULT_FIXED_DIR = (
    REPO
    / "cptac_analysis/results/import_target_regulation_hotspot_v11_147pos_d3_platt"
    / "hotspots"
)
DEFAULT_LINKEDOMICS = REPO / "cptac_analysis/data/source/1.cpatac/LinkedOmicsKB"
DEFAULT_IDMAPPING = REPO / "cptac_analysis/data/source/3.idmapping/HUMAN_9606_idmapping.dat"
DEFAULT_CANCERS = [
    "BRCA", "CCRCC", "COAD", "GBM", "HNSCC",
    "LSCC", "LUAD", "OV", "PDAC", "UCEC",
]


def adjacent_gaps(positions_by_protein: Dict[str, Sequence[int]]) -> np.ndarray:
    gaps: List[int] = []
    for pos in positions_by_protein.values():
        ps = sorted(set(int(p) for p in pos))
        if len(ps) < 2:
            continue
        for a, b in zip(ps[:-1], ps[1:]):
            gaps.append(int(b - a))
    return np.asarray(gaps, dtype=float)


def knee_cutoff(gaps: np.ndarray, max_gap: float = 200.0) -> Tuple[float, float]:
    """Max distance-to-chord knee on sorted gaps capped at max_gap.

    Returns (knee_value, percentile_within_capped).
    """
    sg = np.sort(gaps[np.isfinite(gaps) & (gaps <= max_gap)])
    if len(sg) < 5:
        sg = np.sort(gaps[np.isfinite(gaps)])
    if len(sg) < 3:
        return float(np.median(gaps)), 50.0
    x = np.arange(len(sg)) / (len(sg) - 1)
    y = sg / float(sg.max())
    line = y[0] + (y[-1] - y[0]) * x
    idx = int(np.argmax(np.abs(y - line)))
    return float(sg[idx]), 100.0 * idx / (len(sg) - 1)


def gmm_cutoff(gaps: np.ndarray, max_gap: float = 500.0, seed: int = 42) -> Optional[float]:
    """2-component GMM on log1p(gaps); return intersection of components if found."""
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        return None
    x = gaps[np.isfinite(gaps) & (gaps > 0) & (gaps <= max_gap)]
    if len(x) < 30:
        return None
    z = np.log1p(x).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=seed, covariance_type="full")
    gmm.fit(z)
    means = gmm.means_.ravel()
    vars_ = gmm.covariances_.ravel()
    weights = gmm.weights_.ravel()
    order = np.argsort(means)
    m1, m2 = means[order]
    v1, v2 = vars_[order]
    w1, w2 = weights[order]
    # Solve w1*N(m1,v1)=w2*N(m2,v2) on the log1p scale between means
    grid = np.linspace(m1, m2, 2000)
    dens1 = w1 * np.exp(-0.5 * (grid - m1) ** 2 / v1) / np.sqrt(2 * np.pi * v1)
    dens2 = w2 * np.exp(-0.5 * (grid - m2) ** 2 / v2) / np.sqrt(2 * np.pi * v2)
    diff = dens1 - dens2
    # first sign change from positive (local) to negative (background)
    crosses = np.where(np.diff(np.sign(diff)))[0]
    if len(crosses) == 0:
        # fallback: equal-density nearest mid
        mid = float(np.expm1(0.5 * (m1 + m2)))
        return mid
    i = int(crosses[0])
    # linear interpolate zero crossing
    t = diff[i] / (diff[i] - diff[i + 1] + 1e-12)
    z_star = float(grid[i] + t * (grid[i + 1] - grid[i]))
    return float(np.expm1(z_star))


def choose_primary(
    knee: float,
    gmm: Optional[float],
    median: float,
    lo: float = 5.0,
    hi: float = 50.0,
) -> Tuple[int, str]:
    """Pick integer primary cutoff with provenance string."""
    if gmm is not None and lo <= gmm <= hi:
        return int(round(gmm)), "gmm_intersection"
    if lo <= knee <= hi:
        return int(round(knee)), "knee_gaps_le_200"
    # clamp median into a usable window
    d = int(round(np.clip(median, lo, hi)))
    return d, "median_clipped"


def size_bins(n: pd.Series) -> pd.Series:
    out = n.astype(int).astype(str)
    out = out.where(n.lt(4), "≥4")
    return out


def catalog_summary(hotspots: pd.DataFrame) -> Dict:
    if hotspots.empty:
        return {"n_hotspots": 0}
    bins = size_bins(hotspots["n_members"]).value_counts().to_dict()
    return {
        "n_hotspots": int(len(hotspots)),
        "n_genes": int(hotspots["gene_name"].nunique()),
        "n_proteins": int(hotspots["protein_acc"].nunique()),
        "n_members_total": int(hotspots["n_members"].sum()),
        "size_bins": {str(k): int(v) for k, v in sorted(bins.items(), key=lambda kv: (len(str(kv[0])), str(kv[0])))},
        "median_n_members": float(hotspots["n_members"].median()),
        "median_span_aa": float(hotspots["span_aa"].median()),
    }


def jaccard_member_sets(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Jaccard over frozensets of member INDEX strings per hotspot, matched loosely
    by unordered member set equality rate among catalogs (set-of-sets Jaccard).
    """
    def sets(df: pd.DataFrame):
        s = set()
        for _, row in df.iterrows():
            members = tuple(sorted(str(x) for x in str(row["member_indices"]).split(";") if x))
            if members:
                s.add(members)
        return s

    A, B = sets(a), sets(b)
    if not A and not B:
        return 1.0
    return float(len(A & B) / len(A | B)) if (A | B) else np.nan


def write_gap_diagnostics(
    gaps: np.ndarray,
    out_dir: Path,
    candidates: Dict[str, float],
    primary: int,
) -> None:
    gap_df = pd.DataFrame({"adjacent_gap_aa": gaps})
    gap_df.to_csv(out_dir / "adjacent_gaps.csv", index=False)

    rows = []
    for p in [10, 25, 50, 60, 70, 75, 80, 85, 90, 95, 99]:
        rows.append({"stat": f"P{p}", "value": float(np.percentile(gaps, p))})
    rows.append({"stat": "mean", "value": float(gaps.mean())})
    rows.append({"stat": "median", "value": float(np.median(gaps))})
    rows.append({"stat": "n_gaps", "value": float(len(gaps))})
    for name, val in candidates.items():
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            continue
        rows.append({"stat": name, "value": float(val)})
    rows.append({"stat": "primary_cutoff", "value": float(primary)})
    pd.DataFrame(rows).to_csv(out_dir / "gap_cutoff_candidates.csv", index=False)

    # fraction of adjacent gaps merged at common cutoffs
    frac_rows = []
    cutoff_vals = [
        float(v)
        for k, v in candidates.items()
        if k != "knee_percentile_in_capped" and v is not None and np.isfinite(v)
    ]
    for c in sorted(set([5, 8, 10, 12, 15, 20, 25, 30, primary] + [int(round(v)) for v in cutoff_vals])):
        frac_rows.append(
            {
                "cutoff": int(c),
                "frac_gaps_le": float((gaps <= c).mean()),
                "n_gaps_le": int((gaps <= c).sum()),
            }
        )
    pd.DataFrame(frac_rows).to_csv(out_dir / "gap_cutoff_coverage.csv", index=False)

    try:
        plot_adjacent_gap_histogram(gaps, out_dir, candidates, primary)
    except Exception as exc:  # pragma: no cover
        print(f"Plot skipped: {exc}")


def plot_adjacent_gap_histogram(
    gaps: np.ndarray,
    out_dir: Path,
    candidates: Dict[str, float],
    primary: int,
    xmax: float = 60.0,
) -> None:
    """Simple gap histogram with a single primary-cutoff marker."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.8,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    caps = gaps[np.isfinite(gaps) & (gaps <= xmax)]
    _ = candidates  # kept for API compatibility

    fig, ax = plt.subplots(figsize=(6.2, 3.5), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bins = np.arange(0, xmax + 2, 2)
    ax.hist(
        caps,
        bins=bins,
        color="#8FA6C0",
        edgecolor="white",
        linewidth=0.55,
        zorder=1,
    )

    ax.axvline(primary, color="#C2410C", ls="-", lw=2.0, zorder=3)
    ymax = ax.get_ylim()[1]
    ax.text(
        primary + 1.0,
        ymax * 0.95,
        f"{primary} aa",
        color="#C2410C",
        fontsize=10,
        va="top",
        ha="left",
        fontweight="medium",
    )

    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#EEF0F3", lw=0.7, zorder=0)
    ax.set_axisbelow(True)

    ax.set_xlabel("Adjacent gap (aa)")
    ax.set_ylabel("Count")
    ax.set_title("Known/HC adjacent gap distribution", pad=8)

    fig.tight_layout()
    fig.savefig(out_dir / "adjacent_gap_histogram.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "adjacent_gap_histogram.pdf", bbox_inches="tight")
    plt.close(fig)


def prepare_sites(args: argparse.Namespace) -> pd.DataFrame:
    sites = load_all_tf_sites(args.summary)
    if args.skip_cptac_observed:
        sites["cptac_observed"] = False
    else:
        print("Scanning CPTAC phospho matrices for observed sites...")
        observed = collect_cptac_observed_indices(
            sites,
            args.linkedomics_base,
            args.idmapping_path,
            args.cancer_types,
        )
        sites["cptac_observed"] = sites["INDEX"].astype(str).isin(observed)
    sites = annotate_site_flags(sites)
    region = load_region_flags(sites["INDEX"].astype(str).tolist())
    sites = sites.merge(region, on="INDEX", how="left")
    for col in ["in_DBD", "in_NLS", "in_NES", "in_1433", "in_IDR"]:
        if col in sites.columns:
            sites[col] = sites[col].fillna(0).astype(int)
    if "Region" not in sites.columns:
        sites["Region"] = ""
    return sites


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fixed-hotspot-dir", type=Path, default=DEFAULT_FIXED_DIR)
    parser.add_argument("--linkedomics-base", type=Path, default=DEFAULT_LINKEDOMICS)
    parser.add_argument("--idmapping-path", type=Path, default=DEFAULT_IDMAPPING)
    parser.add_argument("--cancer-types", nargs="+", default=DEFAULT_CANCERS)
    parser.add_argument(
        "--skip-cptac-observed",
        action="store_true",
        help="Skip CPTAC matrix scan (faster; n_cptac_observed will be 0)",
    )
    parser.add_argument(
        "--force-cutoff",
        type=int,
        default=None,
        help="Override primary cutoff with this integer (still writes diagnostics)",
    )
    parser.add_argument(
        "--extra-distances",
        type=int,
        nargs="*",
        default=[10, 15, 30],
        help="Also write catalogs at these fixed distances for comparison",
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sites = prepare_sites(args)
    anchors = sites.loc[sites["is_keep_anchor"].astype(bool)].copy()
    n_anchors = len(anchors)
    pos_by_acc = (
        anchors.groupby("protein_acc")["position"]
        .apply(lambda s: sorted(set(int(x) for x in s)))
        .to_dict()
    )
    gaps = adjacent_gaps(pos_by_acc)
    print(f"Anchors Known|HC={n_anchors}; adjacent gaps={len(gaps)}")

    knee, knee_pct = knee_cutoff(gaps, max_gap=200.0)
    gmm = gmm_cutoff(gaps)
    median = float(np.median(gaps))
    candidates = {
        "knee_gaps_le_200": knee,
        "knee_percentile_in_capped": knee_pct,
        "gmm_intersection": gmm if gmm is not None else np.nan,
        "median": median,
        "P60": float(np.percentile(gaps, 60)),
        "P70": float(np.percentile(gaps, 70)),
        "P75": float(np.percentile(gaps, 75)),
    }
    if args.force_cutoff is not None:
        primary = int(args.force_cutoff)
        primary_method = "forced"
    else:
        primary, primary_method = choose_primary(knee, gmm, median)

    print(
        f"Candidates: knee={knee:.1f} (P≈{knee_pct:.1f} within ≤200), "
        f"GMM={gmm}, median={median:.1f} → primary={primary} ({primary_method})"
    )
    write_gap_diagnostics(gaps, out_dir, candidates, primary)

    # Distances to build: primary + sensitivity percentiles + extras
    sens = sorted(
        {
            primary,
            int(round(candidates["P60"])),
            int(round(candidates["P70"])),
            int(round(candidates["P75"])),
            int(round(knee)),
            *([int(round(gmm))] if gmm is not None and np.isfinite(gmm) else []),
            *[int(d) for d in (args.extra_distances or [])],
        }
    )
    sens = [d for d in sens if d >= 1]

    summaries = {}
    for d in sens:
        hotspots, members = build_hotspots_for_distance(sites, d)
        hotspots.to_csv(out_dir / f"hotspots_d{d}.csv", index=False)
        members.to_csv(out_dir / f"hotspot_members_d{d}.csv", index=False)
        summaries[str(d)] = catalog_summary(hotspots)
        print(
            f"  d={d}: hotspots={summaries[str(d)]['n_hotspots']} "
            f"genes={summaries[str(d)]['n_genes']} bins={summaries[str(d)].get('size_bins')}"
        )

    # Compare primary vs fixed d=10 (prefer freshly built; else fixed dir)
    compare_rows = []
    primary_hs = pd.read_csv(out_dir / f"hotspots_d{primary}.csv")
    for d_ref in [10, 15, 30]:
        path_local = out_dir / f"hotspots_d{d_ref}.csv"
        path_fixed = args.fixed_hotspot_dir / f"hotspots_d{d_ref}.csv"
        ref_path = path_local if path_local.exists() else path_fixed
        if not ref_path.exists():
            continue
        ref = pd.read_csv(ref_path)
        compare_rows.append(
            {
                "adaptive_cutoff": primary,
                "reference_cutoff": d_ref,
                "n_hotspots_adaptive": len(primary_hs),
                "n_hotspots_reference": len(ref),
                "jaccard_member_sets": jaccard_member_sets(primary_hs, ref),
                "n_genes_adaptive": int(primary_hs["gene_name"].nunique()),
                "n_genes_reference": int(ref["gene_name"].nunique()),
            }
        )
    if compare_rows:
        pd.DataFrame(compare_rows).to_csv(out_dir / "adaptive_vs_fixed_comparison.csv", index=False)

    config = {
        "method": "adaptive_gap_single_linkage_known_hc_only",
        "summary": str(args.summary),
        "n_anchors": n_anchors,
        "n_adjacent_gaps": int(len(gaps)),
        "gap_candidates": {k: (None if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in candidates.items()},
        "primary_cutoff": primary,
        "primary_method": primary_method,
        "distances_written": sens,
        "catalog_summaries": summaries,
        "definition": (
            "Cluster Known/HC anchors with single-linkage gap<=adaptive_cutoff; "
            "singletons allowed; members=Known/HC only"
        ),
        "skip_cptac_observed": bool(args.skip_cptac_observed),
    }
    (out_dir / "adaptive_cutoff_config.json").write_text(json.dumps(config, indent=2))
    # Convenience pointer for downstream scripts
    (out_dir / "PRIMARY_CUTOFF.txt").write_text(f"{primary}\n{primary_method}\n")
    print(f"Wrote adaptive catalogs to {out_dir}")
    print(f"PRIMARY_CUTOFF={primary} ({primary_method})")


if __name__ == "__main__":
    main()
