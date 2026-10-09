#!/usr/bin/env python3
"""Hotspot × cancer heatmap of High−Low target-expression delta with BH-significance stars.

One panel per (Import|Export)×(activate|repress) folder under a regulon_only results root.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Repo-relative results root (cptac_analysis/results)
RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Keep text as editable strings in AI (not per-glyph path outlines).
# Re-apply after seaborn import in case style helpers touch fontsettings.
EDITABLE_VECTOR_FONT_RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "text.usetex": False,
    "pdf.use14corefonts": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "axes.unicode_minus": False,
}
matplotlib.rcParams.update(EDITABLE_VECTOR_FONT_RC)
sns.set_style("white")
matplotlib.rcParams.update(EDITABLE_VECTOR_FONT_RC)

CANCER_ORDER = [
    "BRCA",
    "CCRCC",
    "COAD",
    "GBM",
    "HNSCC",
    "LSCC",
    "LUAD",
    "OV",
    "PDAC",
    "UCEC",
]

ARMS = [
    ("Import_activate", "Import", "activate", "nuclear"),
    ("Import_repress", "Import", "repress", "nuclear"),
    ("Export_activate", "Export", "activate", "cytoplasmic"),
    ("Export_repress", "Export", "repress", "cytoplasmic"),
]


def _stars(q: float) -> str:
    if pd.isna(q) or q >= 0.05:
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    return "*"


def _delta_matches_hypothesis(direction: str, regulation: str, delta: float) -> bool:
    if pd.isna(delta):
        return False
    d = str(direction).lower()
    r = str(regulation).lower()
    if d == "import" and r == "activate":
        return float(delta) > 0
    if d == "import" and r == "repress":
        return float(delta) < 0
    if d == "export" and r == "activate":
        return float(delta) < 0
    if d == "export" and r == "repress":
        return float(delta) > 0
    return True


def _load_arm_table(arm_dir: Path, direction: str, regulation: str) -> Optional[pd.DataFrame]:
    """Prefer CPTAC by_site_all (full cancer×hotspot matrix); else figure4b comparison table."""
    loc = "nuclear" if direction == "Import" else "cytoplasmic"
    fig4b = arm_dir / "figure4" / f"figure4b_{loc}_hotspot_targets"
    candidates = [
        arm_dir / "cptac/high_low_phospho_boxplots/high_low_phospho_comparison_by_site_all.csv",
        fig4b / f"all_cancers_{direction}_{regulation}_sites_comparison_table.csv",
    ]
    df = None
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path, low_memory=False)
            break
    if df is None or df.empty:
        return None

    df = df[
        df["direction_short"].astype(str).eq(direction)
        & df["target_regulation"].astype(str).eq(regulation)
    ].copy()
    if df.empty:
        return None

    df["delta"] = pd.to_numeric(df["delta_high_minus_low"], errors="coerce")
    df["q"] = pd.to_numeric(df["wilcoxon_q_bh"], errors="coerce")
    if "site_label" not in df.columns or df["site_label"].isna().all():
        df["site_label"] = df.get("hotspot_label", df["site"])
    df["site_label"] = df["site_label"].astype(str)
    df["cancer_type"] = df["cancer_type"].astype(str)

    two_sided = False
    if "alternative" in df.columns:
        two_sided = df["alternative"].astype(str).str.lower().eq("two-sided").any()
    elif "expected_direction" in df.columns:
        two_sided = df["expected_direction"].astype(str).eq("High vs Low").any()
    elif "matches_hypothesis" in df.columns:
        two_sided = True

    if "matches_hypothesis" not in df.columns:
        df["matches_hypothesis"] = [
            _delta_matches_hypothesis(direction, regulation, d) for d in df["delta"]
        ]
    elif df["matches_hypothesis"].dtype != bool:
        df["matches_hypothesis"] = df["matches_hypothesis"].astype(str).str.lower().isin(["true", "1"])

    # Stars match final boxplot rule
    bh = df["q"].lt(0.05)
    df["is_plot_sig"] = (bh & df["matches_hypothesis"]) if two_sided else bh
    df["star"] = [_stars(q) if sig else "" for q, sig in zip(df["q"], df["is_plot_sig"])]
    df.attrs["two_sided"] = two_sided
    return df


def _order_hotspots(df: pd.DataFrame) -> list:
    """Order hotspots by n significant cancers (desc), then |median delta|."""
    g = (
        df.groupby("site_label", sort=False)
        .agg(
            n_sig=("is_plot_sig", "sum"),
            med_abs_delta=("delta", lambda s: float(np.nanmedian(np.abs(s)))),
        )
        .reset_index()
        .sort_values(["n_sig", "med_abs_delta"], ascending=[False, False])
    )
    return g["site_label"].tolist()


def plot_heatmap(
    df: pd.DataFrame,
    out_stem: Path,
    *,
    title: str,
    dpi: int = 300,
) -> Tuple[Path, Path]:
    cancers = [c for c in CANCER_ORDER if c in set(df["cancer_type"])]
    extra = sorted(set(df["cancer_type"]) - set(cancers))
    cancers = cancers + extra
    hotspots = _order_hotspots(df)

    mat = (
        df.pivot_table(index="site_label", columns="cancer_type", values="delta", aggfunc="first")
        .reindex(index=hotspots, columns=cancers)
    )
    star_mat = (
        df.pivot_table(index="site_label", columns="cancer_type", values="star", aggfunc="first")
        .reindex(index=hotspots, columns=cancers)
        .fillna("")
    )
    annot = star_mat.astype(object)
    for i in mat.index:
        for j in mat.columns:
            if pd.isna(mat.loc[i, j]):
                annot.loc[i, j] = ""

    n_row, n_col = mat.shape
    fig_w = max(6.5, 0.55 * n_col + 3.2)
    fig_h = max(4.0, 0.32 * n_row + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmax = np.nanpercentile(np.abs(mat.to_numpy(dtype=float)), 95) if mat.notna().any().any() else 0.1
    vmax = float(max(vmax, 0.05))
    sns.heatmap(
        mat,
        ax=ax,
        cmap="RdBu_r",
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        annot=annot,
        fmt="",
        annot_kws={"size": 9, "weight": "bold", "color": "black"},
        linewidths=0.4,
        linecolor="#E8E8E8",
        cbar_kws={"label": "Δ expression (High − Low phospho)", "shrink": 0.8},
        mask=mat.isna(),
    )
    ax.set_xlabel("Cancer")
    ax.set_ylabel("Hotspot")
    ax.set_title(title, fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf", "svg"):
        p = Path(f"{out_stem}.{ext}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
        paths.append(p)
    plt.close(fig)

    # Also write matrix CSV
    mat_out = Path(f"{out_stem}_delta_matrix.csv")
    mat.to_csv(mat_out)
    star_out = Path(f"{out_stem}_significance_stars.csv")
    star_mat.to_csv(star_out)
    return paths[0], mat_out


def process_root(root: Path, dpi: int = 300) -> None:
    print(f"=== {root.name} ===")
    for arm, direction, regulation, loc in ARMS:
        arm_dir = root / arm
        if not arm_dir.exists():
            print(f"  skip missing {arm}")
            continue
        df = _load_arm_table(arm_dir, direction, regulation)
        if df is None or df.empty:
            print(f"  skip empty table {arm}")
            continue
        out_dir = arm_dir / "figure4" / f"figure4b_{loc}_hotspot_targets"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = out_dir / f"All_cancers_{direction}_{regulation}_hotspot_cancer_delta_heatmap"
        title = (
            f"{direction} × {regulation}: Hotspot × cancer Δ(High−Low)\n"
            f"(* BH q<0.05"
            + (", hypothesis-concordant)" if df.attrs.get("two_sided") else ")")
        )
        png, csv = plot_heatmap(df, stem, title=title, dpi=dpi)
        n_sig = int(df["is_plot_sig"].sum())
        print(f"  {arm}: {df['site_label'].nunique()} hotspots × {df['cancer_type'].nunique()} cancers; "
              f"plot-sig cells={n_sig} → {png.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        nargs="+",
        type=Path,
        default=[
            RESULTS_ROOT / "hotspot_mixed_pred_filter_pure_regulon_only_20260928",
            RESULTS_ROOT / "hotspot_mixed_pred_filter_pure_regulon_only_twosided_20260929",
        ],
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    for root in args.roots:
        process_root(root, dpi=args.dpi)


if __name__ == "__main__":
    main()
