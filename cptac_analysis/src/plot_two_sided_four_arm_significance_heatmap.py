#!/usr/bin/env python3
"""Two-sided-only hotspot × cancer significance heatmap (4 arms).

Dots: BH q<0.05, any direction. Only hotspots with ≥1 significant cancer.
Panel names: Nuclear accumulation / Cytoplasmic redistribution × activated/repressed.
"""

from __future__ import annotations

from pathlib import Path

# Repo-relative results root (cptac_analysis/results)
RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

# Keep text as editable strings in AI (not per-glyph path outlines).
matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "text.usetex": False,
        "pdf.use14corefonts": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "axes.unicode_minus": False,
    }
)

TWO_ROOT = RESULTS_ROOT / "hotspot_mixed_pred_filter_pure_regulon_only_twosided_20260929"
OUT_DIR = TWO_ROOT / "combined_four_arm_heatmap"

# Figure size requested for main text panel (mm → inches).
FIG_W_IN = 176.0 / 25.4
FIG_H_IN = 87.0 / 25.4
PT_AXIS = 7.0   # cancer / hotspot labels + colorbar tick numbers
PT_OTHER = 8.0  # title, section labels, axis name, legends, significance stars

# Evidence markers (match provided legend swatches)
COLOR_MIXED = "#C07952"  # Mixed-evidence hotspot (known_containing)
COLOR_PRED = "#5D8896"   # Predicted candidate hotspot (known_independent)
LABEL_MIXED = "Mixed-evidence hotspot"
LABEL_PRED = "Predicted candidate hotspot"

CANCER_ORDER = [
    "BRCA", "CCRCC", "COAD", "GBM", "HNSCC", "LSCC", "LUAD", "OV", "PDAC", "UCEC"
]
ARMS = [
    ("Import_activate", "activate", "Nuclear accumulation\n(activated)"),
    ("Import_repress", "repress", "Nuclear accumulation\n(repressed)"),
    ("Export_activate", "activate", "Cytoplasmic redistribution\n(activated)"),
    ("Export_repress", "repress", "Cytoplasmic redistribution\n(repressed)"),
]

# Midpoint (Δ≈0), missing cells, and section gaps: lightest legend grey
CENTER_GREY = "#F0F0F0"
MISSING_GREY = "#F0F0F0"
GAP_GREY = "#F0F0F0"


def _diverging_grey_center_cmap(n: int = 256, grey_half_width: float = 0.04):
    """RdBu_r with only a narrow grey band around zero; ends stay vivid."""
    from matplotlib.colors import ListedColormap

    base = plt.get_cmap("RdBu_r")
    rgba = base(np.linspace(0.0, 1.0, n))
    grey = np.array(matplotlib.colors.to_rgba(CENTER_GREY))
    center = 0.5
    for i in range(n):
        t = i / (n - 1)
        d = abs(t - center) / grey_half_width
        if d < 1.0:
            # smooth cosine blend only near zero
            w = 0.5 * (1.0 + np.cos(np.pi * d))
            rgba[i] = (1.0 - w) * rgba[i] + w * grey
    return ListedColormap(rgba, name="RdBu_grey_center")


def _kp_class(hotspot_class: object) -> str:
    cls = str(hotspot_class).strip() if pd.notna(hotspot_class) else ""
    return "K" if cls == "known_containing" else "P"


def _load_arm(arm: str, regulation: str) -> pd.DataFrame:
    path = (
        TWO_ROOT
        / arm
        / "cptac_filtered"
        / f"two_side_high_low_phospho_comparison_by_site_{arm.lower()}.csv"
    )
    df = pd.read_csv(path, low_memory=False)
    df["q"] = pd.to_numeric(df["wilcoxon_q_bh"], errors="coerce")
    df["delta"] = pd.to_numeric(df["delta_high_minus_low"], errors="coerce")
    df["site_label"] = df["site_label"].astype(str)
    df["cancer_type"] = df["cancer_type"].astype(str)
    if "hotspot_class" not in df.columns:
        df["hotspot_class"] = ""
    df["kp"] = df["hotspot_class"].map(_kp_class)
    if "evaluable" in df.columns:
        df["evaluable"] = df["evaluable"].astype(bool)
    else:
        df["evaluable"] = df["q"].notna()
    # Stars / BH significance only for n>=10 evaluable tests with q<0.05.
    df["bh_sig"] = df["evaluable"] & df["q"].notna() & df["q"].lt(0.05)
    return df


def build_tables() -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    records = []
    arm_frames: Dict[str, pd.DataFrame] = {}
    for arm, reg, title in ARMS:
        df = _load_arm(arm, reg).copy()
        sig_sites = set(df.loc[df["bh_sig"], "site_label"])
        plot = df[df["site_label"].isin(sig_sites)].copy()
        plot["sig"] = plot["bh_sig"]
        plot["arm"] = arm
        plot["arm_title"] = title
        arm_frames[arm] = plot
        for _, r in plot.iterrows():
            records.append(
                {
                    "arm": arm,
                    "arm_title": title.replace("\n", " "),
                    "cancer_type": r["cancer_type"],
                    "site": r.get("site", ""),
                    "site_label": r["site_label"],
                    "hotspot_class": r["hotspot_class"],
                    "kp": r["kp"],
                    "delta": r["delta"],
                    "wilcoxon_q_bh": r["q"],
                    "bh_significant": bool(r["bh_sig"]),
                }
            )
    return pd.DataFrame(records), arm_frames


def _order_sites(df: pd.DataFrame) -> List[str]:
    g = (
        df.groupby(["site_label", "kp"], sort=False)
        .agg(
            n_sig=("sig", "sum"),
            med_abs=("delta", lambda s: float(np.nanmedian(np.abs(s)))),
        )
        .reset_index()
    )
    g["kp_rank"] = g["kp"].map({"K": 0, "P": 1}).fillna(2)
    g = g.sort_values(
        ["kp_rank", "n_sig", "med_abs", "site_label"],
        ascending=[True, False, False, True],
    )
    return g["site_label"].tolist()


def _stars_from_q(q: float) -> str:
    """BH q → asterisk tier: *** <0.001, ** <0.01, * <0.05."""
    if q is None or not np.isfinite(q) or q >= 0.05:
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    return "*"


def plot_heatmap(arm_frames: Dict[str, pd.DataFrame], out_stem: Path, dpi: int = 300) -> None:
    panels = []
    for arm, _, title in ARMS:
        df = arm_frames.get(arm, pd.DataFrame())
        if df is None or df.empty or not df["sig"].any():
            panels.append((title, None, None, None, []))
            continue
        sites = _order_sites(df)
        cancers = [c for c in CANCER_ORDER if c in set(df["cancer_type"])]
        cancers += sorted(set(df["cancer_type"]) - set(cancers))
        delta = df.pivot_table(index="cancer_type", columns="site_label", values="delta", aggfunc="first")
        qmat = df.pivot_table(index="cancer_type", columns="site_label", values="q", aggfunc="first")
        delta = delta.reindex(index=cancers, columns=sites)
        qmat = qmat.reindex(index=cancers, columns=sites)
        kp = df.drop_duplicates("site_label").set_index("site_label")["kp"].reindex(sites).fillna("P")
        panels.append((title, delta, qmat, kp, sites))

    # Shared cancer axis across arms
    cancers = list(CANCER_ORDER)
    for _, delta, _, _, _ in panels:
        if delta is not None:
            for c in delta.index:
                if c not in cancers:
                    cancers.append(c)

    spacer_n = 1  # one empty column (white stripe) between arms
    col_labels: List[str] = []
    col_kp: List[str] = []
    col_is_spacer: List[bool] = []
    section_spans: List[Tuple[str, int, int]] = []  # title, start, end (inclusive, data cols)
    delta_blocks = []
    q_blocks = []

    for i, (title, delta, qmat, kp, sites) in enumerate(panels):
        if i > 0:
            for _ in range(spacer_n):
                col_labels.append("")
                col_kp.append("")
                col_is_spacer.append(True)
                delta_blocks.append(pd.DataFrame(np.nan, index=cancers, columns=[f"__gap_{i}"]))
                q_blocks.append(pd.DataFrame(np.nan, index=cancers, columns=[f"__gap_{i}"]))

        start = len(col_labels)
        if delta is None or len(sites) == 0:
            col_labels.append("")
            col_kp.append("")
            col_is_spacer.append(True)
            delta_blocks.append(pd.DataFrame(np.nan, index=cancers, columns=[f"__empty_{i}"]))
            q_blocks.append(pd.DataFrame(np.nan, index=cancers, columns=[f"__empty_{i}"]))
            section_spans.append((title, start, start))
            continue

        d = delta.reindex(index=cancers)
        q = qmat.reindex(index=cancers)
        for site in sites:
            col_labels.append(site)
            col_kp.append(str(kp.loc[site]))
            col_is_spacer.append(False)
        delta_blocks.append(d)
        q_blocks.append(q)
        section_spans.append((title, start, start + len(sites) - 1))

    delta_all = pd.concat(delta_blocks, axis=1)
    q_all = pd.concat(q_blocks, axis=1)
    delta_all.columns = range(delta_all.shape[1])
    q_all.columns = range(q_all.shape[1])

    n_cols = delta_all.shape[1]
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
    # Leave room for rotated hotspot labels + colored markers + section titles + bottom legend.
    ax = fig.add_axes([0.055, 0.28, 0.78, 0.52])

    flat = delta_all.to_numpy(dtype=float).ravel()
    flat = flat[np.isfinite(flat)]
    vmax = float(np.nanpercentile(np.abs(flat), 95)) if len(flat) else 0.2
    vmax = max(vmax, 0.05)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = _diverging_grey_center_cmap()
    cmap.set_bad(MISSING_GREY)

    mat = np.ma.masked_invalid(delta_all.to_numpy(dtype=float))
    ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest", zorder=1)

    # Gap columns: grey gutter with a thin white center stripe
    n_rows = len(cancers)
    for j, is_sp in enumerate(col_is_spacer):
        if not is_sp:
            continue
        ax.add_patch(
            mpatches.Rectangle(
                (j - 0.5, -0.5),
                1.0,
                n_rows,
                facecolor=GAP_GREY,
                edgecolor="none",
                zorder=1.5,
                clip_on=True,
            )
        )
        stripe_w = 0.22
        ax.add_patch(
            mpatches.Rectangle(
                (j - stripe_w / 2, -0.5),
                stripe_w,
                n_rows,
                facecolor="white",
                edgecolor="none",
                zorder=1.6,
                clip_on=True,
            )
        )

    # Evidence markers above columns (colored filled circles)
    radius = 0.12
    marker_gap = 0.16
    for j, (lab, kp_j, is_sp) in enumerate(zip(col_labels, col_kp, col_is_spacer)):
        if is_sp or not lab:
            continue
        face = COLOR_MIXED if kp_j == "K" else COLOR_PRED
        ax.add_patch(
            mpatches.Circle(
                (j, -0.5 - marker_gap - radius),
                radius,
                facecolor=face,
                edgecolor=face,
                linewidth=0.4,
                clip_on=False,
                zorder=4,
            )
        )

    # Section titles centered over each arm
    for title, a, b in section_spans:
        mid = (a + b) / 2.0
        ax.text(
            mid,
            -0.5 - marker_gap - radius - 0.42,
            title,
            ha="center",
            va="bottom",
            fontsize=PT_OTHER,
            linespacing=1.05,
            clip_on=False,
            transform=ax.transData,
            zorder=5,
        )

    xticks = [j for j, (lab, is_sp) in enumerate(zip(col_labels, col_is_spacer)) if lab and not is_sp]
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        [col_labels[j] for j in xticks],
        rotation=55,
        ha="right",
        fontsize=PT_AXIS,
        rotation_mode="anchor",
    )
    ax.set_yticks(range(len(cancers)))
    ax.set_yticklabels(cancers, fontsize=PT_AXIS)
    ax.set_ylabel("Cancer", fontsize=PT_OTHER, labelpad=2)
    ax.tick_params(axis="both", length=0, pad=2)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(len(cancers) - 0.5, -0.5)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    # Stars on a dedicated transparent overlay so each is a top-layer editable text object in AI.
    ax_stars = fig.add_axes(ax.get_position(), sharex=ax, sharey=ax, frameon=False, zorder=20)
    ax_stars.set_xlim(ax.get_xlim())
    ax_stars.set_ylim(ax.get_ylim())
    ax_stars.set_axis_off()
    ax_stars.patch.set_alpha(0.0)
    q_arr = q_all.to_numpy(dtype=float)
    for yi in range(q_arr.shape[0]):
        for xi in range(q_arr.shape[1]):
            if col_is_spacer[xi]:
                continue
            stars = _stars_from_q(q_arr[yi, xi])
            if not stars:
                continue
            # One Text object per significant cell (whole "***" string, not per-glyph paths).
            ax_stars.text(
                xi,
                yi,
                stars,
                ha="center",
                va="center",
                fontsize=PT_OTHER,
                color="black",
                fontweight="normal",
                clip_on=False,
                zorder=30,
                fontfamily="sans-serif",
            )

    cax = fig.add_axes([0.875, 0.30, 0.012, 0.42])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Δ expression (High − Low)", fontsize=PT_OTHER)
    cb.ax.tick_params(labelsize=PT_AXIS, length=2, width=0.6)
    cb.outline.set_linewidth(0.6)

    fig.suptitle(
        "Two-sided hotspot–target expression changes",
        fontsize=PT_OTHER,
        y=0.98,
    )

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLOR_MIXED,
            markeredgecolor=COLOR_MIXED,
            markersize=6.5,
            label=LABEL_MIXED,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLOR_PRED,
            markeredgecolor=COLOR_PRED,
            markersize=6.5,
            label=LABEL_PRED,
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=PT_OTHER,
        handletextpad=0.4,
        columnspacing=1.6,
        bbox_to_anchor=(0.45, 0.01),
    )

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            f"{out_stem}.{ext}",
            dpi=dpi,
            facecolor="white",
            # Keep absolute size; avoid bbox_inches='tight' shrinking away from 176×87 mm.
            bbox_inches=None,
            pad_inches=0,
        )
    plt.close(fig)


def main() -> None:
    long_df, arm_frames = build_tables()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(OUT_DIR / "two_sided_four_arm_significance_long.csv", index=False)

    summary = []
    for arm, _, title in ARMS:
        df = arm_frames[arm]
        summary.append(
            {
                "arm": arm,
                "title": title.replace("\n", " "),
                "n_sig_sites": int(df.loc[df["sig"], "site_label"].nunique()) if len(df) else 0,
                "n_sig_cells": int(df["sig"].sum()) if len(df) else 0,
            }
        )
    pd.DataFrame(summary).to_csv(OUT_DIR / "two_sided_four_arm_summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))

    stem = OUT_DIR / "two_sided_four_arm_hotspot_cancer_heatmap"
    plot_heatmap(arm_frames, stem)
    print(f"Wrote {stem}.png")


if __name__ == "__main__":
    main()
