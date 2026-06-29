#!/usr/bin/env python3
"""
Draw Import activate low/high phospho target-expression boxplots for one or more
phosphosites across cancers (x-axis = cancer types with available data).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_CPTAC_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

_PIPELINE_RUN = _CPTAC_ROOT / "results" / "import_target_regulation"
_BOXPLOT_DIR = _PIPELINE_RUN / "high_low_phospho_boxplots"

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd
import seaborn as sns

from run_import_target_regulation_analysis import (
  DEFAULT_CANCER_LIST,
  EDITABLE_VECTOR_FONT_RC,
  TargetRegulationBoxplotPipeline,
  TempoConfig,
)

matplotlib.rcParams.update(EDITABLE_VECTOR_FONT_RC)

DEFAULT_POINTS_CSV = str(_BOXPLOT_DIR / "target_gene_high_low_expression_points_plotted.csv")
DEFAULT_STATS_CSV = str(_BOXPLOT_DIR / "high_low_phospho_comparison_by_site_plotted.csv")
DEFAULT_OUTPUT_DIR = str(_CPTAC_ROOT / "results/phosphosite_across_cancers_boxplots")
DEFAULT_SITE = "ENSG00000100644|S643"
DEFAULT_SITE_LABEL = "HIF1A_S643"
DEFAULT_CANCERS = ["HNSCC", "UCEC"]

PRESET_SITES: Dict[str, str] = {
  "HIF1A_S643": "ENSG00000100644|S643",
  "STAT3_Y705": "ENSG00000168610|Y705",
  "STAT3_S701": "ENSG00000168610|S701",
  "E2F4_S244": "ENSG00000205250|S244",
  "E2F4_T224": "ENSG00000205250|T224",
  "NFATC2_S53": "ENSG00000101096|S53",
  "HSF1_S326": "ENSG00000185122|S326",
}
DEFAULT_BATCH_SITE_LABELS = [
  "STAT3_Y705",
  "STAT3_S701",
  "E2F4_S244",
  "NFATC2_S53",
  "HSF1_S326",
]

# Match plot_activate_merged_across_cancers() in run_import_target_regulation_analysis.py
MERGED_BOX_STYLE = {
    "pair_offset": 0.20,
    "box_width": 0.34,
    "box_alpha": 0.88,
    "box_edgecolor": "#3A3A3A",
    "box_edgewidth": 0.85,
    "scatter_s": 10.0,
    "scatter_alpha": 0.4,
    "jitter_range": 0.06,
    "group_color_map": {
        "low": "#95B0B5",
        "high": "#E7A983",
    },
    "y_cap": 1.0,
    "star_y_frac": 0.90,
    "fig_height": 5.0,
    "fig_width_base": 3.0,
    "fig_width_per_group": 0.85,
    "fig_width_min": 11.0,
    "x_pad_extra": 0.06,
    "x_label_rotation": 35,
    "x_label_ha": "right",
    "x_label_fontsize": 14.5,
    "axis_label_fontsize": 15.5,
    "y_tick_labelsize": 14.5,
}

# Per-site overrides; other sites keep MERGED_BOX_STYLE defaults.
SITE_PLOT_OVERRIDES: Dict[str, Dict[str, object]] = {
    "HSF1_S326": {
        "auto_ylim_from_data": True,
        "y_upper_cap": 1.0,
        "ylim_data_padding_frac": 0.08,
        "ylim_bottom_extra_frac": 0.05,
    },
    "HIF1A_S643": {
        "cancer_types": ["HNSCC"],
        "pair_offset": 0.08,
        "box_width": 0.14,
        "jitter_range": 0.03,
        "scatter_s": 8.0,
        "fig_width": 3.4,
        "fig_height": 3.6,
        "x_label_rotation": 0,
        "x_label_ha": "center",
        "subplots_bottom": 0.16,
        "y_lim": (-0.6, 0.6),
        "y_ticks": [-0.6, -0.3, 0.0, 0.3, 0.6],
    },
}


def _site_plot_style(site_label: str) -> Dict[str, object]:
    style = dict(MERGED_BOX_STYLE)
    style.update(SITE_PLOT_OVERRIDES.get(str(site_label), {}))
    return style


def _ylim_for_plot(
    style: Dict[str, object],
    data_lists: List[List[float]],
) -> Tuple[float, float, float, List[float]]:
    if "y_lim" in style:
        y_lower, y_upper = style["y_lim"]
        y_lower, y_upper = float(y_lower), float(y_upper)
        star_y = y_upper * float(style["star_y_frac"])
        y_ticks = [float(t) for t in style.get("y_ticks", [y_lower, y_upper])]
        return y_lower, y_upper, star_y, y_ticks

    y_cap = float(style["y_cap"])
    star_y = y_cap * float(style["star_y_frac"])

    if not style.get("auto_ylim_from_data"):
        y_ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
        return -y_cap, y_cap, star_y, y_ticks

    from matplotlib.ticker import MaxNLocator

    values = [float(v) for sublist in data_lists for v in sublist if pd.notna(v)]
    if not values:
        y_ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
        return -y_cap, y_cap, star_y, y_ticks

    y_upper_cap = float(style.get("y_upper_cap", y_cap))
    data_min = float(min(values))
    data_max = float(max(values))
    span = max(data_max - data_min, max(abs(data_max), abs(data_min), 0.1))
    pad = float(style.get("ylim_data_padding_frac", 0.08))
    bottom_extra = float(style.get("ylim_bottom_extra_frac", 0.05))

    y_data_lower = data_min - pad * span
    y_data_upper = min(y_upper_cap, data_max + pad * span)
    y_range = y_data_upper - y_data_lower
    y_lower = y_data_lower - bottom_extra * y_range
    y_upper = y_upper_cap
    star_y = y_upper * float(style["star_y_frac"])
    y_ticks = MaxNLocator(nbins=5, prune=None).tick_values(y_lower, y_upper)
    return y_lower, y_upper, star_y, [float(t) for t in y_ticks]


def _paired_low_high_values(site_sub: pd.DataFrame) -> Tuple[List[float], List[float], int]:
    wide = (
        site_sub.pivot_table(
            index="target_gene_id",
            columns="phospho_group",
            values="group_mean_expression",
            aggfunc="mean",
        )
        if {"low", "high"}.issubset(set(site_sub["phospho_group"]))
        else pd.DataFrame()
    )
    if wide.empty or not {"low", "high"}.issubset(wide.columns):
        return [], [], 0

    wide = wide.sort_index()
    low = pd.to_numeric(wide["low"], errors="coerce")
    high = pd.to_numeric(wide["high"], errors="coerce")
    valid = low.notna() & high.notna()
    return low[valid].tolist(), high[valid].tolist(), int(valid.sum())


def resolve_site(site_label: str, site: Optional[str] = None) -> Tuple[str, str]:
    label = str(site_label).strip()
    if site:
        return str(site).strip(), label
    if label in PRESET_SITES:
        return PRESET_SITES[label], label
    if "|" in label:
        return label, label.split("|", 1)[-1]
    raise ValueError(f"Unknown site label {label!r}; pass --site or use a preset label.")


def detect_cancers_with_data(
    df_points: pd.DataFrame,
    site: str,
    pipeline: TargetRegulationBoxplotPipeline,
    cancer_order: Optional[List[str]] = None,
    direction_short: str = "Import",
    target_regulation: str = "activate",
) -> List[str]:
    cancer_order = cancer_order or list(DEFAULT_CANCER_LIST)
    df_plot = df_points[
        df_points["direction_short"].astype(str).eq(direction_short)
        & df_points["target_regulation"].astype(str).eq(target_regulation)
        & df_points["site"].astype(str).eq(site)
    ].copy()
    available: List[str] = []
    for cancer_type in cancer_order:
        cancer_sub = df_plot[df_plot["cancer_type"].astype(str).eq(cancer_type)].copy()
        if cancer_sub.empty:
            continue
        _, _, n_box_points = _paired_low_high_values(cancer_sub)
        if n_box_points >= pipeline.config.min_box_points:
            available.append(cancer_type)
    return available


def order_cancers_by_significance_and_delta(
    df_stats: pd.DataFrame,
    site: str,
    available_cancers: List[str],
    pipeline: TargetRegulationBoxplotPipeline,
    direction_short: str = "Import",
    target_regulation: str = "activate",
) -> List[str]:
    """Order cancers like Figure B: significance first, then high-low delta."""
    if not available_cancers:
        return []

    if df_stats.empty:
        return list(available_cancers)

    stat_sub = df_stats[
        df_stats["direction_short"].astype(str).eq(direction_short)
        & df_stats["target_regulation"].astype(str).eq(target_regulation)
        & df_stats["site"].astype(str).eq(site)
        & df_stats["cancer_type"].astype(str).isin([str(c) for c in available_cancers])
    ].copy()
    if stat_sub.empty:
        return list(available_cancers)

    sig_col = pipeline._plot_significance_column(stat_sub)
    stat_sub["significance"] = stat_sub[sig_col].fillna("ns").astype(str)
    stat_sub["significance_rank"] = stat_sub["significance"].map(
        TargetRegulationBoxplotPipeline._significance_rank
    )

    if pipeline.config.use_bh_pvalue_correction and "wilcoxon_q_bh" in stat_sub.columns:
        stat_sub["p_sort"] = pd.to_numeric(stat_sub["wilcoxon_q_bh"], errors="coerce").fillna(np.inf)
    elif "p_sort" in stat_sub.columns:
        stat_sub["p_sort"] = pd.to_numeric(stat_sub["p_sort"], errors="coerce").fillna(np.inf)
    else:
        stat_sub["p_sort"] = pd.to_numeric(stat_sub["wilcoxon_p_expected"], errors="coerce").fillna(np.inf)

    stat_sub["delta_high_minus_low"] = pd.to_numeric(
        stat_sub["delta_high_minus_low"],
        errors="coerce",
    ).fillna(-np.inf)

    stat_sub = stat_sub.sort_values(
        ["significance_rank", "p_sort", "delta_high_minus_low", "cancer_type"],
        ascending=[False, True, False, True],
    )
    ordered = stat_sub["cancer_type"].astype(str).drop_duplicates().tolist()
    for cancer_type in available_cancers:
        cancer_type = str(cancer_type)
        if cancer_type not in ordered:
            ordered.append(cancer_type)
    return ordered


def filter_significant_cancers(
    df_stats: pd.DataFrame,
    site: str,
    cancer_types: List[str],
    pipeline: TargetRegulationBoxplotPipeline,
    direction_short: str = "Import",
    target_regulation: str = "activate",
) -> List[str]:
    """Keep only cancers where the site is significant (BH by default)."""
    if not cancer_types or df_stats.empty:
        return list(cancer_types)

    stat_sub = df_stats[
        df_stats["direction_short"].astype(str).eq(direction_short)
        & df_stats["target_regulation"].astype(str).eq(target_regulation)
        & df_stats["site"].astype(str).eq(site)
        & df_stats["cancer_type"].astype(str).isin([str(c) for c in cancer_types])
    ].copy()
    if stat_sub.empty:
        return list(cancer_types)

    significant = {
        str(row["cancer_type"])
        for _, row in stat_sub.iterrows()
        if pipeline._plot_significance_from_row(row) != "ns"
    }
    return [cancer for cancer in cancer_types if str(cancer) in significant]


def _figure_size_for_n_cancers(n_cancers: int) -> Tuple[float, float]:
    """Match merged across-cancer boxplot figure sizing."""
    style = MERGED_BOX_STYLE
    fig_width = max(
        style["fig_width_min"],
        style["fig_width_per_group"] * n_cancers + style["fig_width_base"],
    )
    return fig_width, style["fig_height"]


def _tight_xlim_from_groups(
    cancer_groups: List[Dict[str, object]],
    *,
    box_width: float,
    jitter_range: float,
    x_pad_extra: float,
) -> Tuple[float, float]:
    x_pad = box_width / 2.0 + jitter_range + x_pad_extra
    x_min = float(cancer_groups[0]["x_low"]) - x_pad
    x_max = float(cancer_groups[-1]["x_high"]) + x_pad
    return x_min, x_max


def _draw_y_axis_tick_marks(
    ax,
    tick_len_axes: float = 0.020,
    color: str = "#333333",
    linewidth: float = 1.0,
) -> None:
    """Draw tick marks on the left side of the y-axis spine (outward toward labels)."""
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ymin, ymax = ax.get_ylim()
    tick_offset = tick_len_axes / 4.0
    for y in ax.get_yticks():
        if ymin - 0.01 <= y <= ymax + 0.01:
            ax.plot(
                [-tick_len_axes + tick_offset, 0.0],
                [y, y],
                transform=trans,
                color=color,
                linewidth=linewidth,
                clip_on=False,
                solid_capstyle="butt",
                zorder=12,
            )


def plot_site_across_cancers(
    df_points: pd.DataFrame,
    df_stats: pd.DataFrame,
    cancer_types: List[str],
    site: str,
    direction_short: str = "Import",
    target_regulation: str = "activate",
    site_label: Optional[str] = None,
    output_dir: Path | None = None,
    output_prefix: Optional[str] = None,
    dpi: int = 300,
    pipeline: Optional[TargetRegulationBoxplotPipeline] = None,
    significant_only: bool = False,
) -> Path:
    if pipeline is None:
        pipeline = TargetRegulationBoxplotPipeline(TempoConfig())

    site_label = site_label or DEFAULT_SITE_LABEL
    output_dir = output_dir or Path(DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = output_prefix or f"{site_label}_Import_activate_across_cancers_high_low_phospho_target_expression_boxplot"

    style = _site_plot_style(site_label)
    group_color_map = dict(style["group_color_map"])
    site_label_color_map = {
        "known_positive": "#C97B49",
        "new_predicted": "#5E8FA1",
    }

    df_plot = df_points.copy()
    df_plot = df_plot[
        df_plot["direction_short"].astype(str).eq(direction_short)
        & df_plot["target_regulation"].astype(str).eq(target_regulation)
        & df_plot["site"].astype(str).eq(site)
    ].copy()

    forced_cancers = style.get("cancer_types")
    if forced_cancers:
        cancer_types = [str(c) for c in forced_cancers]

    cancer_types = order_cancers_by_significance_and_delta(
        df_stats=df_stats,
        site=site,
        available_cancers=list(cancer_types),
        pipeline=pipeline,
        direction_short=direction_short,
        target_regulation=target_regulation,
    )
    if significant_only:
        filtered = filter_significant_cancers(
            df_stats=df_stats,
            site=site,
            cancer_types=cancer_types,
            pipeline=pipeline,
            direction_short=direction_short,
            target_regulation=target_regulation,
        )
        if filtered:
            cancer_types = filtered
        else:
            print(f"Warning: no significant cancers for {site_label}; skipping plot")
            raise ValueError(f"No significant cancers for site={site}")

    cancer_groups: List[Dict[str, object]] = []
    positions: List[float] = []
    data_lists: List[List[float]] = []
    box_colors: List[str] = []
    scatter_colors: List[str] = []
    x_centers: List[float] = []
    ticklabels: List[str] = []
    stat_pos: Dict[str, float] = {}
    pair_offset = style["pair_offset"]
    current_x = 1.0

    for cancer_type in cancer_types:
        cancer_sub = df_plot[df_plot["cancer_type"].astype(str).eq(cancer_type)].copy()
        if cancer_sub.empty:
            print(f"Warning: no points for {cancer_type} / {site}")
            continue

        low_values, high_values, n_box_points = _paired_low_high_values(cancer_sub)
        if n_box_points < pipeline.config.min_box_points:
            print(f"Warning: {cancer_type} has only {n_box_points} paired targets (< min_box_points)")
            continue

        x_low = current_x - pair_offset
        x_high = current_x + pair_offset
        cancer_groups.append(
            {
                "cancer_type": cancer_type,
                "x_low": x_low,
                "x_high": x_high,
                "low_values": low_values,
                "high_values": high_values,
            }
        )
        positions.extend([x_low, x_high])
        data_lists.extend([low_values, high_values])
        box_colors.extend([group_color_map["low"], group_color_map["high"]])
        scatter_colors.extend([group_color_map["low"], group_color_map["high"]])
        x_centers.append(current_x)
        ticklabels.append(cancer_type)
        stat_pos[cancer_type] = current_x
        current_x += 1.0

    if not cancer_groups:
        raise ValueError(f"No valid boxplot data for site={site} across cancers={cancer_types}")

    site_status = pipeline.site_label_status(
        df_plot.drop_duplicates(subset=["site", "ACC_ID", "RESIDUE", "POSITION"]),
        direction_short,
    )
    site_title_color = site_label_color_map.get(site_status, site_label_color_map["new_predicted"])

    if "fig_width" in style and "fig_height" in style:
        fig_width = float(style["fig_width"])
        fig_height = float(style["fig_height"])
    else:
        fig_width, fig_height = _figure_size_for_n_cancers(len(x_centers))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    rng = np.random.default_rng(pipeline.config.random_seed)
    pipeline._draw_scatter_then_boxplot(
        ax,
        data_lists,
        positions,
        box_colors,
        scatter_colors=scatter_colors,
        widths=style["box_width"],
        box_alpha=style["box_alpha"],
        scatter_s=style["scatter_s"],
        scatter_alpha=style["scatter_alpha"],
        jitter_range=style["jitter_range"],
        box_edgecolor=style["box_edgecolor"],
        box_edgewidth=style["box_edgewidth"],
        rng=rng,
    )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        ticklabels,
        rotation=style["x_label_rotation"],
        ha=style["x_label_ha"],
        fontsize=style["x_label_fontsize"],
    )
    ax.set_ylabel("Mean target expression", fontsize=style["axis_label_fontsize"])
    ax.set_xlabel("Cancer type", fontsize=style["axis_label_fontsize"])

    stat_sub = df_stats[
        df_stats["direction_short"].astype(str).eq(direction_short)
        & df_stats["target_regulation"].astype(str).eq(target_regulation)
        & df_stats["site"].astype(str).eq(site)
    ].copy() if not df_stats.empty else pd.DataFrame()

    y_lower, y_upper, star_y, y_ticks = _ylim_for_plot(style, data_lists)
    ax.set_ylim(y_lower, y_upper)
    ax.set_yticks(y_ticks)
    ax.set_xlim(
        *_tight_xlim_from_groups(
            cancer_groups,
            box_width=style["box_width"],
            jitter_range=style["jitter_range"],
            x_pad_extra=style["x_pad_extra"],
        )
    )

    ax.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.30)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax)
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["left"].set_color("#4A4A4A")

    if not stat_sub.empty:
        for _, stat_row in stat_sub.iterrows():
            cancer_type = str(stat_row.get("cancer_type", ""))
            sig_label = pipeline._plot_significance_from_row(stat_row)
            if sig_label == "ns" or cancer_type not in stat_pos:
                continue
            ax.text(
                stat_pos[cancer_type],
                star_y,
                sig_label,
                ha="center",
                va="center",
                fontsize=17.0,
                fontweight="normal",
                color="#2F2F2F",
            )

    ax.set_title(
        f"{site_label} | Import activate targets",
        fontsize=13.0,
        color=site_title_color,
        fontweight="bold" if site_status == "known_positive" else "normal",
        pad=8,
    )

    plt.tight_layout()
    fig.subplots_adjust(left=0.22, bottom=float(style.get("subplots_bottom", 0.22)))

    ax.tick_params(
        axis="y",
        which="major",
        labelsize=style["y_tick_labelsize"],
        length=0,
        pad=8,
        color="#333333",
    )
    ax.tick_params(axis="x", length=0)
    _draw_y_axis_tick_marks(ax)

    out_base = output_dir / output_prefix
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_base.with_suffix(f".{ext}"), dpi=dpi)
    plt.close(fig)

    summary_rows = []
    cancer_order_map = {str(cancer): idx + 1 for idx, cancer in enumerate(cancer_types)}
    for cancer_type in cancer_types:
        cancer_sub = df_plot[df_plot["cancer_type"].astype(str).eq(cancer_type)].copy()
        low_values, high_values, n_points = _paired_low_high_values(cancer_sub)
        if n_points == 0:
            continue
        row = {
            "cancer_type": cancer_type,
            "site": site,
            "site_label": site_label,
            "n_paired_targets": n_points,
            "mean_low": float(np.mean(low_values)),
            "mean_high": float(np.mean(high_values)),
            "delta_high_minus_low": float(np.mean(high_values) - np.mean(low_values)),
            "plot_order": cancer_order_map.get(str(cancer_type)),
        }
        if not stat_sub.empty:
            stat_row = stat_sub[stat_sub["cancer_type"].astype(str).eq(str(cancer_type))]
            if not stat_row.empty:
                stat_row = stat_row.iloc[0]
                row["significance"] = pipeline._plot_significance_from_row(stat_row)
                row["wilcoxon_p_expected"] = stat_row.get("wilcoxon_p_expected")
                row["wilcoxon_q_bh"] = stat_row.get("wilcoxon_q_bh")
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(output_dir / f"{output_prefix}_summary.csv", index=False)

    print(f"Saved: {out_base}.png")
    return out_base.with_suffix(".png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot phosphosite low/high target-expression boxes across cancers."
    )
    parser.add_argument(
        "--points-csv",
        default=DEFAULT_POINTS_CSV,
        help="Path to target_gene_high_low_expression_points_plotted.csv",
    )
    parser.add_argument(
        "--stats-csv",
        default=DEFAULT_STATS_CSV,
        help="Path to high_low_phospho_comparison_by_site_plotted.csv",
    )
    parser.add_argument("--site", default=None, help="Internal site key, e.g. ENSG00000100644|S643")
    parser.add_argument("--site-label", default=DEFAULT_SITE_LABEL, help="Display label, e.g. HIF1A_S643")
    parser.add_argument(
        "--site-labels",
        nargs="+",
        default=None,
        help="Batch mode: plot multiple preset site labels (overrides --site-label)",
    )
    parser.add_argument(
        "--cancers",
        nargs="+",
        default=None,
        help="Cancer types in plot order (default: auto-detect from data)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: phosphosite_across_cancers_boxplots under script dir)",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--significant-only",
        action="store_true",
        help="Plot only BH-significant cancers (default: all cancers with site data)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    points_path = Path(args.points_csv)
    if not points_path.is_absolute():
        points_path = base_dir / points_path

    stats_path = Path(args.stats_csv)
    if not stats_path.is_absolute():
        stats_path = base_dir / stats_path

    output_dir = Path(args.output_dir) if args.output_dir else Path(DEFAULT_OUTPUT_DIR)

    df_points = pd.read_csv(points_path, low_memory=False)
    df_stats = pd.read_csv(stats_path) if stats_path.exists() else pd.DataFrame()

    pipeline = TargetRegulationBoxplotPipeline(TempoConfig())

    site_labels = list(args.site_labels) if args.site_labels else [args.site_label]
    for site_label in site_labels:
        site, resolved_label = resolve_site(site_label, args.site if len(site_labels) == 1 else None)
        if args.cancers:
            cancer_types = list(args.cancers)
        else:
            cancer_types = detect_cancers_with_data(
                df_points=df_points,
                site=site,
                pipeline=pipeline,
            )
        if not cancer_types:
            print(f"Warning: skipping {resolved_label}; no cancers with enough paired targets")
            continue
        if args.significant_only:
            cancer_types = filter_significant_cancers(
                df_stats=df_stats,
                site=site,
                cancer_types=cancer_types,
                pipeline=pipeline,
            )
            if not cancer_types:
                print(f"Warning: skipping {resolved_label}; no significant cancers")
                continue
        site_style = _site_plot_style(resolved_label)
        forced_cancers = site_style.get("cancer_types")
        if forced_cancers:
            cancer_types = [str(c) for c in forced_cancers]
        print(f"Plotting {resolved_label} across: {', '.join(cancer_types)}")
        try:
            plot_site_across_cancers(
                df_points=df_points,
                df_stats=df_stats,
                cancer_types=cancer_types,
                site=site,
                site_label=resolved_label,
                output_dir=output_dir,
                dpi=args.dpi,
                pipeline=pipeline,
                significant_only=args.significant_only,
            )
        except ValueError as exc:
            print(f"Warning: skipping {resolved_label}: {exc}")


if __name__ == "__main__":
    main()
