#!/usr/bin/env python3
"""
Combined boxplot for all BH-significant Import activate-target phosphosites across cancers.

X-axis: one group per (cancer, site), labeled as "{CANCER}: {SITE_LABEL}".
Repress-target sites are excluded by default (same as per-cancer boxplots).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_CPTAC_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

_PIPELINE_RUN = _CPTAC_ROOT / "results" / "import_target_regulation"
_BOXPLOT_DIR = _PIPELINE_RUN / "high_low_phospho_boxplots"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd
import seaborn as sns

from run_import_target_regulation_analysis import (
    EDITABLE_VECTOR_FONT_RC,
    TargetRegulationBoxplotPipeline,
    TempoConfig,
)

matplotlib.rcParams.update(EDITABLE_VECTOR_FONT_RC)

DEFAULT_POINTS_CSV = _BOXPLOT_DIR / "target_gene_high_low_expression_points_plotted.csv"
DEFAULT_STATS_CSV = _BOXPLOT_DIR / "high_low_phospho_comparison_by_site_plotted.csv"
DEFAULT_SIG_CSV = _BOXPLOT_DIR / "significant_sites_by_cancer_plotted.csv"
DEFAULT_OUTPUT_DIR = _BOXPLOT_DIR / "all_significant_sites_combined"

FIGURE_TITLE = "All BH-significant Import activate-target phosphosites across cancers"
LEGEND_LABELS = {
    "low": "Low phospho",
    "high": "High phospho",
    "known_positive": "Known positive site",
    "new_predicted": "New predicted site",
}


def _draw_y_axis_tick_marks(
    ax,
    tick_len_axes: float = 0.022 / 3.0,
    color: str = "#333333",
    linewidth: float = 1.0,
) -> List[plt.Line2D]:
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ymin, ymax = ax.get_ylim()
    tick_artists: List[plt.Line2D] = []
    for tick in ax.yaxis.get_major_ticks():
        y = float(tick.get_loc())
        if not tick.label1.get_visible():
            continue
        if not str(tick.label1.get_text()).strip():
            continue
        if y < ymin - 1e-9 or y > ymax + 1e-9:
            continue
        line, = ax.plot(
            [-tick_len_axes + tick_len_axes / 4.0, 0.0],
            [y, y],
            transform=trans,
            color=color,
            linewidth=linewidth,
            clip_on=False,
            solid_capstyle="butt",
            zorder=12,
        )
        tick_artists.append(line)
    return tick_artists


def _paired_low_high_values(site_sub: pd.DataFrame) -> Tuple[List[float], List[float], int]:
    wide = site_sub.pivot_table(
        index="target_gene_id",
        columns="phospho_group",
        values="group_mean_expression",
        aggfunc="mean",
    )
    if wide.empty or not {"low", "high"}.issubset(wide.columns):
        return [], [], 0

    wide = wide.sort_index()
    low = pd.to_numeric(wide["low"], errors="coerce")
    high = pd.to_numeric(wide["high"], errors="coerce")
    valid = low.notna() & high.notna()
    return low[valid].tolist(), high[valid].tolist(), int(valid.sum())


def _format_x_label(cancer_type: str, site_label: str) -> str:
    return f"{cancer_type}: {site_label}"


Y_AXIS_PADDING_LOWER = 0.25
Y_AXIS_PADDING_UPPER = 0.35
SIG_LABEL_OFFSET_ABOVE_DATA = 0.12


def _dynamic_expression_ylim(
    values: Sequence[float],
    *,
    padding_lower: float = Y_AXIS_PADDING_LOWER,
    padding_upper: float = Y_AXIS_PADDING_UPPER,
    sig_label_offset: float = SIG_LABEL_OFFSET_ABOVE_DATA,
) -> Tuple[float, float, float]:
    """Set y limits from data extrema with modest fixed padding (~0.2–0.4)."""
    clean = [float(v) for v in values if pd.notna(v)]
    if not clean:
        return -1.0, 1.0, 0.85

    data_min = float(min(clean))
    data_max = float(max(clean))
    y_lower = data_min - padding_lower
    y_upper = data_max + padding_upper
    sig_y = data_max + sig_label_offset
    if sig_y >= y_upper:
        y_upper = sig_y + 0.08
    return y_lower, y_upper, sig_y


def load_significant_entries(
    df_sig: Optional[pd.DataFrame],
    df_stats: pd.DataFrame,
    pipeline: TargetRegulationBoxplotPipeline,
    direction_short: str = "Import",
    target_regulation: str = "activate",
) -> pd.DataFrame:
    if df_sig is not None and not df_sig.empty:
        out = df_sig.copy()
    else:
        stat_sub = df_stats[df_stats["direction_short"].astype(str).eq(direction_short)].copy()
        sig_col = pipeline._plot_significance_column(stat_sub)
        stat_sub["significance"] = stat_sub[sig_col].fillna("ns").astype(str)
        out = stat_sub[stat_sub["significance"].ne("ns")].copy()

    out = out[out["direction_short"].astype(str).eq(direction_short)].copy()
    if target_regulation:
        out = out[out["target_regulation"].astype(str).eq(str(target_regulation))].copy()
    if out.empty:
        return out

    sig_col = pipeline._plot_significance_column(out)
    out["significance"] = out[sig_col].fillna("ns").astype(str)
    out["significance_rank"] = out["significance"].map(
        TargetRegulationBoxplotPipeline._significance_rank
    )
    if pipeline.config.use_bh_pvalue_correction and "wilcoxon_q_bh" in out.columns:
        out["p_sort"] = pd.to_numeric(out["wilcoxon_q_bh"], errors="coerce").fillna(np.inf)
    elif "p_sort" in out.columns:
        out["p_sort"] = pd.to_numeric(out["p_sort"], errors="coerce").fillna(np.inf)
    else:
        out["p_sort"] = pd.to_numeric(out["wilcoxon_p_expected"], errors="coerce").fillna(np.inf)
    out["delta_high_minus_low"] = pd.to_numeric(out["delta_high_minus_low"], errors="coerce").fillna(-np.inf)
    out["high_mean_sort"] = pd.to_numeric(out.get("high_mean_sort", out.get("mean_high_phospho_expression")), errors="coerce").fillna(-np.inf)

    cancer_counts = out.groupby("cancer_type").size().sort_values(ascending=False)
    cancer_rank = {cancer: rank for rank, cancer in enumerate(cancer_counts.index.astype(str))}
    out["cancer_rank"] = out["cancer_type"].astype(str).map(cancer_rank).fillna(999).astype(int)
    out["cancer_n_sig"] = out["cancer_type"].astype(str).map(cancer_counts.astype(int)).fillna(0).astype(int)
    out = out.sort_values(
        ["cancer_rank", "significance_rank", "p_sort", "delta_high_minus_low", "site_label"],
        ascending=[True, False, True, False, True],
    )
    return out


def _site_status_lookup(
    df_points: pd.DataFrame,
    pipeline: TargetRegulationBoxplotPipeline,
    direction_short: str = "Import",
) -> Dict[Tuple[str, str], str]:
    lookup: Dict[Tuple[str, str], str] = {}
    group_cols = ["cancer_type", "site"]
    for (cancer_type, site), site_sub in df_points.groupby(group_cols, dropna=False):
        status = pipeline.site_label_status(
            site_sub.drop_duplicates(subset=["site", "ACC_ID", "RESIDUE", "POSITION"]),
            direction_short,
        )
        lookup[(str(cancer_type), str(site))] = status
    return lookup


def export_combined_sites_tables(
    df_points: pd.DataFrame,
    df_stats: pd.DataFrame,
    sig_entries: pd.DataFrame,
    output_dir: Path,
    pipeline: TargetRegulationBoxplotPipeline,
    direction_short: str = "Import",
    target_regulation: str = "activate",
) -> Dict[str, Path]:
    """Export Import activate-target site tables for cancers in the combined figure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cancer_types = sorted(sig_entries["cancer_type"].astype(str).unique().tolist())
    if not cancer_types:
        raise ValueError("No cancers found in significant entries.")

    stat_sub = df_stats[
        df_stats["direction_short"].astype(str).eq(direction_short)
        & df_stats["target_regulation"].astype(str).eq(target_regulation)
        & df_stats["cancer_type"].astype(str).isin(cancer_types)
    ].copy()
    if stat_sub.empty:
        raise ValueError("No Import activate site statistics found for selected cancers.")

    sig_col = pipeline._plot_significance_column(stat_sub)
    stat_sub["significance_bh"] = stat_sub[sig_col].fillna("ns").astype(str)
    stat_sub["is_bh_significant"] = stat_sub["significance_bh"].ne("ns")
    stat_sub["significance_rank"] = stat_sub["significance_bh"].map(
        TargetRegulationBoxplotPipeline._significance_rank
    )
    if pipeline.config.use_bh_pvalue_correction and "wilcoxon_q_bh" in stat_sub.columns:
        stat_sub["p_sort"] = pd.to_numeric(stat_sub["wilcoxon_q_bh"], errors="coerce").fillna(np.inf)
    elif "p_sort" in stat_sub.columns:
        stat_sub["p_sort"] = pd.to_numeric(stat_sub["p_sort"], errors="coerce").fillna(np.inf)
    else:
        stat_sub["p_sort"] = pd.to_numeric(stat_sub["wilcoxon_p_expected"], errors="coerce").fillna(np.inf)
    stat_sub["delta_high_minus_low"] = pd.to_numeric(stat_sub["delta_high_minus_low"], errors="coerce")

    cancer_sig_counts = (
        stat_sub.loc[stat_sub["is_bh_significant"], "cancer_type"]
        .astype(str)
        .value_counts()
        .sort_values(ascending=False)
    )
    cancer_rank = {cancer: rank for rank, cancer in enumerate(cancer_sig_counts.index.astype(str))}
    for cancer in cancer_types:
        cancer_rank.setdefault(cancer, len(cancer_rank))
    stat_sub["cancer_rank"] = stat_sub["cancer_type"].astype(str).map(cancer_rank).astype(int)
    stat_sub["cancer_n_bh_significant"] = (
        stat_sub["cancer_type"].astype(str).map(cancer_sig_counts).fillna(0).astype(int)
    )

    plotted_keys = set(
        zip(
            sig_entries["cancer_type"].astype(str),
            sig_entries["site"].astype(str),
        )
    )
    stat_sub["in_combined_figure"] = [
        (str(ct), str(site)) in plotted_keys
        for ct, site in zip(stat_sub["cancer_type"], stat_sub["site"])
    ]
    stat_sub["x_label"] = [
        _format_x_label(str(ct), str(sl))
        for ct, sl in zip(stat_sub["cancer_type"], stat_sub["site_label"])
    ]

    status_lookup = _site_status_lookup(
        df_points[
            df_points["direction_short"].astype(str).eq(direction_short)
            & df_points["target_regulation"].astype(str).eq(target_regulation)
            & df_points["cancer_type"].astype(str).isin(cancer_types)
        ],
        pipeline,
        direction_short,
    )
    stat_sub["site_status"] = [
        status_lookup.get((str(ct), str(site)), "new_predicted")
        for ct, site in zip(stat_sub["cancer_type"], stat_sub["site"])
    ]
    stat_sub["site_status_label"] = stat_sub["site_status"].map(
        {
            "known_positive": "Known positive site",
            "new_predicted": "New predicted site",
        }
    )

    stat_sub = stat_sub.sort_values(
        ["cancer_rank", "significance_rank", "p_sort", "delta_high_minus_low", "site_label"],
        ascending=[True, False, True, False, True],
    ).reset_index(drop=True)
    stat_sub["table_row_order"] = np.arange(1, len(stat_sub) + 1)

    export_cols = [
        "table_row_order",
        "cancer_type",
        "cancer_rank",
        "cancer_n_bh_significant",
        "direction_short",
        "target_regulation",
        "site",
        "site_label",
        "x_label",
        "tf_name",
        "site_status",
        "site_status_label",
        "n_paired_target_genes",
        "mean_low_phospho_expression",
        "mean_high_phospho_expression",
        "delta_high_minus_low",
        "expected_direction",
        "wilcoxon_p_raw",
        "significance_raw",
        "wilcoxon_q_bh",
        "significance_bh",
        "is_bh_significant",
        "in_combined_figure",
    ]
    export_cols = [col for col in export_cols if col in stat_sub.columns]
    detail_path = output_dir / "all_cancers_Import_activate_sites_comparison_table.csv"
    stat_sub[export_cols].to_csv(detail_path, index=False)

    sig_only = stat_sub[stat_sub["is_bh_significant"]].copy()
    sig_path = output_dir / "all_cancers_Import_activate_bh_significant_sites_table.csv"
    sig_only[export_cols].to_csv(sig_path, index=False)

    nonsig_only = stat_sub[~stat_sub["is_bh_significant"]].copy()
    nonsig_path = output_dir / "all_cancers_Import_activate_nonsignificant_sites_table.csv"
    nonsig_only[export_cols].to_csv(nonsig_path, index=False)

    summary_rows = []
    for cancer_type in sorted(cancer_types, key=lambda c: cancer_rank.get(c, 999)):
        cancer_sub = stat_sub[stat_sub["cancer_type"].astype(str).eq(cancer_type)]
        summary_rows.append(
            {
                "cancer_type": cancer_type,
                "cancer_rank": cancer_rank.get(cancer_type, 999),
                "n_import_activate_sites": int(len(cancer_sub)),
                "n_bh_significant": int(cancer_sub["is_bh_significant"].sum()),
                "n_nonsignificant": int((~cancer_sub["is_bh_significant"]).sum()),
                "n_in_combined_figure": int(cancer_sub["in_combined_figure"].sum()),
                "n_known_positive_sites": int(cancer_sub["site_status"].eq("known_positive").sum()),
                "n_new_predicted_sites": int(cancer_sub["site_status"].eq("new_predicted").sum()),
            }
        )
    summary_path = output_dir / "all_cancers_Import_activate_sites_by_cancer_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"Saved table: {detail_path} ({len(stat_sub)} rows)")
    print(f"Saved table: {sig_path} ({len(sig_only)} rows)")
    print(f"Saved table: {nonsig_path} ({len(nonsig_only)} rows)")
    print(f"Saved table: {summary_path}")
    return {
        "all_sites": detail_path,
        "significant": sig_path,
        "nonsignificant": nonsig_path,
        "summary": summary_path,
    }


def plot_all_significant_sites_combined(
    df_points: pd.DataFrame,
    df_stats: pd.DataFrame,
    df_sig: Optional[pd.DataFrame],
    output_dir: Path,
    dpi: int = 300,
    pipeline: Optional[TargetRegulationBoxplotPipeline] = None,
    direction_short: str = "Import",
    target_regulation: str = "activate",
) -> List[Path]:
    if pipeline is None:
        pipeline = TargetRegulationBoxplotPipeline(
            TempoConfig(use_bh_pvalue_correction=True)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    sig_entries = load_significant_entries(
        df_sig,
        df_stats,
        pipeline,
        direction_short,
        target_regulation=target_regulation,
    )
    if sig_entries.empty:
        raise ValueError("No significant sites found to plot.")

    group_color_map = {"low": "#8FAFBC", "high": "#E3A07A"}
    site_label_color_map = {"known_positive": "#C97B49", "new_predicted": "#5E8FA1"}

    pair_offset = 0.16
    site_step = 1.0
    box_width = 0.23
    x_padding = 0.35

    positions: List[float] = []
    data_lists: List[List[float]] = []
    box_colors: List[str] = []
    scatter_colors: List[str] = []
    x_centers: List[float] = []
    ticklabels: List[str] = []
    ticklabel_colors: List[str] = []
    stat_positions: List[float] = []
    stat_labels: List[str] = []
    plotted_rows: List[Dict[str, object]] = []

    current_x = 1.0

    for _, sig_row in sig_entries.iterrows():
        cancer_type = str(sig_row["cancer_type"])
        site = str(sig_row["site"])
        site_label = str(sig_row.get("site_label", site))
        target_regulation = str(sig_row.get("target_regulation", "activate"))

        site_sub = df_points[
            df_points["cancer_type"].astype(str).eq(cancer_type)
            & df_points["direction_short"].astype(str).eq(direction_short)
            & df_points["target_regulation"].astype(str).eq(target_regulation)
            & df_points["site"].astype(str).eq(site)
        ].copy()
        if site_sub.empty:
            continue

        low_values, high_values, n_box_points = _paired_low_high_values(site_sub)
        if n_box_points < pipeline.config.min_box_points:
            continue

        positions.extend([current_x - pair_offset, current_x + pair_offset])
        data_lists.extend([low_values, high_values])
        box_colors.extend([group_color_map["low"], group_color_map["high"]])
        scatter_colors.extend([group_color_map["low"], group_color_map["high"]])

        site_status = pipeline.site_label_status(
            site_sub.drop_duplicates(subset=["site", "ACC_ID", "RESIDUE", "POSITION"]),
            direction_short,
        )
        x_centers.append(current_x)
        ticklabels.append(_format_x_label(cancer_type, site_label))
        ticklabel_colors.append(site_label_color_map.get(site_status, site_label_color_map["new_predicted"]))
        stat_positions.append(current_x)
        stat_labels.append(pipeline._plot_significance_from_row(sig_row))

        plotted_rows.append(
            {
                "cancer_type": cancer_type,
                "site": site,
                "site_label": site_label,
                "target_regulation": target_regulation,
                "x_label": ticklabels[-1],
                "n_paired_target_genes": n_box_points,
                "significance_bh": stat_labels[-1],
                "delta_high_minus_low": float(sig_row.get("delta_high_minus_low", np.nan)),
                "wilcoxon_q_bh": float(sig_row.get("wilcoxon_q_bh", np.nan)),
            }
        )

        current_x += site_step

    if not data_lists:
        raise ValueError("No significant sites had enough paired target genes to plot.")

    manifest = pd.DataFrame(plotted_rows)
    manifest.to_csv(output_dir / "plotted_significant_sites_manifest.csv", index=False)

    export_combined_sites_tables(
        df_points,
        df_stats,
        sig_entries,
        output_dir,
        pipeline,
        direction_short=direction_short,
        target_regulation=target_regulation if target_regulation else "activate",
    )

    n_groups = len(x_centers)
    fig_width = max(12.0, 0.42 * n_groups + 4.0)
    fig_height = 5.2 if n_groups <= 20 else 5.8
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor("white")

    rng = np.random.default_rng(42)
    pipeline._draw_scatter_then_boxplot(
        ax,
        data_lists,
        positions,
        box_colors,
        scatter_colors=scatter_colors,
        widths=box_width,
        box_alpha=0.88,
        scatter_s=3.2,
        scatter_alpha=0.35,
        jitter_range=0.045,
        rng=rng,
    )

    label_rotation = 55 if n_groups > 12 else 0
    label_ha = "right" if label_rotation else "center"
    label_fontsize = 11.5 if n_groups > 24 else 12.5
    ax.set_xticks(x_centers)
    ax.set_xticklabels(ticklabels, rotation=label_rotation, ha=label_ha, fontsize=label_fontsize)
    for tick_label, tick_color in zip(ax.get_xticklabels(), ticklabel_colors):
        tick_label.set_color(tick_color)
        if tick_color == site_label_color_map["known_positive"]:
            tick_label.set_fontweight("bold")

    ax.set_ylabel("Mean target expression", fontsize=14.0)
    ax.set_xlabel("Cancer: phosphosite", fontsize=14.0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.28)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax)

    all_values = [value for sublist in data_lists for value in sublist]
    y_lower, y_upper, sig_y = _dynamic_expression_ylim(all_values)
    ax.set_ylim(y_lower, y_upper)

    for x_pos, sig_label in zip(stat_positions, stat_labels):
        if sig_label == "ns":
            continue
        ax.text(
            x_pos,
            sig_y,
            sig_label,
            ha="center",
            va="bottom",
            fontsize=13.5,
            color="#2F2F2F",
        )

    y_ticks = MaxNLocator(nbins=5, prune=None).tick_values(y_lower, y_upper)
    ax.set_yticks(y_ticks)
    ax.tick_params(axis="y", which="major", labelsize=12.5, length=0, pad=10, color="#333333")
    ax.set_xlim(x_centers[0] - pair_offset - x_padding, x_centers[-1] + pair_offset + x_padding)

    handles = [
        Patch(facecolor=group_color_map["low"], edgecolor="#333333", label=LEGEND_LABELS["low"]),
        Patch(facecolor=group_color_map["high"], edgecolor="#333333", label=LEGEND_LABELS["high"]),
        Line2D([0], [0], color=site_label_color_map["known_positive"], marker="o", linestyle="",
               markersize=5, label=LEGEND_LABELS["known_positive"]),
        Line2D([0], [0], color=site_label_color_map["new_predicted"], marker="o", linestyle="",
               markersize=5, label=LEGEND_LABELS["new_predicted"]),
    ]
    legend = fig.legend(
        handles=handles,
        frameon=False,
        fontsize=10.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04 if label_rotation else -0.10),
        ncol=4,
    )
    ax.set_title(FIGURE_TITLE, fontsize=14.5, pad=8)

    bottom = 0.34 if label_rotation else 0.24
    plt.tight_layout()
    fig.subplots_adjust(left=0.08, bottom=bottom)

    tick_artists = _draw_y_axis_tick_marks(ax)
    bbox_extra = tick_artists + list(ax.get_yticklabels()) + [ax.yaxis.get_label(), legend]

    saved_paths: List[Path] = []
    out_base = output_dir / "All_cancers_Import_activate_significant_sites_combined_boxplot"
    for ext in ["png", "pdf", "svg"]:
        out_path = out_base.with_suffix(f".{ext}")
        fig.savefig(
            out_path,
            dpi=dpi,
            bbox_inches="tight",
            bbox_extra_artists=bbox_extra,
            pad_inches=0.12,
        )
        saved_paths.append(out_path)
    plt.close(fig)

    print(f"Plotted {n_groups} significant (cancer, site) groups.")
    print(f"Saved: {out_base}.pdf")
    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot all BH-significant Import phosphosites across cancers in one combined figure."
    )
    parser.add_argument("--points-csv", type=Path, default=Path(DEFAULT_POINTS_CSV))
    parser.add_argument("--stats-csv", type=Path, default=Path(DEFAULT_STATS_CSV))
    parser.add_argument("--significant-csv", type=Path, default=Path(DEFAULT_SIG_CSV))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--include-repress",
        action="store_true",
        help="Include repress-target significant sites (default: activate only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(__file__).resolve().parent
    points_csv = args.points_csv if args.points_csv.is_absolute() else base / args.points_csv
    stats_csv = args.stats_csv if args.stats_csv.is_absolute() else base / args.stats_csv
    sig_csv = args.significant_csv if args.significant_csv.is_absolute() else base / args.significant_csv
    output_dir = args.output_dir if args.output_dir.is_absolute() else base / args.output_dir

    df_points = pd.read_csv(points_csv)
    df_stats = pd.read_csv(stats_csv)
    df_sig = pd.read_csv(sig_csv) if sig_csv.exists() else None

    plot_all_significant_sites_combined(
        df_points,
        df_stats,
        df_sig,
        output_dir,
        dpi=args.dpi,
        target_regulation="" if args.include_repress else "activate",
    )


if __name__ == "__main__":
    main()
