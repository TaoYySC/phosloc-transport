#!/usr/bin/env python3
"""Redraw four-arm merged across-cancer boxplots with min10 site filter.

Cohort: regulon_only pairs restricted to cancer×site rows that are evaluable
(n_paired_target_genes >= 10) in each arm's cptac_filtered comparison table.

Figure specs (manuscript panel, no legend):
  - overall size 87 × 47 mm (panel without legend)
  - legend removed
  - no x-axis label
  - tick labels 7 pt
  - y-axis label \"Mean target expression\" 8 pt
  - y-axis tick marks 0.9 mm
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo-relative results root (cptac_analysis/results)
RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from run_import_target_regulation_analysis import (  # noqa: E402
    TargetRegulationBoxplotPipeline,
    TempoConfig,
)

TWO_ROOT = RESULTS_ROOT / "hotspot_mixed_pred_filter_pure_regulon_only_twosided_20260929"

ARMS: List[Tuple[str, str, str, str]] = [
    # arm_folder, direction_short, regulation, filtered_csv_suffix
    ("Import_activate", "Import", "activate", "import_activate"),
    ("Import_repress", "Import", "repress", "import_repress"),
    ("Export_activate", "Export", "activate", "export_activate"),
    ("Export_repress", "Export", "repress", "export_repress"),
]

# Points live under Import_activate / Export_activate cptac (repress is symlink).
POINTS_ARM = {
    "Import_activate": "Import_activate",
    "Import_repress": "Import_activate",
    "Export_activate": "Export_activate",
    "Export_repress": "Export_activate",
}

FIG_W_IN = 87.0 / 25.4
FIG_H_IN = 47.0 / 25.4
PT_TICK = 7.0
PT_YLABEL = 8.0
TICK_LEN_PT = 0.9 / 25.4 * 72.0  # 0.9 mm → points
MIN_N = 10


def _apply_editable_fonts() -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": PT_TICK,
        }
    )


def _load_evaluable_sites(arm: str, suffix: str, regulation: str) -> pd.DataFrame:
    path = (
        TWO_ROOT
        / arm
        / "cptac_filtered"
        / f"two_side_high_low_phospho_comparison_by_site_{suffix}.csv"
    )
    df = pd.read_csv(path)
    df["n_paired_target_genes"] = pd.to_numeric(df["n_paired_target_genes"], errors="coerce")
    if "evaluable" in df.columns:
        keep = df["evaluable"].astype(bool)
    else:
        keep = df["n_paired_target_genes"].ge(MIN_N)
    out = df.loc[
        keep & df["target_regulation"].astype(str).eq(regulation),
        ["cancer_type", "site"],
    ].drop_duplicates()
    return out


def _load_pairs(points_arm: str) -> pd.DataFrame:
    path = (
        TWO_ROOT
        / points_arm
        / "cptac"
        / "merged_all_sites_boxplots"
        / "merged_high_low_expression_pairs.csv"
    )
    return pd.read_csv(path, low_memory=False)


def _plot_one_arm(
    pipeline: TargetRegulationBoxplotPipeline,
    *,
    arm: str,
    direction_short: str,
    regulation: str,
    df_pairs: pd.DataFrame,
    df_stats: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    group_color_map = {"low": "#0072B2", "high": "#C93C37"}

    df_sub = df_pairs[
        df_pairs["direction_short"].astype(str).eq(direction_short)
        & df_pairs["target_regulation"].astype(str).eq(regulation)
    ].copy()
    if df_sub.empty:
        print(f"[{arm}] no pairs after filter")
        return

    available_cancers = set(df_sub["cancer_type"].dropna().astype(str))
    stat_sub_direction = df_stats[
        df_stats["direction_short"].astype(str).eq(direction_short)
        & df_stats["target_regulation"].astype(str).eq(regulation)
    ].copy()
    if not stat_sub_direction.empty:
        order_stats = stat_sub_direction.rename(
            columns={
                "wilcoxon_p_expected": "mannwhitney_p",
                "wilcoxon_q_bh": "mannwhitney_q_bh",
                "mean_high_phospho_expression": "mean_high_activity",
            }
        )
        cancer_order = pipeline._ordered_cancers_by_activity_significance(
            order_stats,
            direction_short,
            [str(c) for c in pipeline.config.cancer_types if str(c) in available_cancers]
            + sorted(available_cancers - {str(c) for c in pipeline.config.cancer_types}),
        )
    else:
        cancer_order = [c for c in pipeline.config.cancer_types if str(c) in available_cancers]
        cancer_order.extend(sorted(available_cancers - set(cancer_order)))

    positions: List[float] = []
    data_lists: List[List[float]] = []
    box_colors: List[str] = []
    scatter_colors: List[str] = []
    x_centers: List[float] = []
    ticklabels: List[str] = []
    stat_pos: Dict[str, float] = {}
    summary_rows: List[Dict[str, object]] = []

    current_x = 1.0
    pair_offset = 0.20
    for cancer_type in cancer_order:
        csub = df_sub[df_sub["cancer_type"].astype(str).eq(str(cancer_type))].copy()
        if csub.empty:
            continue
        low_values = pd.to_numeric(csub["low_phospho_mean_expression"], errors="coerce")
        high_values = pd.to_numeric(csub["high_phospho_mean_expression"], errors="coerce")
        valid = low_values.notna() & high_values.notna()
        low_list = low_values[valid].tolist()
        high_list = high_values[valid].tolist()
        n_points = int(valid.sum())
        if n_points < pipeline.config.min_box_points:
            continue

        positions.extend([current_x - pair_offset, current_x + pair_offset])
        data_lists.extend([low_list, high_list])
        box_colors.extend([group_color_map["low"], group_color_map["high"]])
        scatter_colors.extend([group_color_map["low"], group_color_map["high"]])
        x_centers.append(current_x)
        ticklabels.append(str(cancer_type))
        stat_pos[str(cancer_type)] = current_x
        summary_rows.append(
            {
                "cancer_type": str(cancer_type),
                "direction_short": direction_short,
                "target_regulation": regulation,
                "n_paired_points": n_points,
                "n_unique_target_genes": int(csub.loc[valid, "target_gene_id"].nunique())
                if "target_gene_id" in csub.columns
                else n_points,
                "n_unique_sites": int(csub.loc[valid, "site"].nunique())
                if "site" in csub.columns
                else np.nan,
                "mean_low_phospho_expression": float(np.mean(low_list)),
                "mean_high_phospho_expression": float(np.mean(high_list)),
                "delta_high_minus_low": float(np.mean(high_list) - np.mean(low_list)),
            }
        )
        current_x += 1.0

    if not data_lists:
        print(f"[{arm}] no cancer groups passed min_box_points")
        return

    pd.DataFrame(summary_rows).to_csv(
        out_dir / f"{direction_short.lower()}_{regulation}_merged_across_cancers_plot_summary.csv",
        index=False,
    )
    df_pairs.to_csv(out_dir / "merged_high_low_expression_pairs.csv", index=False)
    df_stats.to_csv(
        out_dir / "merged_high_low_comparison_by_cancer_direction_regulation.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
    rng = np.random.default_rng(pipeline.config.random_seed)
    pipeline._draw_scatter_then_boxplot(
        ax,
        data_lists,
        positions,
        box_colors,
        scatter_colors=scatter_colors,
        widths=0.34,
        box_alpha=0.88,
        scatter_s=2.2,
        scatter_alpha=0.12,
        jitter_range=0.06,
        rng=rng,
        box_edgewidth=0.6,
        medianprops={"color": "#2F2F2F", "linewidth": 0.8},
        whiskerprops={"color": "#4A4A4A", "linewidth": 0.6},
        capprops={"color": "#4A4A4A", "linewidth": 0.6},
    )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(ticklabels, rotation=35, ha="right", fontsize=PT_TICK)
    ax.set_ylabel("Mean target expression", fontsize=PT_YLABEL, labelpad=2)
    ax.set_xlabel("")  # no "Cancer type"
    ax.tick_params(axis="x", labelsize=PT_TICK, length=0, pad=1.5, width=0.6)
    ax.tick_params(
        axis="y",
        labelsize=PT_TICK,
        length=TICK_LEN_PT,
        width=0.6,
        pad=1.5,
        direction="out",
    )
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.30)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["left"].set_color("#4A4A4A")
    ax.spines["bottom"].set_linewidth(0.6)

    y_cap = 1.0
    ax.set_ylim(-y_cap, y_cap)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    star_y = y_cap * 0.90

    stat_sub = df_stats[
        df_stats["direction_short"].astype(str).eq(direction_short)
        & df_stats["target_regulation"].astype(str).eq(regulation)
    ].copy()
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
                fontsize=PT_TICK,
                color="#2F2F2F",
            )

    # No legend: 87 × 47 mm is the panel size without legend.
    fig.set_size_inches(FIG_W_IN, FIG_H_IN)
    fig.subplots_adjust(left=0.155, right=0.985, bottom=0.24, top=0.96)

    import shutil

    prefix = f"{direction_short}_{regulation}_merged_all_cancers_high_low_phospho_target_expression_boxplot"
    for ext in ("png", "pdf", "svg"):
        out_path = out_dir / f"{prefix}.{ext}"
        fig.savefig(out_path, dpi=pipeline.config.dpi, bbox_inches=None, pad_inches=0)
        shutil.copy2(
            out_path,
            out_dir
            / f"{direction_short}_{regulation}_merged_all_cancers_high_low_hotspot_activity_boxplot.{ext}",
        )

    plt.close(fig)
    print(f"[{arm}] wrote {len(x_centers)} cancers → {out_dir}")


def main() -> None:
    _apply_editable_fonts()
    cfg = TempoConfig(
        test_alternative="two-sided",
        use_bh_pvalue_correction=True,
        dpi=300,
        min_box_points=3,
    )
    # Prefer LinkedOmics paths from an existing run_config if present
    run_cfg = TWO_ROOT / "Import_activate" / "cptac" / "run_config.json"
    if run_cfg.exists():
        import json

        meta = json.loads(run_cfg.read_text(encoding="utf-8"))
        for key in (
            "linkedomics_base",
            "chip_dir",
            "signed_regulon_path",
            "idmapping_path",
            "cancer_types",
        ):
            if key in meta and hasattr(cfg, key):
                setattr(cfg, key, meta[key])

    pipeline = TargetRegulationBoxplotPipeline(cfg)
    manifest_rows = []

    for arm, direction_short, regulation, suffix in ARMS:
        points_arm = POINTS_ARM[arm]
        pairs_all = _load_pairs(points_arm)
        evaluable = _load_evaluable_sites(arm, suffix, regulation)
        pairs = pairs_all.merge(evaluable, on=["cancer_type", "site"], how="inner")
        pairs = pairs[
            pairs["direction_short"].astype(str).eq(direction_short)
            & pairs["target_regulation"].astype(str).eq(regulation)
        ].copy()

        stats = pipeline._compare_merged_high_low_by_cancer_regulation(pairs)
        stats = pipeline._add_p_value_correction(
            stats,
            group_cols=["direction_short", "target_regulation"],
            p_col="wilcoxon_p_expected",
            q_col="wilcoxon_q_bh",
        )

        out_dir = TWO_ROOT / arm / "figure4" / "merged_all_hotspots_across_cancers"
        _plot_one_arm(
            pipeline,
            arm=arm,
            direction_short=direction_short,
            regulation=regulation,
            df_pairs=pairs,
            df_stats=stats,
            out_dir=out_dir,
        )
        # also refresh source under cptac/merged for this regulation snapshot
        src_out = TWO_ROOT / points_arm / "cptac" / "merged_all_sites_boxplots_min10"
        src_out.mkdir(parents=True, exist_ok=True)
        arm_src = src_out / f"{arm}"
        arm_src.mkdir(parents=True, exist_ok=True)
        pairs.to_csv(arm_src / "merged_high_low_expression_pairs.csv", index=False)
        stats.to_csv(
            arm_src / "merged_high_low_comparison_by_cancer_direction_regulation.csv",
            index=False,
        )

        manifest_rows.append(
            {
                "arm": arm,
                "n_evaluable_sites": int(len(evaluable)),
                "n_pairs_plotted": int(len(pairs)),
                "n_cancers": int(pairs["cancer_type"].nunique()) if not pairs.empty else 0,
                "fig_mm": "87x47",
                "out_dir": str(out_dir),
            }
        )

    man = pd.DataFrame(manifest_rows)
    man_path = TWO_ROOT / "merged_min10_87x47_redraw_manifest.csv"
    man.to_csv(man_path, index=False)
    print(man.to_string(index=False))
    print("Wrote", man_path)


if __name__ == "__main__":
    main()
