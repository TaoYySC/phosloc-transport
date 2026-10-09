#!/usr/bin/env python3
"""Figure 4 panels for hotspot-level CPTAC analysis (4b–4f).

Expects outputs from run_hotspot_target_regulation_analysis.py under --results-dir.
Writes figure4/ subdirectory; does not overwrite unit-site figure names.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from run_import_target_regulation_analysis import (  # noqa: E402
    EDITABLE_VECTOR_FONT_RC,
    TargetRegulationBoxplotPipeline,
    TempoConfig,
    _draw_y_axis_tick_marks,
    _set_forest_effect_x_axis,
)
from plot_significant_sites_combined import (  # noqa: E402
    plot_all_significant_sites_combined,
)
from plot_phosphosite_across_cancers import (  # noqa: E402
    detect_cancers_with_data,
    plot_site_across_cancers,
)

matplotlib.rcParams.update(EDITABLE_VECTOR_FONT_RC)
sns.set_style("white")

_CPTAC_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = (
    _CPTAC_ROOT
    / "results/import_target_regulation_hotspot_v11_147pos_d3_platt_anchor_only"
)

CLASS_ORDER = ["known_containing", "known_proximal", "known_independent"]
CLASS_LABELS = {
    "known_containing": "Known-containing",
    "known_proximal": "Known-proximal",
    "known_independent": "Known-independent",
}
CLASS_COLORS = {
    "known_containing": "#C97B49",
    "known_proximal": "#7A9E9F",
    "known_independent": "#2F6F8F",
}

# Overridden by --direction-short / --target-regulation in main()
DIRECTION_SHORT = "Import"
TARGET_REGULATION = "activate"


def _save_fig(fig: plt.Figure, path_stem: Path, dpi: int = 300) -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"{path_stem}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _load_tables(results_dir: Path) -> Dict[str, pd.DataFrame]:
    points = results_dir / "all_target_gene_mean_expression_points.csv"
    stats = results_dir / "high_low_phospho_boxplots/high_low_phospho_comparison_by_site_all.csv"
    stratified = results_dir / "stratified_hotspot_associations/hotspot_association_stats_all.csv"
    figure_a = (
        results_dir
        / "target_logfc_activity_random_analysis"
        / "figureA_gene_level_observed_vs_random_logfc_by_cancer"
        / "FigureA_gene_level_observed_and_matched_random_logfc.csv"
    )
    forest_stats = (
        results_dir
        / "target_logfc_activity_random_analysis"
        / "figureA_gene_level_observed_vs_random_logfc_by_cancer"
        / "FigureA_gene_level_observed_vs_random_by_cancer_stats.csv"
    )
    concordance = (
        results_dir
        / "activity_concordance"
        / "significant_hotspot_max_direction_weighted_concordance.csv"
    )
    out: Dict[str, pd.DataFrame] = {}
    out["points"] = pd.read_csv(points) if points.exists() else pd.DataFrame()
    out["stats"] = pd.read_csv(stats) if stats.exists() else pd.DataFrame()
    if stratified.exists():
        out["stats_all"] = pd.read_csv(stratified)
    else:
        out["stats_all"] = out["stats"].copy()
    out["figure_a"] = pd.read_csv(figure_a) if figure_a.exists() else pd.DataFrame()
    out["forest_stats"] = pd.read_csv(forest_stats) if forest_stats.exists() else pd.DataFrame()
    out["concordance"] = pd.read_csv(concordance) if concordance.exists() else pd.DataFrame()
    return out


def _bh_sig_for_plot(
    stats: pd.DataFrame,
    pipeline: TargetRegulationBoxplotPipeline,
) -> pd.DataFrame:
    """BH q<0.05 rows; optionally keep only hypothesis-concordant deltas."""
    if stats.empty or "wilcoxon_q_bh" not in stats.columns:
        return stats.iloc[0:0].copy()
    out = stats[pd.to_numeric(stats["wilcoxon_q_bh"], errors="coerce") < 0.05].copy()
    if getattr(pipeline.config, "require_expected_direction", False) and not out.empty:
        before = len(out)
        out = pipeline._filter_hypothesis_concordant(
            out, direction_short=DIRECTION_SHORT, target_regulation=TARGET_REGULATION
        )
        print(
            f"expected-direction filter: {before} BH-sig → {len(out)} hypothesis-concordant "
            f"({DIRECTION_SHORT}×{TARGET_REGULATION})"
        )
    return out


def plot_4b_nuclear_hotspot_boxplots(
    points: pd.DataFrame,
    stats: pd.DataFrame,
    output_dir: Path,
    pipeline: TargetRegulationBoxplotPipeline,
) -> None:
    """4b: combined BH-significant hotspot activate-target boxplots for DIRECTION_SHORT."""
    if points.empty or stats.empty:
        print("4b skipped: missing points/stats")
        return
    panel_dir = output_dir / (
        "figure4b_nuclear_hotspot_targets"
        if DIRECTION_SHORT == "Import"
        else "figure4b_cytoplasmic_hotspot_targets"
    )
    panel_dir.mkdir(parents=True, exist_ok=True)

    activate_stats = stats[
        stats["direction_short"].astype(str).eq(DIRECTION_SHORT)
        & stats["target_regulation"].astype(str).eq(TARGET_REGULATION)
    ].copy()
    if "wilcoxon_q_bh" not in activate_stats.columns:
        activate_stats = pipeline._add_p_value_correction(
            activate_stats,
            group_cols=["cancer_type", "direction_short", "target_regulation"],
        )
    sig_for_plot = _bh_sig_for_plot(activate_stats, pipeline)
    # Always export full site tables (sig + nonsig, all directions); plot may be empty.
    try:
        from plot_significant_sites_combined import export_combined_sites_tables

        export_combined_sites_tables(
            points[
                points["direction_short"].astype(str).eq(DIRECTION_SHORT)
                & points["target_regulation"].astype(str).eq(TARGET_REGULATION)
            ].copy(),
            activate_stats,
            sig_for_plot,
            panel_dir,
            pipeline,
            direction_short=DIRECTION_SHORT,
            target_regulation=TARGET_REGULATION,
        )
    except Exception as exc:
        print(f"4b table export failed: {exc}")

    if sig_for_plot.empty:
        print(
            f"4b plot skipped: no hypothesis-concordant BH-significant "
            f"{TARGET_REGULATION} hotspots"
        )
        # Remove stale plot artifacts from prior runs
        for pattern in (
            f"All_cancers_{DIRECTION_SHORT}_{TARGET_REGULATION}_significant_sites_combined_boxplot.*",
            f"Figure4b_*_{TARGET_REGULATION}_targets_combined.*",
            "plotted_significant_sites_manifest.csv",
        ):
            for stale in panel_dir.glob(pattern):
                stale.unlink(missing_ok=True)
                print(f"removed stale: {stale.name}")
        return

    sig = sig_for_plot
    # Prefer hotspot_label for display
    if "hotspot_label" in sig.columns:
        sig["site_label"] = sig["hotspot_label"].fillna(sig.get("site_label"))
    plotted = points[
        points["direction_short"].astype(str).eq(DIRECTION_SHORT)
        & points["target_regulation"].astype(str).eq(TARGET_REGULATION)
    ].copy()
    if "hotspot_label" in plotted.columns:
        plotted["site_label"] = plotted["hotspot_label"].fillna(plotted.get("site_label"))
    # Ensure hotspot_class is on points for axis coloring
    if "hotspot_class" not in plotted.columns and "hotspot_class" in sig.columns:
        plotted = plotted.merge(
            sig[["site", "hotspot_class"]].drop_duplicates("site"),
            on="site",
            how="left",
        )
    elif "hotspot_class" in plotted.columns and "hotspot_class" in sig.columns:
        # fill gaps from stats
        class_map = sig.drop_duplicates("site").set_index("site")["hotspot_class"]
        plotted["hotspot_class"] = plotted["hotspot_class"].fillna(plotted["site"].map(class_map))

    try:
        plot_all_significant_sites_combined(
            plotted,
            activate_stats,
            sig,
            panel_dir,
            dpi=pipeline.config.dpi,
            pipeline=pipeline,
            direction_short=DIRECTION_SHORT,
            target_regulation=TARGET_REGULATION,
        )
        # Rename default combined stem if present
        stem_in = f"All_cancers_{DIRECTION_SHORT}_{TARGET_REGULATION}_significant_sites_combined_boxplot"
        stem_out = (
            f"Figure4b_nuclear_hotspot_{TARGET_REGULATION}_targets_combined"
            if DIRECTION_SHORT == "Import"
            else f"Figure4b_cytoplasmic_hotspot_{TARGET_REGULATION}_targets_combined"
        )
        for src in panel_dir.glob(f"{stem_in}.*"):
            dst = panel_dir / src.name.replace(stem_in, stem_out)
            if src != dst:
                shutil.copy2(src, dst)
    except Exception as exc:
        print(f"4b combined plot failed: {exc}")


def plot_4c_observed_vs_random(
    results_dir: Path,
    output_dir: Path,
) -> None:
    """4c: copy / re-export observed vs matched-random forest from extended analysis."""
    panel_dir = output_dir / "figure4c_observed_vs_random"
    panel_dir.mkdir(parents=True, exist_ok=True)
    src_dir = (
        results_dir
        / "target_logfc_activity_random_analysis"
        / "figureA_gene_level_observed_vs_random_logfc_by_cancer"
    )
    if not src_dir.exists():
        print("4c skipped: Figure A directory missing")
        return

    # Copy key figures with Figure4c names (prefer current DIRECTION × TARGET_REGULATION)
    copied = 0
    tag = f"{DIRECTION_SHORT}_{TARGET_REGULATION}".lower()
    for path in sorted(src_dir.iterdir()):
        if path.suffix.lower() not in {".pdf", ".png", ".svg", ".csv"}:
            continue
        name = path.name
        name_l = name.lower()
        # Keep CSVs; for plots require matching direction×regulation when tagged
        is_plot = path.suffix.lower() in {".pdf", ".png", ".svg"}
        if is_plot and ("import_" in name_l or "export_" in name_l) and tag not in name_l:
            continue
        if "forest" in name_l or "observed_vs" in name_l or "matched_random" in name_l:
            stem = f"Figure4c_{path.stem}"
            shutil.copy2(path, panel_dir / f"{stem}{path.suffix}")
            copied += 1
        elif path.suffix == ".csv":
            shutil.copy2(path, panel_dir / path.name)
            copied += 1
    print(f"4c: copied {copied} artifacts from Figure A outputs ({DIRECTION_SHORT}/{TARGET_REGULATION})")


def plot_4d_stratification(
    points: pd.DataFrame,
    stats: pd.DataFrame,
    output_dir: Path,
    pipeline: TargetRegulationBoxplotPipeline,
    dpi: int = 300,
) -> None:
    """4d: three hotspot classes as merged-style high/low target-expression boxes.

    Same visual language as Import_activate_merged_all_cancers_*: for each class,
    pool cancer×hotspot×target pairs and show low vs high boxes with Wilcoxon stars.
    """
    from scipy.stats import wilcoxon

    panel_dir = output_dir / "figure4d_hotspot_class_stratification"
    panel_dir.mkdir(parents=True, exist_ok=True)

    # Supplement: previous per cancer×hotspot stats by class
    if not stats.empty and "hotspot_class" in stats.columns:
        df_stats = stats[
            stats["direction_short"].astype(str).eq(DIRECTION_SHORT)
            & stats["target_regulation"].astype(str).eq(TARGET_REGULATION)
        ].copy()
        if not df_stats.empty:
            df_stats["wilcoxon_q_bh"] = pd.to_numeric(df_stats.get("wilcoxon_q_bh"), errors="coerce")
            df_stats["delta_high_minus_low"] = pd.to_numeric(
                df_stats.get("delta_high_minus_low"), errors="coerce"
            )
            df_stats["is_sig"] = df_stats["wilcoxon_q_bh"] < 0.05
            df_stats.to_csv(panel_dir / f"Figure4d_{TARGET_REGULATION}_hotspot_stats_by_class.csv", index=False)

    if points.empty or "hotspot_class" not in points.columns:
        print("4d skipped: need Import-activate points with hotspot_class")
        return

    pts = points[
        points["direction_short"].astype(str).eq(DIRECTION_SHORT)
        & points["target_regulation"].astype(str).eq(TARGET_REGULATION)
        & points["hotspot_class"].astype(str).isin(CLASS_ORDER)
    ].copy()
    if pts.empty:
        print("4d skipped: no class-labeled Import-activate points")
        return

    try:
        pairs = pipeline._build_merged_high_low_pairs(pts)
    except Exception as exc:
        print(f"4d skipped: merged pairs failed ({exc})")
        return
    if pairs.empty:
        print("4d skipped: empty merged pairs")
        return

    site_class = pts[["site", "hotspot_class"]].dropna().drop_duplicates("site")
    site_class["hotspot_class"] = site_class["hotspot_class"].astype(str)
    pairs = pairs.merge(site_class, on="site", how="left")
    pairs = pairs[pairs["hotspot_class"].astype(str).isin(CLASS_ORDER)].copy()
    pairs.to_csv(panel_dir / "Figure4d_merged_pairs_by_class.csv", index=False)

    group_color = {"low": "#0072B2", "high": "#C93C37"}
    positions: List[float] = []
    data_lists: List[List[float]] = []
    box_colors: List[str] = []
    x_centers: List[float] = []
    ticklabels: List[str] = []
    summary_rows: List[Dict[str, object]] = []
    pair_offset = 0.20
    current_x = 1.0

    for cls in CLASS_ORDER:
        sub = pairs[pairs["hotspot_class"].astype(str).eq(cls)].copy()
        low = pd.to_numeric(sub["low_phospho_mean_expression"], errors="coerce")
        high = pd.to_numeric(sub["high_phospho_mean_expression"], errors="coerce")
        valid = low.notna() & high.notna()
        low_list = low[valid].astype(float).tolist()
        high_list = high[valid].astype(float).tolist()
        n_pairs = int(valid.sum())
        n_hotspots = int(sub.loc[valid, "site"].nunique()) if n_pairs else 0
        n_genes = (
            int(sub.loc[valid, "target_gene_id"].nunique())
            if n_pairs and "target_gene_id" in sub.columns
            else 0
        )
        if n_pairs < pipeline.config.min_box_points:
            summary_rows.append(
                {
                    "hotspot_class": cls,
                    "label": CLASS_LABELS[cls],
                    "n_paired_points": n_pairs,
                    "n_unique_hotspots": n_hotspots,
                    "n_unique_target_genes": n_genes,
                    "mean_low": np.nan,
                    "mean_high": np.nan,
                    "delta_high_minus_low": np.nan,
                    "wilcoxon_p_expected": np.nan,
                    "significance": "ns",
                    "plotted": False,
                }
            )
            continue

        alternative = pipeline._expected_high_low_alternative(DIRECTION_SHORT, TARGET_REGULATION)
        p_value = np.nan
        try:
            if not np.allclose(np.asarray(high_list) - np.asarray(low_list), 0):
                p_value = float(wilcoxon(high_list, low_list, alternative=alternative).pvalue)
        except ValueError:
            p_value = np.nan
        mean_low = float(np.mean(low_list))
        mean_high = float(np.mean(high_list))
        summary_rows.append(
            {
                "hotspot_class": cls,
                "label": CLASS_LABELS[cls],
                "n_paired_points": n_pairs,
                "n_unique_hotspots": n_hotspots,
                "n_unique_target_genes": n_genes,
                "mean_low": mean_low,
                "mean_high": mean_high,
                "delta_high_minus_low": mean_high - mean_low,
                "wilcoxon_p_expected": p_value,
                "significance": pipeline._p_to_stars(p_value),
                "plotted": True,
            }
        )
        positions.extend([current_x - pair_offset, current_x + pair_offset])
        data_lists.extend([low_list, high_list])
        box_colors.extend([group_color["low"], group_color["high"]])
        x_centers.append(current_x)
        ticklabels.append(CLASS_LABELS[cls])
        current_x += 1.0

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = pipeline._add_p_value_correction(
            summary,
            group_cols=[],
            p_col="wilcoxon_p_expected",
            q_col="wilcoxon_q_bh",
        )
    summary.to_csv(panel_dir / "Figure4d_class_summary.csv", index=False)

    if not data_lists:
        print("4d skipped: no class passed min_box_points")
        return

    sig_labels: List[str] = []
    for cls in CLASS_ORDER:
        row = summary[summary["hotspot_class"].eq(cls)]
        if row.empty or not bool(row.iloc[0].get("plotted", False)):
            continue
        if pipeline.config.use_bh_pvalue_correction and "significance_bh" in row.columns:
            sig_labels.append(str(row.iloc[0]["significance_bh"]))
        else:
            sig_labels.append(str(row.iloc[0]["significance"]))

    fig_w = max(7.5, 2.2 * len(x_centers) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, 5.0))
    rng = np.random.default_rng(pipeline.config.random_seed)
    pipeline._draw_scatter_then_boxplot(
        ax,
        data_lists,
        positions,
        box_colors,
        scatter_colors=box_colors,
        widths=0.34,
        box_alpha=0.88,
        scatter_s=3.5,
        scatter_alpha=0.12,
        jitter_range=0.06,
        rng=rng,
    )
    ax.set_xticks(x_centers)
    ax.set_xticklabels(ticklabels, rotation=0, ha="center", fontsize=12)
    ax.set_ylabel("Mean target expression", fontsize=14)
    ax.set_xlabel("Hotspot class", fontsize=14)
    ax.set_title("Target expression by hotspot class", fontsize=13)
    ax.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.30)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax)
    y_cap = 1.0
    ax.set_ylim(-y_cap, y_cap)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    star_y = y_cap * 0.90
    for x, lab in zip(x_centers, sig_labels):
        if lab and lab != "ns":
            ax.text(x, star_y, lab, ha="center", va="center", fontsize=16, color="#2F2F2F")
    # Legend outside axes, top-right of the figure (avoids overlap with boxes)
    ax.legend(
        handles=[
            Patch(
                facecolor=group_color["low"],
                edgecolor="#333333",
                label="Low phospho (hotspot)",
            ),
            Patch(
                facecolor=group_color["high"],
                edgecolor="#333333",
                label="High phospho (hotspot)",
            ),
        ],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=10,
    )
    _draw_y_axis_tick_marks(ax)
    fig.tight_layout()
    _save_fig(fig, panel_dir / "Figure4d_hotspot_class_stratification", dpi=dpi)
    # Alias matching merged-figure naming
    for ext in ("pdf", "png", "svg"):
        src = panel_dir / f"Figure4d_hotspot_class_stratification.{ext}"
        if src.exists():
            shutil.copy2(src, panel_dir / f"Figure4d_class_merged_high_low_boxplot.{ext}")
    # Remove obsolete abstract median-only panel if present
    for ext in ("pdf", "png", "svg"):
        old = panel_dir / f"Figure4d_median_effect_by_class.{ext}"
        if old.exists():
            old.unlink()
    print(f"4d merged-style class boxplot → {panel_dir}")
    print(summary.to_string(index=False))


SIZE_BIN_ORDER = ["1", "2", "3", "≥4"]
SIZE_BIN_LABELS_MEASURED = {
    "1": "1 measured",
    "2": "2 measured",
    "3": "3 measured",
    "≥4": "≥4 measured",
}
SIZE_BIN_LABELS_MEMBERS = {
    "1": "1 member",
    "2": "2 members",
    "3": "3 members",
    "≥4": "≥4 members",
}
# Back-compat alias
SIZE_BIN_LABELS = SIZE_BIN_LABELS_MEASURED



def _hotspot_size_bin(n: object) -> str:
    try:
        v = int(float(n))
    except (TypeError, ValueError):
        return ""
    if v <= 0:
        return ""
    if v == 1:
        return "1"
    if v == 2:
        return "2"
    if v == 3:
        return "3"
    return "≥4"


def _plot_size_column_stratification(
    points: pd.DataFrame,
    stats: pd.DataFrame,
    output_dir: Path,
    pipeline: TargetRegulationBoxplotPipeline,
    size_col: str,
    panel_subdir: str,
    xlabel: str,
    title: str,
    bin_labels: Dict[str, str],
    dpi: int = 300,
) -> None:
    """Stratify Import-activate high/low target expression by a size column.

    Bins: 1 / 2 / 3 / ≥4. Same merged-box visual language as Figure 4d.
    """
    from scipy.stats import wilcoxon

    panel_dir = output_dir / panel_subdir
    panel_dir.mkdir(parents=True, exist_ok=True)

    if not stats.empty and size_col in stats.columns:
        df_stats = stats[
            stats["direction_short"].astype(str).eq(DIRECTION_SHORT)
            & stats["target_regulation"].astype(str).eq(TARGET_REGULATION)
        ].copy()
        if not df_stats.empty:
            df_stats[size_col] = pd.to_numeric(df_stats[size_col], errors="coerce")
            df_stats["size_bin"] = df_stats[size_col].map(_hotspot_size_bin)
            df_stats["wilcoxon_q_bh"] = pd.to_numeric(df_stats.get("wilcoxon_q_bh"), errors="coerce")
            df_stats["delta_high_minus_low"] = pd.to_numeric(
                df_stats.get("delta_high_minus_low"), errors="coerce"
            )
            df_stats["is_sig"] = df_stats["wilcoxon_q_bh"] < 0.05
            df_stats.to_csv(panel_dir / f"activate_hotspot_stats_by_{size_col}.csv", index=False)
            hotspot_key = "hotspot_id" if "hotspot_id" in df_stats.columns else "site"
            bin_stats = (
                df_stats.groupby("size_bin", dropna=False)
                .agg(
                    n_cancer_hotspot_pairs=("site", "size"),
                    n_unique_hotspots=(hotspot_key, "nunique"),
                    n_bh_significant=("is_sig", "sum"),
                    median_delta=("delta_high_minus_low", "median"),
                    mean_delta=("delta_high_minus_low", "mean"),
                )
                .reindex(SIZE_BIN_ORDER)
                .reset_index()
            )
            bin_stats.to_csv(panel_dir / f"activate_stats_summary_by_{size_col}.csv", index=False)

    if points.empty or size_col not in points.columns:
        print(f"{size_col} stratification skipped: missing column on points")
        return

    pts = points[
        points["direction_short"].astype(str).eq(DIRECTION_SHORT)
        & points["target_regulation"].astype(str).eq(TARGET_REGULATION)
    ].copy()
    pts[size_col] = pd.to_numeric(pts[size_col], errors="coerce")
    pts["size_bin"] = pts[size_col].map(_hotspot_size_bin)
    pts = pts[pts["size_bin"].astype(str).isin(SIZE_BIN_ORDER)].copy()
    if pts.empty:
        print(f"{size_col} stratification skipped: no Import-activate points in bins")
        return

    try:
        pairs = pipeline._build_merged_high_low_pairs(pts)
    except Exception as exc:
        print(f"{size_col} stratification skipped: merged pairs failed ({exc})")
        return
    if pairs.empty:
        print(f"{size_col} stratification skipped: empty merged pairs")
        return

    site_bin = (
        pts[["cancer_type", "site", "size_bin", size_col]]
        .dropna(subset=["size_bin"])
        .drop_duplicates(["cancer_type", "site"])
    )
    pairs = pairs.merge(site_bin, on=["cancer_type", "site"], how="left")
    pairs = pairs[pairs["size_bin"].astype(str).isin(SIZE_BIN_ORDER)].copy()
    pairs.to_csv(panel_dir / f"merged_pairs_by_{size_col}.csv", index=False)

    group_color = {"low": "#0072B2", "high": "#C93C37"}
    positions: List[float] = []
    data_lists: List[List[float]] = []
    box_colors: List[str] = []
    x_centers: List[float] = []
    ticklabels: List[str] = []
    summary_rows: List[Dict[str, object]] = []
    pair_offset = 0.20
    current_x = 1.0

    for size_bin in SIZE_BIN_ORDER:
        sub = pairs[pairs["size_bin"].astype(str).eq(size_bin)].copy()
        low = pd.to_numeric(sub["low_phospho_mean_expression"], errors="coerce")
        high = pd.to_numeric(sub["high_phospho_mean_expression"], errors="coerce")
        valid = low.notna() & high.notna()
        low_list = low[valid].astype(float).tolist()
        high_list = high[valid].astype(float).tolist()
        n_pairs = int(valid.sum())
        n_hotspots = int(sub.loc[valid, "site"].nunique()) if n_pairs else 0
        n_genes = (
            int(sub.loc[valid, "target_gene_id"].nunique())
            if n_pairs and "target_gene_id" in sub.columns
            else 0
        )
        n_cancers = int(sub.loc[valid, "cancer_type"].nunique()) if n_pairs else 0
        if n_pairs < pipeline.config.min_box_points:
            summary_rows.append(
                {
                    "size_bin": size_bin,
                    "label": bin_labels[size_bin],
                    "n_paired_points": n_pairs,
                    "n_unique_hotspots": n_hotspots,
                    "n_unique_target_genes": n_genes,
                    "n_unique_cancers": n_cancers,
                    "mean_low": np.nan,
                    "mean_high": np.nan,
                    "delta_high_minus_low": np.nan,
                    "wilcoxon_p_expected": np.nan,
                    "significance": "ns",
                    "plotted": False,
                }
            )
            continue

        alternative = pipeline._expected_high_low_alternative(DIRECTION_SHORT, TARGET_REGULATION)
        p_value = np.nan
        try:
            if not np.allclose(np.asarray(high_list) - np.asarray(low_list), 0):
                p_value = float(wilcoxon(high_list, low_list, alternative=alternative).pvalue)
        except ValueError:
            p_value = np.nan
        mean_low = float(np.mean(low_list))
        mean_high = float(np.mean(high_list))
        summary_rows.append(
            {
                "size_bin": size_bin,
                "label": bin_labels[size_bin],
                "n_paired_points": n_pairs,
                "n_unique_hotspots": n_hotspots,
                "n_unique_target_genes": n_genes,
                "n_unique_cancers": n_cancers,
                "mean_low": mean_low,
                "mean_high": mean_high,
                "delta_high_minus_low": mean_high - mean_low,
                "wilcoxon_p_expected": p_value,
                "significance": pipeline._p_to_stars(p_value),
                "plotted": True,
            }
        )
        positions.extend([current_x - pair_offset, current_x + pair_offset])
        data_lists.extend([low_list, high_list])
        box_colors.extend([group_color["low"], group_color["high"]])
        x_centers.append(current_x)
        ticklabels.append(bin_labels[size_bin])
        current_x += 1.0

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = pipeline._add_p_value_correction(
            summary,
            group_cols=[],
            p_col="wilcoxon_p_expected",
            q_col="wilcoxon_q_bh",
        )
    summary.to_csv(panel_dir / f"merged_boxplot_summary_by_{size_col}.csv", index=False)

    if not data_lists:
        print(f"{size_col} stratification skipped: no bin passed min_box_points")
        return

    sig_labels: List[str] = []
    for size_bin in SIZE_BIN_ORDER:
        row = summary[summary["size_bin"].eq(size_bin)]
        if row.empty or not bool(row.iloc[0].get("plotted", False)):
            continue
        if pipeline.config.use_bh_pvalue_correction and "significance_bh" in row.columns:
            sig_labels.append(str(row.iloc[0]["significance_bh"]))
        else:
            sig_labels.append(str(row.iloc[0]["significance"]))

    fig_w = max(7.5, 2.0 * len(x_centers) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, 5.0))
    rng = np.random.default_rng(pipeline.config.random_seed)
    pipeline._draw_scatter_then_boxplot(
        ax,
        data_lists,
        positions,
        box_colors,
        scatter_colors=box_colors,
        widths=0.34,
        box_alpha=0.88,
        scatter_s=3.5,
        scatter_alpha=0.12,
        jitter_range=0.06,
        rng=rng,
    )
    ax.set_xticks(x_centers)
    ax.set_xticklabels(ticklabels, rotation=0, ha="center", fontsize=12)
    ax.set_ylabel("Mean target expression", fontsize=14)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_title(title, fontsize=13)
    ax.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.30)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax)
    y_cap = 1.0
    ax.set_ylim(-y_cap, y_cap)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    star_y = y_cap * 0.90
    for x, lab in zip(x_centers, sig_labels):
        if lab and lab != "ns":
            ax.text(x, star_y, lab, ha="center", va="center", fontsize=16, color="#2F2F2F")
    ax.legend(
        handles=[
            Patch(
                facecolor=group_color["low"],
                edgecolor="#333333",
                label="Low phospho (hotspot)",
            ),
            Patch(
                facecolor=group_color["high"],
                edgecolor="#333333",
                label="High phospho (hotspot)",
            ),
        ],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=10,
    )
    _draw_y_axis_tick_marks(ax)
    fig.tight_layout()
    _save_fig(
        fig,
        panel_dir / f"{DIRECTION_SHORT}_{TARGET_REGULATION}_merged_by_{size_col}_high_low_boxplot",
        dpi=dpi,
    )
    print(f"{size_col} stratification → {panel_dir}")
    print(summary.to_string(index=False))


def plot_measured_members_stratification(
    points: pd.DataFrame,
    stats: pd.DataFrame,
    output_dir: Path,
    pipeline: TargetRegulationBoxplotPipeline,
    dpi: int = 300,
) -> None:
    """Stratify by CPTAC-measured member count (cancer×hotspot)."""
    _plot_size_column_stratification(
        points,
        stats,
        output_dir,
        pipeline,
        size_col="n_measured_members",
        panel_subdir="stratification_by_n_measured_members",
        xlabel="CPTAC-measured members per hotspot",
        title="Target expression by n_measured_members",
        bin_labels=SIZE_BIN_LABELS_MEASURED,
        dpi=dpi,
    )


def plot_n_members_stratification(
    points: pd.DataFrame,
    stats: pd.DataFrame,
    output_dir: Path,
    pipeline: TargetRegulationBoxplotPipeline,
    dpi: int = 300,
) -> None:
    """Stratify by catalog hotspot size (Known/HC members: 1 / 2 / 3 / ≥4)."""
    _plot_size_column_stratification(
        points,
        stats,
        output_dir,
        pipeline,
        size_col="n_members",
        panel_subdir="stratification_by_n_members",
        xlabel="Known/HC members per hotspot",
        title="Target expression by n_members (anchor-only hotspots)",
        bin_labels=SIZE_BIN_LABELS_MEMBERS,
        dpi=dpi,
    )

def _select_hotspot_id(
    points: pd.DataFrame,
    stats: pd.DataFrame,
    *,
    gene_name: str,
    preferred_ids: Sequence[str],
    class_filter: Optional[str] = None,
) -> Optional[str]:
    """Pick a hotspot_id for an example panel (preferred list, else most cancers / best q)."""
    pts = points[points["tf_name"].astype(str).eq(gene_name)].copy()
    st = stats[stats["tf_name"].astype(str).eq(gene_name)].copy() if not stats.empty else stats
    if class_filter and "hotspot_class" in pts.columns:
        pts = pts[pts["hotspot_class"].astype(str).eq(class_filter)]
        if not st.empty and "hotspot_class" in st.columns:
            st = st[st["hotspot_class"].astype(str).eq(class_filter)]
    if pts.empty or "hotspot_id" not in pts.columns:
        return None
    avail = set(pts["hotspot_id"].dropna().astype(str))
    for hid in preferred_ids:
        if hid in avail:
            return hid
    act = st[
        st["direction_short"].astype(str).eq(DIRECTION_SHORT)
        & st["target_regulation"].astype(str).eq(TARGET_REGULATION)
    ] if not st.empty else pd.DataFrame()
    if not act.empty and "wilcoxon_q_bh" in act.columns:
        rank = (
            act.groupby("hotspot_id")
            .agg(n_cancer=("cancer_type", "nunique"), min_q=("wilcoxon_q_bh", "min"))
            .sort_values(["n_cancer", "min_q"], ascending=[False, True])
        )
        if not rank.empty:
            return str(rank.index[0])
    return str(pts["hotspot_id"].value_counts().index[0]) if avail else None


def _plot_hotspot_across_cancers_like_multi(
    points: pd.DataFrame,
    stats: pd.DataFrame,
    *,
    gene_name: str,
    hotspot_id: str,
    panel_name: str,
    output_dir: Path,
    pipeline: TargetRegulationBoxplotPipeline,
    class_filter: Optional[str] = None,
) -> None:
    """Same drawer as multi_cancer_significant_hotspots (plot_site_across_cancers)."""
    panel_dir = output_dir / panel_name
    panel_dir.mkdir(parents=True, exist_ok=True)

    pts = points[
        points["direction_short"].astype(str).eq(DIRECTION_SHORT)
        & points["target_regulation"].astype(str).eq(TARGET_REGULATION)
        & points["tf_name"].astype(str).eq(gene_name)
        & points["hotspot_id"].astype(str).eq(hotspot_id)
    ].copy()
    st = stats[
        stats["direction_short"].astype(str).eq(DIRECTION_SHORT)
        & stats["target_regulation"].astype(str).eq(TARGET_REGULATION)
        & stats["tf_name"].astype(str).eq(gene_name)
        & stats["hotspot_id"].astype(str).eq(hotspot_id)
    ].copy() if not stats.empty and "hotspot_id" in stats.columns else pd.DataFrame()
    if class_filter and "hotspot_class" in pts.columns:
        pts = pts[pts["hotspot_class"].astype(str).eq(class_filter)]
        if not st.empty and "hotspot_class" in st.columns:
            st = st[st["hotspot_class"].astype(str).eq(class_filter)]
    if pts.empty:
        print(f"{panel_name} skipped: no points for {gene_name} / {hotspot_id}")
        return

    hotspot_label = (
        str(pts["hotspot_label"].dropna().iloc[0])
        if "hotspot_label" in pts.columns and pts["hotspot_label"].notna().any()
        else gene_name
    )
    site = (
        str(pts["site"].dropna().iloc[0])
        if "site" in pts.columns and pts["site"].notna().any()
        else ""
    )
    cls = (
        str(pts["hotspot_class"].dropna().iloc[0])
        if "hotspot_class" in pts.columns and pts["hotspot_class"].notna().any()
        else (class_filter or "")
    )
    pts.to_csv(panel_dir / f"{gene_name}_hotspot_points.csv", index=False)
    st.to_csv(panel_dir / f"{gene_name}_hotspot_stats.csv", index=False)
    if not site or site == "nan":
        print(f"{panel_name} skipped: no site key for {hotspot_label}")
        return

    cancers = detect_cancers_with_data(
        df_points=points,
        site=site,
        pipeline=pipeline,
        cancer_order=list(pipeline.config.cancer_types),
    )
    if not cancers:
        print(f"{panel_name} skipped: no cancers with paired targets for {hotspot_label}")
        return

    def _status_for_class(site_sub, direction_short, _cls=cls):  # noqa: ANN001
        if _cls == "known_containing":
            return "known_positive"
        return "new_predicted"

    orig_status = pipeline.site_label_status
    pipeline.site_label_status = _status_for_class  # type: ignore[method-assign]
    try:
        plot_site_across_cancers(
            df_points=points,
            df_stats=stats,
            cancer_types=cancers,
            site=site,
            direction_short=DIRECTION_SHORT,
            target_regulation=TARGET_REGULATION,
            site_label=hotspot_label,
            output_dir=panel_dir,
            output_prefix=(
                f"{hotspot_label}_{DIRECTION_SHORT}_{TARGET_REGULATION}_across_cancers_high_low_hotspot_activity_boxplot"
            ),
            dpi=pipeline.config.dpi,
            pipeline=pipeline,
            significant_only=False,
        )
        print(
            f"{panel_name}: {hotspot_label} ({cls}) across {len(cancers)} cancers → {panel_dir}"
        )
    finally:
        pipeline.site_label_status = orig_status  # type: ignore[method-assign]


def _multi_cancer_significant_hotspots(
    stats: pd.DataFrame,
    pipeline: Optional[TargetRegulationBoxplotPipeline] = None,
    *,
    min_sig_cancers: int = 2,
) -> pd.DataFrame:
    """Hotspots with BH-significant association in ≥N cancers."""
    if stats.empty:
        return pd.DataFrame()
    act = stats[
        stats["direction_short"].astype(str).eq(DIRECTION_SHORT)
        & stats["target_regulation"].astype(str).eq(TARGET_REGULATION)
    ].copy()
    if act.empty or "wilcoxon_q_bh" not in act.columns:
        return pd.DataFrame()
    act["wilcoxon_q_bh"] = pd.to_numeric(act["wilcoxon_q_bh"], errors="coerce")
    if pipeline is not None:
        sig = _bh_sig_for_plot(act, pipeline)
    else:
        sig = act[act["wilcoxon_q_bh"] < 0.05].copy()
    if sig.empty or "hotspot_id" not in sig.columns:
        return pd.DataFrame()
    rows = []
    for hotspot_id, sub in sig.groupby("hotspot_id", sort=False):
        cancers = sorted(sub["cancer_type"].astype(str).unique())
        if len(cancers) < min_sig_cancers:
            continue
        site = str(sub["site"].dropna().iloc[0]) if "site" in sub.columns and sub["site"].notna().any() else ""
        label = (
            str(sub["hotspot_label"].dropna().iloc[0])
            if "hotspot_label" in sub.columns and sub["hotspot_label"].notna().any()
            else str(hotspot_id)
        )
        tf = str(sub["tf_name"].dropna().iloc[0]) if "tf_name" in sub.columns else ""
        cls = (
            str(sub["hotspot_class"].dropna().iloc[0])
            if "hotspot_class" in sub.columns and sub["hotspot_class"].notna().any()
            else ""
        )
        delta = pd.to_numeric(sub.get("delta_high_minus_low"), errors="coerce")
        rows.append(
            {
                "hotspot_id": str(hotspot_id),
                "hotspot_label": label,
                "site": site,
                "tf_name": tf,
                "hotspot_class": cls,
                "n_sig_cancers": len(cancers),
                "sig_cancers": ",".join(cancers),
                "median_delta_sig": float(delta.median()) if delta.notna().any() else np.nan,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["n_sig_cancers", "median_delta_sig"], ascending=[False, False]
    ).reset_index(drop=True)


def plot_multi_cancer_significant_hotspots(
    points: pd.DataFrame,
    stats: pd.DataFrame,
    output_dir: Path,
    pipeline: TargetRegulationBoxplotPipeline,
    *,
    min_sig_cancers: int = 2,
) -> None:
    """Across-cancer boxplots for hotspots significant in multiple cancers.

    Uses the same drawing style as plot_phosphosite_across_cancers.py, but with
    hotspot activity rows (site = ENSG|HS_...).
    """
    panel_dir = output_dir / "multi_cancer_significant_hotspots"
    panel_dir.mkdir(parents=True, exist_ok=True)
    catalog = _multi_cancer_significant_hotspots(
        stats, pipeline, min_sig_cancers=min_sig_cancers
    )
    catalog.to_csv(panel_dir / "multi_cancer_significant_hotspots_manifest.csv", index=False)
    if catalog.empty:
        print(f"multi-cancer hotspots skipped: none with ≥{min_sig_cancers} BH-sig cancers")
        return

    # Title color: known_containing → known_positive orange; else teal (new_predicted).
    orig_status = pipeline.site_label_status

    for _, row in catalog.iterrows():
        site = str(row["site"])
        label = str(row["hotspot_label"])
        cls = str(row.get("hotspot_class", ""))
        if not site or site == "nan":
            # Resolve site from points if stats lacked it
            hid = str(row["hotspot_id"])
            hit = points[points["hotspot_id"].astype(str).eq(hid)] if "hotspot_id" in points.columns else pd.DataFrame()
            if hit.empty or "site" not in hit.columns:
                print(f"  skip {label}: no site key")
                continue
            site = str(hit["site"].dropna().iloc[0])

        cancers = detect_cancers_with_data(
            df_points=points,
            site=site,
            pipeline=pipeline,
            cancer_order=list(pipeline.config.cancer_types),
        )
        if not cancers:
            print(f"  skip {label}: no cancers with paired targets")
            continue

        def _status_for_class(site_sub, direction_short, _cls=cls):  # noqa: ANN001
            if _cls == "known_containing":
                return "known_positive"
            return "new_predicted"

        pipeline.site_label_status = _status_for_class  # type: ignore[method-assign]
        try:
            plot_site_across_cancers(
                df_points=points,
                df_stats=stats,
                cancer_types=cancers,
                site=site,
                direction_short=DIRECTION_SHORT,
                target_regulation=TARGET_REGULATION,
                site_label=label,
                output_dir=panel_dir,
                output_prefix=(
                    f"{label}_{DIRECTION_SHORT}_{TARGET_REGULATION}_across_cancers_high_low_hotspot_activity_boxplot"
                ),
                dpi=pipeline.config.dpi,
                pipeline=pipeline,
                significant_only=False,
            )
            print(
                f"  plotted {label} ({cls}): {row['n_sig_cancers']} sig / "
                f"{len(cancers)} cancers with data [{row['sig_cancers']}]"
            )
        except Exception as exc:
            print(f"  failed {label}: {exc}")
        finally:
            pipeline.site_label_status = orig_status  # type: ignore[method-assign]

    print(f"multi-cancer hotspot plots → {panel_dir}")


def plot_4e_hsf1(points, stats, output_dir, pipeline) -> None:
    # Prefer nuclear HSF1 hotspots that cover the S320–S326 regulatory region when present.
    preferred_ids = [
        "Q00613_H7_S320_S326",
        "Q00613_H7_S303_S333",
        "Q00613_H4_S230_S230",
        "Q00613_H5_S244_S244",
        "Q00613_H3_S216_S218",
        "Q00613_H2_S230_S244",
    ]
    chosen = _select_hotspot_id(
        points,
        stats,
        gene_name="HSF1",
        preferred_ids=preferred_ids,
    )
    if not chosen:
        print("4e skipped: no HSF1 hotspot")
        return
    try:
        _plot_hotspot_across_cancers_like_multi(
            points,
            stats,
            gene_name="HSF1",
            hotspot_id=chosen,
            panel_name="figure4e_HSF1_hotspot",
            output_dir=output_dir,
            pipeline=pipeline,
        )
    except ValueError as exc:
        print(f"4e skipped: {exc}")


def plot_4f_irf(points, stats, output_dir, pipeline) -> None:
    preferred_ids = [
        "Q00978_H1_S131_S139",
        "Q00978_H1_S131_S140",
    ]
    chosen = _select_hotspot_id(
        points,
        stats,
        gene_name="IRF9",
        preferred_ids=preferred_ids,
        class_filter="known_independent",
    )
    if not chosen:
        print("4f skipped: no IRF9 known_independent hotspot")
        return
    try:
        _plot_hotspot_across_cancers_like_multi(
            points,
            stats,
            gene_name="IRF9",
            hotspot_id=chosen,
            panel_name="figure4f_IRF_known_independent_hotspot",
            output_dir=output_dir,
            pipeline=pipeline,
            class_filter="known_independent",
        )
    except ValueError as exc:
        print(f"4f skipped: {exc}")


def export_merged_activate_hotspot_panel(results_dir: Path, output_dir: Path) -> None:
    """Copy pipeline merged-all-hotspots across-cancer figure + tables into figure4/.

    Source is produced by HotspotTargetRegulationPipeline.plot_all_merged_boxplots()
    (same logic as unit-site Import_activate_merged_all_cancers_*.pdf).
    """
    src_dir = results_dir / "merged_all_sites_boxplots"
    panel_dir = output_dir / "merged_all_hotspots_across_cancers"
    panel_dir.mkdir(parents=True, exist_ok=True)
    if not src_dir.exists():
        print("merged hotspot panel skipped: missing merged_all_sites_boxplots/")
        return

    stems = [
        f"{DIRECTION_SHORT}_{TARGET_REGULATION}_merged_all_cancers_high_low_phospho_target_expression_boxplot",
        f"{DIRECTION_SHORT.lower()}_{TARGET_REGULATION}_merged_across_cancers_plot_summary",
        "merged_high_low_comparison_by_cancer_direction_regulation",
        "merged_high_low_expression_pairs",
    ]
    copied = 0
    for stem in stems:
        for path in src_dir.glob(f"{stem}.*"):
            dest = panel_dir / path.name
            shutil.copy2(path, dest)
            copied += 1
    # Friendly alias without the legacy "phospho" wording in the filename
    for ext in ("pdf", "png", "svg"):
        src = src_dir / f"{DIRECTION_SHORT}_{TARGET_REGULATION}_merged_all_cancers_high_low_phospho_target_expression_boxplot.{ext}"
        if src.exists():
            shutil.copy2(
                src,
                panel_dir / f"{DIRECTION_SHORT}_{TARGET_REGULATION}_merged_all_cancers_high_low_hotspot_activity_boxplot.{ext}",
            )
            copied += 1
    print(f"merged hotspot panel → {panel_dir} ({copied} files)")


def export_concordance_panel(concordance: pd.DataFrame, output_dir: Path) -> None:
    panel_dir = output_dir / "activity_concordance_supplement"
    panel_dir.mkdir(parents=True, exist_ok=True)
    if concordance.empty:
        print("Concordance supplement skipped: empty table")
        return
    concordance.to_csv(panel_dir / "significant_hotspot_activity_concordance.csv", index=False)
    summary = {
        "n": int(len(concordance)),
        "max_same_direction_frac": float(
            pd.to_numeric(concordance.get("max_same_direction"), errors="coerce").mean()
        ),
        "direction_weighted_same_direction_frac": float(
            pd.to_numeric(concordance.get("direction_weighted_same_direction"), errors="coerce").mean()
        ),
    }
    (panel_dir / "concordance_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Simple bar
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    labels = ["max vs mean_z", "dir-weighted vs mean_z"]
    vals = [summary["max_same_direction_frac"], summary["direction_weighted_same_direction_frac"]]
    ax.bar([0, 1], vals, color=["#5E8FA1", "#2F6F8F"], edgecolor="#333333", width=0.55)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Fraction same effect direction")
    ax.set_title("Activity score concordance\n(BH-significant hotspots)")
    sns.despine(ax=ax)
    _draw_y_axis_tick_marks(ax)
    _save_fig(fig, panel_dir / "Figure4_activity_concordance_bars")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--direction-short",
        choices=["Import", "Export"],
        default="Import",
        help="Which direction_short rows to plot (must match CPTAC scan).",
    )
    parser.add_argument(
        "--target-regulation",
        choices=["activate", "repress"],
        default="activate",
        help="Which target_regulation rows to plot (activate or repress).",
    )
    parser.add_argument(
        "--require-expected-direction",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Keep only BH-significant sites whose High−Low delta matches the transport "
            "hypothesis. Default: on for two-sided runs, off for directional."
        ),
    )
    args = parser.parse_args()

    global DIRECTION_SHORT, TARGET_REGULATION
    DIRECTION_SHORT = args.direction_short
    TARGET_REGULATION = args.target_regulation

    results_dir = args.results_dir
    output_dir = args.output_dir or (results_dir / "figure4")
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = _load_tables(results_dir)
    config = TempoConfig(
        output_dir=str(results_dir),
        use_bh_pvalue_correction=True,
        dpi=args.dpi,
    )
    # Prefer cancer order / test settings from run_config if present
    run_cfg = results_dir / "run_config.json"
    if run_cfg.exists():
        meta = json.loads(run_cfg.read_text(encoding="utf-8"))
        if meta.get("cancer_types"):
            config.cancer_types = list(meta["cancer_types"])
        if meta.get("test_alternative"):
            config.test_alternative = str(meta["test_alternative"])
    if args.require_expected_direction is None:
        config.require_expected_direction = (
            str(config.test_alternative).lower() == "two-sided"
        )
    else:
        config.require_expected_direction = bool(args.require_expected_direction)
    pipeline = TargetRegulationBoxplotPipeline(config)
    print(
        f"test_alternative={config.test_alternative} | "
        f"require_expected_direction={config.require_expected_direction}"
    )

    print(f"Figure 4 output: {output_dir}")
    plot_4b_nuclear_hotspot_boxplots(tables["points"], tables["stats_all"], output_dir, pipeline)
    plot_4c_observed_vs_random(results_dir, output_dir)
    plot_4d_stratification(
        tables["points"], tables["stats_all"], output_dir, pipeline, dpi=args.dpi
    )
    plot_measured_members_stratification(
        tables["points"], tables["stats_all"], output_dir, pipeline, dpi=args.dpi
    )
    plot_n_members_stratification(
        tables["points"], tables["stats_all"], output_dir, pipeline, dpi=args.dpi
    )
    plot_4e_hsf1(tables["points"], tables["stats_all"], output_dir, pipeline)
    plot_4f_irf(tables["points"], tables["stats_all"], output_dir, pipeline)
    try:
        plot_multi_cancer_significant_hotspots(
            tables["points"], tables["stats_all"], output_dir, pipeline, min_sig_cancers=2
        )
    except ValueError as exc:
        print(f"multi-cancer panel skipped: {exc}")
    export_merged_activate_hotspot_panel(results_dir, output_dir)
    export_concordance_panel(tables["concordance"], output_dir)
    print("Figure 4 panels complete.")


if __name__ == "__main__":
    main()
