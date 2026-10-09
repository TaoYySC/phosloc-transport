#!/usr/bin/env python3
"""
STAT3_S701 survival figure panels (separate files):

  A) KM — match km_curves/KM_STAT3_S701_CCRCC_LSCC_combined style
  B) Forest — minimal annotations + legend
  C) Boxplot — reuse plot_phosphosite_across_cancers.py style
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib.lines import Line2D

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from plot_phosphosite_across_cancers import (  # noqa: E402
    DEFAULT_POINTS_CSV,
    DEFAULT_STATS_CSV,
    detect_cancers_with_data,
    plot_site_across_cancers,
    resolve_site,
)
from run_import_target_regulation_analysis import (  # noqa: E402
    TargetRegulationBoxplotPipeline,
    TempoConfig,
)

ROOT = Path(__file__).resolve().parent.parent
V4 = ROOT / "results/survival_analysis/phosphosite_cox_3tier_v4"
OUT = V4 / "stat3_s701_figure_panels"

SITE_LABEL = "STAT3_S701"
SITE = "ENSG00000168610|S701"
# Match previous KM / boxplot palette
COLOR_LOW = "#8FAFBC"
COLOR_HIGH = "#E3A07A"


def _stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=350, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] {stem.with_suffix('.pdf')}")


def plot_a_km(sample_df: pd.DataFrame, cox: pd.DataFrame) -> None:
    """Match KM_STAT3_S701_CCRCC_LSCC_combined layout/colors."""
    cancers = ["CCRCC", "LSCC"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5), sharey=True)
    fig.subplots_adjust(wspace=0.22, left=0.08, right=0.98, top=0.78, bottom=0.16)

    for ax, cancer in zip(axes, cancers):
        sub = sample_df.loc[
            sample_df["cancer_type"].eq(cancer) & sample_df["site"].eq(SITE)
        ].copy()
        sub = sub.dropna(subset=["OS_days", "OS_event", "phospho_group"])
        low = sub.loc[sub["phospho_group"].eq("low")]
        high = sub.loc[sub["phospho_group"].eq("high")]
        lr = logrank_test(
            low["OS_days"],
            high["OS_days"],
            event_observed_A=low["OS_event"],
            event_observed_B=high["OS_event"],
        )
        row = cox.loc[
            cox["cancer_type"].eq(cancer)
            & cox["site_label"].eq(SITE_LABEL)
            & cox["model"].eq("univariate")
            & cox["status"].eq("success")
        ].iloc[0]
        hr = float(row["HR_phospho"])
        star = _stars(float(lr.p_value))

        tmax = float(sub["OS_days"].max())
        for group, color, lab in [
            ("low", COLOR_LOW, "Low phospho"),
            ("high", COLOR_HIGH, "High phospho"),
        ]:
            g = sub.loc[sub["phospho_group"].eq(group)]
            kmf = KaplanMeierFitter()
            kmf.fit(g["OS_days"], event_observed=g["OS_event"], label=f"{lab} (n={len(g)})")
            kmf.plot_survival_function(
                ax=ax, color=color, ci_show=True, ci_alpha=0.14, linewidth=2.3
            )

        ax.set_ylim(0, 1.02)
        ax.set_xlim(0, tmax * 1.02)
        ax.set_xlabel("Days")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.35)
        ax.set_axisbelow(True)

        # title block matching previous combined KM
        ax.text(
            0.5,
            1.16,
            f"{cancer} · STAT3 S701",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12.5,
            color="#222222",
        )
        ax.text(
            0.5,
            1.05,
            f"HR = {hr:.2f} {star}".rstrip(),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            color="#333333",
        )

        handles, labels = ax.get_legend_handles_labels()
        keep, seen = [], set()
        for h, lab in zip(handles, labels):
            key = lab.split(" (n=")[0]
            if key in {"Low phospho", "High phospho"} and key not in seen:
                keep.append((h, lab))
                seen.add(key)
        ax.legend(
            [k[0] for k in keep],
            [k[1] for k in keep],
            loc="lower left",
            frameon=False,
            fontsize=9,
        )

        if ax is axes[0]:
            ax.set_ylabel("Survival probability")

    _save(fig, OUT / "A_KM_STAT3_S701_CCRCC_LSCC")


def plot_b_forest(cox: pd.DataFrame) -> None:
    """Clean forest: HR+CI only; categories in legend."""
    m1 = cox.loc[
        cox["site_label"].eq(SITE_LABEL) & cox["model"].eq("univariate")
    ].copy()
    prefer = ["CCRCC", "LSCC", "GBM", "UCEC", "OV", "BRCA"]
    m1["ord"] = m1["cancer_type"].map({c: i for i, c in enumerate(prefer)}).fillna(99)
    m1 = m1.sort_values(["ord", "cancer_type"]).reset_index(drop=True)

    n = len(m1)
    fig, ax = plt.subplots(figsize=(6.2, max(3.0, 0.52 * n + 1.0)))

    y = np.arange(n)[::-1]
    yticklabels = []
    color_sig = "#C0392B"
    color_ns = "#555555"
    color_skip = "#B0B0B0"

    for i, (_, r) in enumerate(m1.iterrows()):
        yi = y[i]
        cancer = str(r["cancer_type"])
        endpoint = str(r["endpoint"])
        yticklabels.append(f"{cancer} ({endpoint})")
        status = str(r["status"])

        if status != "success" or not np.isfinite(float(r.get("HR_phospho", np.nan))):
            ax.plot(1.0, yi, marker="o", color=color_skip, markersize=6, zorder=3)
            continue

        hr = float(r["HR_phospho"])
        lo = max(float(r["CI_low"]), 0.05)
        hi = min(float(r["CI_high"]), 40.0)
        q = float(r["q_bh"]) if np.isfinite(float(r.get("q_bh", np.nan))) else np.nan
        is_sig = np.isfinite(q) and q <= 0.05
        color = color_sig if is_sig else color_ns
        ax.hlines(yi, lo, hi, color=color, linewidth=1.9, zorder=2)
        ax.plot(hr, yi, "o", color=color, markersize=7.5, zorder=3)
        # only mark BH hits with a star next to the point (minimal ink)
        if is_sig:
            ax.text(hi * 1.08, yi, "*", va="center", ha="left", fontsize=12, color=color_sig)

    ax.axvline(1.0, color="#999999", linestyle="--", linewidth=1.0, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(yticklabels)
    ax.set_xscale("log")
    ax.set_xlim(0.08, 30)
    ax.set_xticks([0.1, 0.25, 0.5, 1, 2, 4, 10, 20])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel("Hazard ratio (High vs Low phospho)")
    ax.set_title("STAT3 S701 · univariate Cox", fontsize=12.5, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.4)

    legend_handles = [
        Line2D([0], [0], marker="o", color=color_sig, linestyle="-", linewidth=1.8,
               markersize=7, label="BH q ≤ 0.05"),
        Line2D([0], [0], marker="o", color=color_ns, linestyle="-", linewidth=1.8,
               markersize=7, label="Not significant"),
        Line2D([0], [0], marker="o", color=color_skip, linestyle="None",
               markersize=6, label="Insufficient events"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="lower right", fontsize=9)

    fig.tight_layout()
    _save(fig, OUT / "B_forest_STAT3_S701_across_cancers")


def plot_c_boxplot() -> None:
    """Reuse plot_phosphosite_across_cancers drawing style (target expression)."""
    from plot_phosphosite_across_cancers import SITE_PLOT_OVERRIDES

    # Align low/high colors with KM / main CPTAC palette while keeping that script's layout.
    SITE_PLOT_OVERRIDES["STAT3_S701"] = {
        "group_color_map": {"low": COLOR_LOW, "high": COLOR_HIGH},
    }

    points_path = Path(DEFAULT_POINTS_CSV)
    stats_path = Path(DEFAULT_STATS_CSV)
    df_points = pd.read_csv(points_path, low_memory=False)
    df_stats = pd.read_csv(stats_path) if stats_path.exists() else pd.DataFrame()
    pipeline = TargetRegulationBoxplotPipeline(TempoConfig())
    site, site_label = resolve_site(SITE_LABEL, SITE)
    cancer_types = detect_cancers_with_data(
        df_points=df_points, site=site, pipeline=pipeline
    )
    prefer = ["CCRCC", "LSCC", "GBM", "UCEC", "OV", "BRCA", "HNSCC", "LUAD", "PDAC", "COAD"]
    ordered = [c for c in prefer if c in cancer_types]
    ordered += [c for c in cancer_types if c not in ordered]

    out_path = plot_site_across_cancers(
        df_points=df_points,
        df_stats=df_stats,
        cancer_types=ordered,
        site=site,
        site_label=site_label,
        output_dir=OUT,
        output_prefix="C_phospho_boxplot_STAT3_S701_by_cancer",
        dpi=350,
        pipeline=pipeline,
        significant_only=False,
    )
    print(f"[OK] {out_path}")


def main() -> None:
    _setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    sample_df = pd.read_csv(V4 / "sample_phospho_groups.csv")
    cox = pd.read_csv(V4 / "cox_results_bh_fdr.csv")
    plot_a_km(sample_df, cox)
    plot_b_forest(cox)
    plot_c_boxplot()
    print(f"[DONE] {OUT}")


if __name__ == "__main__":
    main()
