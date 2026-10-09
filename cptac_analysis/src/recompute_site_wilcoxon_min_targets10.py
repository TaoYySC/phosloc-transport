#!/usr/bin/env python3
"""Recompute site-level two-sided Wilcoxon with n_targets >= 10 filter.

Rules:
  - n < 10: not_evaluable; no P / BH q
  - n >= 10: two-sided Wilcoxon signed-rank test
  - BH correction within each cancer, only over evaluable tests
"""

from __future__ import annotations

from pathlib import Path

# Repo-relative results root (cptac_analysis/results)
RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

TWO_ROOT = RESULTS_ROOT / "hotspot_mixed_pred_filter_pure_regulon_only_twosided_20260929"
ARMS = [
    ("Import_activate", "activate"),
    ("Import_repress", "repress"),
    ("Export_activate", "activate"),
    ("Export_repress", "repress"),
]
MIN_N = 10


def _bh_adjust(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce")
    q = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.notna()
    if valid.sum() == 0:
        return q
    p_valid = p.loc[valid].astype(float)
    order = np.argsort(p_valid.to_numpy())
    ranked_p = p_valid.to_numpy()[order]
    m = len(ranked_p)
    adjusted = ranked_p * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    q.loc[p_valid.index[order]] = adjusted
    return q


def _p_to_stars(p_value: float) -> str:
    if pd.isna(p_value):
        return "ns"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def _join_vals(series: pd.Series, ndigits: int = 6) -> str:
    out: List[str] = []
    for v in series:
        if pd.isna(v):
            out.append("")
        else:
            out.append(f"{float(v):.{ndigits}g}")
    return ",".join(out)


def recompute_arm(arm: str, regulation: str) -> pd.DataFrame:
    arm_dir = TWO_ROOT / arm
    points_path = (
        arm_dir / "cptac" / "high_low_phospho_boxplots" / "target_gene_high_low_expression_points_all.csv"
    )
    old_summary_path = (
        arm_dir / "cptac_filtered" / f"two_side_high_low_phospho_comparison_by_site_{arm.lower()}.csv"
    )
    points = pd.read_csv(points_path, low_memory=False)
    old = pd.read_csv(old_summary_path, low_memory=False)

    pts = points[points["target_regulation"].astype(str).str.lower().eq(regulation)].copy()
    # Restrict to sites already in the filtered summary (same universe).
    key = ["cancer_type", "site", "site_label"]
    pts = pts.merge(old[key].drop_duplicates(), on=key, how="inner")

    rows = []
    group_cols = ["cancer_type", "direction_short", "target_regulation", "site", "site_label", "tf_name"]
    for keys, sub in pts.groupby(group_cols, sort=False):
        cancer_type, direction_short, target_regulation, site, site_label, tf_name = keys
        present = set(sub["phospho_group"].astype(str))
        if "low" not in present or "high" not in present:
            continue
        wide = (
            sub.pivot_table(
                index="target_gene_id",
                columns="phospho_group",
                values="group_mean_expression",
                aggfunc="mean",
            )
            .dropna(subset=["low", "high"], how="any")
        )
        n_targets = int(len(wide))
        low = pd.to_numeric(wide["low"], errors="coerce") if n_targets else pd.Series(dtype=float)
        high = pd.to_numeric(wide["high"], errors="coerce") if n_targets else pd.Series(dtype=float)
        mean_low = float(low.mean()) if n_targets else np.nan
        mean_high = float(high.mean()) if n_targets else np.nan
        delta = float(mean_high - mean_low) if n_targets else np.nan

        evaluable = n_targets >= MIN_N
        p_value = np.nan
        if evaluable and not np.allclose((high - low).fillna(0).to_numpy(), 0):
            try:
                p_value = float(wilcoxon(high, low, alternative="two-sided").pvalue)
            except ValueError:
                p_value = np.nan
                evaluable = False

        rows.append(
            {
                "cancer_type": cancer_type,
                "direction_short": direction_short,
                "target_regulation": target_regulation,
                "site": site,
                "site_label": site_label,
                "tf_name": tf_name,
                "n_paired_target_genes": n_targets,
                "evaluable": evaluable,
                "evaluability_status": "evaluable" if evaluable else "not_evaluable",
                "mean_low_phospho_expression": mean_low,
                "mean_high_phospho_expression": mean_high,
                "delta_high_minus_low": delta,
                "expected_direction": "High vs Low",
                "alternative": "two-sided" if evaluable else "not_evaluable",
                "wilcoxon_p_expected": p_value,
                "significance": _p_to_stars(p_value) if evaluable else "not_evaluable",
            }
        )

    stats = pd.DataFrame(rows)
    # Attach hotspot metadata from previous summary when available.
    meta_cols = [
        c
        for c in [
            "cancer_type",
            "site",
            "site_label",
            "hotspot_id",
            "hotspot_label",
            "hotspot_class",
            "n_measured_members",
            "n_members",
        ]
        if c in old.columns
    ]
    stats = stats.merge(old[meta_cols].drop_duplicates(["cancer_type", "site", "site_label"]), on=["cancer_type", "site", "site_label"], how="left")

    # raw p copy + BH within cancer among evaluable tests only (NaN p skipped)
    stats["wilcoxon_p_raw"] = stats["wilcoxon_p_expected"]
    stats["significance_raw"] = stats["significance"]
    stats["wilcoxon_q_bh"] = np.nan
    for _, idx in stats.groupby("cancer_type", dropna=False).groups.items():
        stats.loc[idx, "wilcoxon_q_bh"] = _bh_adjust(stats.loc[idx, "wilcoxon_p_raw"])
    not_eval = ~stats["evaluable"]
    stats.loc[not_eval, "wilcoxon_p_expected"] = np.nan
    stats.loc[not_eval, "wilcoxon_p_raw"] = np.nan
    stats.loc[not_eval, "wilcoxon_q_bh"] = np.nan
    stats.loc[not_eval, "significance_raw"] = "not_evaluable"
    stats["significance_bh"] = stats["wilcoxon_q_bh"].map(_p_to_stars)
    stats.loc[not_eval, "significance_bh"] = "not_evaluable"
    stats["significance"] = stats["significance_bh"]
    stats["wilcoxon_p_for_plot"] = stats["wilcoxon_q_bh"]
    stats.loc[not_eval, "wilcoxon_p_for_plot"] = np.nan

    # Column order close to previous summary
    preferred = [
        "cancer_type",
        "direction_short",
        "target_regulation",
        "site",
        "site_label",
        "tf_name",
        "n_paired_target_genes",
        "evaluable",
        "evaluability_status",
        "mean_low_phospho_expression",
        "mean_high_phospho_expression",
        "delta_high_minus_low",
        "expected_direction",
        "alternative",
        "wilcoxon_p_expected",
        "significance",
        "wilcoxon_p_raw",
        "significance_raw",
        "wilcoxon_q_bh",
        "significance_bh",
        "wilcoxon_p_for_plot",
        "hotspot_id",
        "hotspot_label",
        "hotspot_class",
        "n_measured_members",
        "n_members",
    ]
    cols = [c for c in preferred if c in stats.columns] + [c for c in stats.columns if c not in preferred]
    stats = stats[cols]

    out_summary = arm_dir / "cptac_filtered" / f"two_side_high_low_phospho_comparison_by_site_{arm.lower()}.csv"
    stats.to_csv(out_summary, index=False)

    # Rebuild comma-separated target value tables
    detail_path = (
        arm_dir / "cptac_filtered" / f"two_side_target_gene_high_low_expression_by_site_{arm.lower()}.csv"
    )
    if detail_path.exists():
        detail = pd.read_csv(detail_path, low_memory=False)
    else:
        # rebuild detail from points
        low = pts[pts["phospho_group"].astype(str).eq("low")].copy()
        high = pts[pts["phospho_group"].astype(str).eq("high")].copy()
        meta = [
            c
            for c in [
                "cancer_type",
                "direction_short",
                "site",
                "site_label",
                "tf_name",
                "target_gene_id",
                "target_gene_name",
                "target_regulation",
                "hotspot_id",
                "hotspot_label",
                "hotspot_class",
            ]
            if c in low.columns
        ]
        left = low[meta + ["group_mean_expression"]].rename(
            columns={"group_mean_expression": "low_group_mean_expression"}
        )
        right = high[["cancer_type", "site", "target_gene_id", "group_mean_expression"]].rename(
            columns={"group_mean_expression": "high_group_mean_expression"}
        )
        detail = left.merge(right, on=["cancer_type", "site", "target_gene_id"], how="inner")
        detail["delta_high_minus_low"] = (
            pd.to_numeric(detail["high_group_mean_expression"], errors="coerce")
            - pd.to_numeric(detail["low_group_mean_expression"], errors="coerce")
        )

    # refresh site-level annotations onto long detail
    attach = stats[
        [
            "cancer_type",
            "site",
            "site_label",
            "n_paired_target_genes",
            "evaluable",
            "evaluability_status",
            "mean_low_phospho_expression",
            "mean_high_phospho_expression",
            "delta_high_minus_low",
            "wilcoxon_p_raw",
            "wilcoxon_q_bh",
            "significance_bh",
            "alternative",
        ]
    ].rename(
        columns={
            "mean_low_phospho_expression": "site_mean_low_across_targets",
            "mean_high_phospho_expression": "site_mean_high_across_targets",
            "delta_high_minus_low": "site_delta_high_minus_low",
        }
    )
    # drop old attached cols if present
    drop_old = [
        c
        for c in detail.columns
        if c
        in {
            "n_paired_target_genes",
            "evaluable",
            "evaluability_status",
            "site_mean_low_across_targets",
            "site_mean_high_across_targets",
            "site_delta_high_minus_low",
            "wilcoxon_p_raw",
            "wilcoxon_q_bh",
            "significance_bh",
            "alternative",
            "mean_low_phospho_expression",
            "mean_high_phospho_expression",
            "site_hotspot_class",
        }
    ]
    detail2 = detail.drop(columns=drop_old, errors="ignore").merge(
        attach, on=["cancer_type", "site", "site_label"], how="left"
    )
    detail2.to_csv(detail_path, index=False)

    # comma lists
    d = detail2.copy()
    d["_abs"] = pd.to_numeric(d["delta_high_minus_low"], errors="coerce").abs()
    d = d.sort_values(
        ["cancer_type", "site", "site_label", "_abs", "target_gene_name"],
        ascending=[True, True, True, False, True],
    )
    agg_rows = []
    gcols = ["cancer_type", "direction_short", "target_regulation", "site", "site_label", "tf_name"]
    for keys, g in d.groupby(gcols, sort=False):
        agg_rows.append(
            {
                "cancer_type": keys[0],
                "direction_short": keys[1],
                "target_regulation": keys[2],
                "site": keys[3],
                "site_label": keys[4],
                "tf_name": keys[5],
                "n_targets_listed": len(g),
                "target_gene_names": ",".join(g["target_gene_name"].astype(str).tolist()),
                "target_gene_ids": ",".join(g["target_gene_id"].astype(str).tolist()),
                "low_group_mean_expression_values": _join_vals(g["low_group_mean_expression"]),
                "high_group_mean_expression_values": _join_vals(g["high_group_mean_expression"]),
                "delta_high_minus_low_values": _join_vals(g["delta_high_minus_low"]),
            }
        )
    agg = pd.DataFrame(agg_rows)
    with_vals = stats.merge(agg, on=gcols, how="left")
    with_path = (
        arm_dir
        / "cptac_filtered"
        / f"two_side_high_low_phospho_comparison_by_site_with_target_values_{arm.lower()}.csv"
    )
    with_vals.to_csv(with_path, index=False)

    compact_cols = [
        "cancer_type",
        "direction_short",
        "target_regulation",
        "site_label",
        "tf_name",
        "hotspot_class",
        "n_paired_target_genes",
        "evaluable",
        "evaluability_status",
        "mean_low_phospho_expression",
        "mean_high_phospho_expression",
        "delta_high_minus_low",
        "wilcoxon_p_raw",
        "wilcoxon_q_bh",
        "significance_bh",
        "target_gene_names",
        "low_group_mean_expression_values",
        "high_group_mean_expression_values",
        "delta_high_minus_low_values",
    ]
    compact_cols = [c for c in compact_cols if c in with_vals.columns]
    compact_path = (
        arm_dir / "cptac_filtered" / f"two_side_site_summary_target_values_comma_{arm.lower()}.csv"
    )
    with_vals[compact_cols].to_csv(compact_path, index=False)

    n_eval = int(stats["evaluable"].sum())
    n_ne = int((~stats["evaluable"]).sum())
    n_sig = int(stats["wilcoxon_q_bh"].lt(0.05).fillna(False).sum())
    print(
        f"{arm}: sites={len(stats)} evaluable(n>={MIN_N})={n_eval} "
        f"not_evaluable={n_ne} BH_sig={n_sig}"
    )
    return stats


def main() -> None:
    frames = []
    for arm, reg in ARMS:
        frames.append(recompute_arm(arm, reg))
    all_df = pd.concat(frames, ignore_index=True)
    summary = (
        all_df.groupby(["direction_short", "target_regulation"], as_index=False)
        .agg(
            n_sites=("site", "size"),
            n_evaluable=("evaluable", "sum"),
            n_not_evaluable=("evaluable", lambda s: int((~s).sum())),
            n_bh_sig=("wilcoxon_q_bh", lambda s: int(pd.to_numeric(s, errors="coerce").lt(0.05).fillna(False).sum())),
        )
    )
    out = TWO_ROOT / "site_wilcoxon_min10_filter_summary.csv"
    summary.to_csv(out, index=False)
    print("\nOverall:")
    print(summary.to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
