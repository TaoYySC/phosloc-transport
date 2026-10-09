#!/usr/bin/env python3
"""Combined one-sided ∪ two-sided significance heatmap (4 arms).

- Rows: cancers; columns: hotspots significant in ≥1 cancer (union of tests)
- Color: Δ(High−Low) target expression (from two-sided tables; identical means)
- Dots: BH-significant in one-sided OR two-sided (two-sided: any direction)
- Top bars: K = known_containing, P = predicted-only (known_independent/proximal)
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

ONE_ROOT = RESULTS_ROOT / "hotspot_mixed_pred_filter_pure_regulon_only_20260928"
TWO_ROOT = RESULTS_ROOT / "hotspot_mixed_pred_filter_pure_regulon_only_twosided_20260929"
OUT_DIR = RESULTS_ROOT / "hotspot_mixed_pred_filter_pure_regulon_only_combined_heatmap_20260929"

CANCER_ORDER = [
    "BRCA", "CCRCC", "COAD", "GBM", "HNSCC", "LSCC", "LUAD", "OV", "PDAC", "UCEC"
]
ARMS = [
    ("Import_activate", "Import", "activate", "Import + activated"),
    ("Import_repress", "Import", "repress", "Import + repressed"),
    ("Export_activate", "Export", "activate", "Export + activated"),
    ("Export_repress", "Export", "repress", "Export + repressed"),
]
K_COLOR = "#2B2B2B"
P_COLOR = "#E07A3D"


def _kp_class(hotspot_class: object) -> str:
    cls = str(hotspot_class).strip() if pd.notna(hotspot_class) else ""
    if cls == "known_containing":
        return "K"
    return "P"  # known_independent / known_proximal / other


def _load_arm(root: Path, arm: str, regulation: str) -> pd.DataFrame:
    # Two-sided runs use two_side_*_{direction}_{regulation}.csv; one-sided keep legacy name.
    candidates = [
        root
        / arm
        / "cptac_filtered"
        / f"two_side_high_low_phospho_comparison_by_site_{arm.lower()}.csv",
        root / arm / "cptac_filtered" / f"high_low_phospho_comparison_by_site_{regulation}.csv",
    ]
    path = next((p for p in candidates if p.exists()), candidates[-1])
    df = pd.read_csv(path, low_memory=False)
    df["q"] = pd.to_numeric(df["wilcoxon_q_bh"], errors="coerce")
    df["delta"] = pd.to_numeric(df["delta_high_minus_low"], errors="coerce")
    df["site_label"] = df["site_label"].astype(str)
    df["cancer_type"] = df["cancer_type"].astype(str)
    if "hotspot_class" not in df.columns:
        df["hotspot_class"] = ""
    df["kp"] = df["hotspot_class"].map(_kp_class)
    df["bh_sig"] = df["q"].lt(0.05)
    return df


def build_union_tables() -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Return long table of union-significant cells + per-arm matrices for plotting."""
    records = []
    arm_frames: Dict[str, pd.DataFrame] = {}

    for arm, direction, reg, title in ARMS:
        one = _load_arm(ONE_ROOT, arm, reg)
        two = _load_arm(TWO_ROOT, arm, reg)
        # merge on cancer×site
        key = ["cancer_type", "site", "site_label"]
        m = two.merge(
            one[key + ["q", "bh_sig"]].rename(columns={"q": "q_one", "bh_sig": "sig_one"}),
            on=key,
            how="outer",
            suffixes=("", "_one_dup"),
        )
        # deltas / class from two-sided preferentially
        class_map = (
            pd.concat([two.set_index("site_label")["kp"], one.set_index("site_label")["kp"]])
            .groupby(level=0)
            .first()
        )
        hc_map = (
            pd.concat(
                [two.set_index("site_label")["hotspot_class"], one.set_index("site_label")["hotspot_class"]]
            )
            .groupby(level=0)
            .first()
        )
        delta_map = two.set_index(["cancer_type", "site_label"])["delta"]
        # one-sided delta fill if missing
        delta_one = one.set_index(["cancer_type", "site_label"])["delta"]

        m["sig_two"] = m["bh_sig"].fillna(False)
        if "sig_one" not in m.columns:
            m["sig_one"] = False
        m["sig_one"] = m["sig_one"].fillna(False)
        m["sig_union"] = m["sig_one"] | m["sig_two"]

        def get_delta(row):
            k = (row["cancer_type"], row["site_label"])
            if k in delta_map.index and pd.notna(delta_map.loc[k]):
                val = delta_map.loc[k]
                return float(val.iloc[0]) if isinstance(val, pd.Series) else float(val)
            if k in delta_one.index and pd.notna(delta_one.loc[k]):
                val = delta_one.loc[k]
                return float(val.iloc[0]) if isinstance(val, pd.Series) else float(val)
            return np.nan

        m["delta"] = m.apply(get_delta, axis=1)
        m["kp"] = m["site_label"].map(class_map).fillna("P")
        m["hotspot_class"] = m["site_label"].map(hc_map).fillna("")
        m["arm"] = arm
        m["arm_title"] = title
        m["direction_short"] = direction
        m["target_regulation"] = reg

        # sites with ≥1 union-significant cancer
        sig_sites = set(m.loc[m["sig_union"], "site_label"])
        m_plot = m[m["site_label"].isin(sig_sites)].copy()
        arm_frames[arm] = m_plot

        for _, r in m_plot.iterrows():
            if not r["sig_union"] and pd.isna(r["delta"]):
                continue
            records.append(
                {
                    "arm": arm,
                    "arm_title": title,
                    "direction_short": direction,
                    "target_regulation": reg,
                    "cancer_type": r["cancer_type"],
                    "site": r.get("site", ""),
                    "site_label": r["site_label"],
                    "hotspot_class": r["hotspot_class"],
                    "kp": r["kp"],
                    "delta": r["delta"],
                    "sig_one_sided": bool(r["sig_one"]),
                    "sig_two_sided": bool(r["sig_two"]),
                    "sig_union": bool(r["sig_union"]),
                    "q_one": r.get("q_one", np.nan),
                    "q_two": r.get("q", np.nan),
                }
            )

    return pd.DataFrame(records), arm_frames


def _order_sites(df: pd.DataFrame) -> List[str]:
    """K then P; within each, by n_sig cancers desc then label."""
    g = (
        df.groupby(["site_label", "kp"], sort=False)
        .agg(n_sig=("sig_union", "sum"), med_abs=("delta", lambda s: float(np.nanmedian(np.abs(s)))))
        .reset_index()
    )
    g["kp_rank"] = g["kp"].map({"K": 0, "P": 1}).fillna(2)
    g = g.sort_values(["kp_rank", "n_sig", "med_abs", "site_label"], ascending=[True, False, False, True])
    return g["site_label"].tolist()


def plot_combined(arm_frames: Dict[str, pd.DataFrame], out_stem: Path, dpi: int = 300) -> None:
    panels = []
    for arm, _, _, title in ARMS:
        df = arm_frames.get(arm, pd.DataFrame())
        if df is None or df.empty or not df["sig_union"].any():
            panels.append((title, None, None, None, []))
            continue
        sites = _order_sites(df)
        cancers = [c for c in CANCER_ORDER if c in set(df["cancer_type"])]
        extra = sorted(set(df["cancer_type"]) - set(cancers))
        cancers = cancers + extra
        delta = df.pivot_table(index="cancer_type", columns="site_label", values="delta", aggfunc="first")
        sig = df.pivot_table(index="cancer_type", columns="site_label", values="sig_union", aggfunc="max")
        delta = delta.reindex(index=cancers, columns=sites)
        sig = sig.reindex(index=cancers, columns=sites).fillna(False)
        kp = (
            df.drop_duplicates("site_label")
            .set_index("site_label")["kp"]
            .reindex(sites)
            .fillna("P")
        )
        panels.append((title, delta, sig, kp, sites))

    n_sites = [0 if p[1] is None else p[1].shape[1] for p in panels]
    total = sum(max(n, 1) for n in n_sites)
    # width proportional to n sites
    fig_w = max(14.0, 0.42 * total + 4.0)
    fig_h = 7.2
    fig = plt.figure(figsize=(fig_w, fig_h))
    # leave room for colorbar
    width_ratios = [max(n, 1) for n in n_sites]
    gs = fig.add_gridspec(
        2,
        4,
        height_ratios=[0.12, 1.0],
        width_ratios=width_ratios,
        hspace=0.08,
        wspace=0.08,
        left=0.07,
        right=0.88,
        top=0.86,
        bottom=0.22,
    )

    # shared color scale
    all_vals = []
    for _, delta, _, _, _ in panels:
        if delta is not None:
            all_vals.append(delta.to_numpy(dtype=float).ravel())
    if all_vals:
        flat = np.concatenate(all_vals)
        flat = flat[np.isfinite(flat)]
        vmax = float(np.nanpercentile(np.abs(flat), 95)) if len(flat) else 0.2
    else:
        vmax = 0.2
    vmax = max(vmax, 0.05)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")

    heat_axes = []
    for i, (title, delta, sig, kp, sites) in enumerate(panels):
        ax_bar = fig.add_subplot(gs[0, i])
        ax = fig.add_subplot(gs[1, i])
        heat_axes.append(ax)
        ax_bar.set_xlim(0, max(len(sites), 1))
        ax_bar.set_ylim(0, 1)
        ax_bar.axis("off")
        ax_bar.set_title(title, fontsize=11, pad=2)

        if delta is None or len(sites) == 0:
            ax.text(0.5, 0.5, "no significant\nhotspots", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks(range(len(CANCER_ORDER)))
            if i == 0:
                ax.set_yticklabels(CANCER_ORDER, fontsize=8)
            else:
                ax.set_yticklabels([])
            continue

        # K/P bar
        for j, s in enumerate(sites):
            color = K_COLOR if kp.loc[s] == "K" else P_COLOR
            ax_bar.add_patch(
                mpatches.Rectangle((j, 0.15), 1.0, 0.7, facecolor=color, edgecolor="none")
            )
        # K / P labels spanning contiguous blocks
        for label, color in (("K", K_COLOR), ("P", P_COLOR)):
            idxs = [j for j, s in enumerate(sites) if kp.loc[s] == label]
            if not idxs:
                continue
            # contiguous runs
            run_start = idxs[0]
            prev = idxs[0]
            runs = []
            for j in idxs[1:]:
                if j == prev + 1:
                    prev = j
                else:
                    runs.append((run_start, prev))
                    run_start = prev = j
            runs.append((run_start, prev))
            for a, b in runs:
                ax_bar.text((a + b + 1) / 2, 1.05, label, ha="center", va="bottom", fontsize=9, fontweight="bold")

        mat = delta.to_numpy(dtype=float)
        im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        # dots for significance
        sig_mat = sig.to_numpy(dtype=bool)
        ys, xs = np.where(sig_mat)
        ax.scatter(xs, ys, s=12, c="black", marker="o", zorder=3, linewidths=0)

        ax.set_xticks(range(len(sites)))
        ax.set_xticklabels(sites, rotation=55, ha="right", fontsize=7)
        ax.set_yticks(range(len(delta.index)))
        if i == 0:
            ax.set_yticklabels(list(delta.index), fontsize=8)
            ax.set_ylabel("Cancer")
        else:
            ax.set_yticklabels([])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)

    # colorbar
    cax = fig.add_axes([0.90, 0.28, 0.015, 0.45])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Mean high−low adjusted\ntarget expression", fontsize=8)

    fig.suptitle(
        "One-sided ∪ two-sided hotspot-associated target-expression changes\n"
        "Color: mean high-minus-low target expression; dots: BH-significant cells "
        "(two-sided = any direction)\n"
        "K = Known-containing hotspot; P = Predicted-only hotspot",
        fontsize=11,
        y=0.98,
    )
    # bottom legend
    k_patch = mpatches.Patch(facecolor=K_COLOR, label="Known-containing hotspot")
    p_patch = mpatches.Patch(facecolor=P_COLOR, label="Predicted-only hotspot")
    fig.legend(
        handles=[k_patch, p_patch],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.48, 0.02),
    )

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{out_stem}.{ext}", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    long_df, arm_frames = build_union_tables()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long_path = OUT_DIR / "combined_one_two_sided_significance_long.csv"
    long_df.to_csv(long_path, index=False)

    # summary of columns kept
    summary_rows = []
    for arm, _, _, title in ARMS:
        df = arm_frames[arm]
        if df.empty:
            summary_rows.append({"arm": arm, "title": title, "n_sig_sites": 0, "n_sig_cells": 0})
            continue
        sites = df.loc[df["sig_union"], "site_label"].nunique()
        cells = int(df["sig_union"].sum())
        summary_rows.append({"arm": arm, "title": title, "n_sig_sites": sites, "n_sig_cells": cells})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "combined_heatmap_arm_summary.csv", index=False)
    print(summary.to_string(index=False))

    stem = OUT_DIR / "combined_one_two_sided_hotspot_cancer_heatmap"
    plot_combined(arm_frames, stem)
    print(f"Wrote {stem}.png (+pdf/svg)")
    print(f"Long table: {long_path}")


if __name__ == "__main__":
    main()
