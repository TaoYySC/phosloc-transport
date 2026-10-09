#!/usr/bin/env python3
"""Visualize hotspot high/low activity vs survival for all analyzable pairs.

Outputs under --surv-dir:
  km_curves_all/KM_*.{png,pdf}          per cancer×hotspot KM
  km_plot_manifest_all.csv
  forest_univariate_HR_all.{png,pdf}    M1 Cox HR forest (successful fits)
  km_small_multiples_overview.{png,pdf} compact grid of all KMs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from run_phosphosite_survival_cox import (  # noqa: E402
    KaplanMeierFitter,
    _safe_to_numeric,
    endpoint_columns,
    logrank_test,
    make_km_plot,
)

DEFAULT_SURV = (
    Path(__file__).resolve().parents[1]
    / "results/survival_analysis/hotspot_cox_v11_147pos_d3_platt_anchor_only_mean_z"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--surv-dir", type=Path, default=DEFAULT_SURV)
    p.add_argument(
        "--min-group-n",
        type=int,
        default=3,
        help="Minimum samples per high/low group to draw KM (default 3)",
    )
    return p.parse_args()


def _pair_table(surv_dir: Path) -> pd.DataFrame:
    depth = pd.read_csv(surv_dir / "event_adaptive_depth_manifest.csv")
    keep = [
        c
        for c in [
            "cancer_type",
            "site",
            "site_label",
            "endpoint",
            "n_events_m1",
            "analysis_depth",
            "hotspot_class",
            "n_measured_members",
        ]
        if c in depth.columns
    ]
    return depth[keep].drop_duplicates(["cancer_type", "site"]).copy()


def plot_all_km(
    sample_df: pd.DataFrame,
    pairs: pd.DataFrame,
    cox: pd.DataFrame,
    out_dir: Path,
    *,
    min_group_n: int,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    m1 = cox.loc[
        cox["model"].astype(str).eq("univariate")
        & cox["status"].astype(str).eq("success")
    ].copy()
    if "method" in m1.columns:
        m1 = m1.loc[m1["method"].fillna("ordinary").eq("ordinary")]

    for _, r in pairs.iterrows():
        cancer = str(r["cancer_type"])
        site = str(r["site"])
        label = str(r["site_label"])
        endpoint = str(r.get("endpoint", "OS"))
        if endpoint not in {"OS", "PFS"}:
            endpoint = "OS"
        sub = sample_df.loc[
            sample_df["cancer_type"].eq(cancer) & sample_df["site"].eq(site)
        ]
        n_low = int((sub["phospho_group"].astype(str) == "low").sum())
        n_high = int((sub["phospho_group"].astype(str) == "high").sum())
        if n_low < min_group_n or n_high < min_group_n:
            rows.append(
                {
                    "cancer_type": cancer,
                    "site": site,
                    "site_label": label,
                    "endpoint": endpoint,
                    "n_events_m1": r.get("n_events_m1"),
                    "analysis_depth": r.get("analysis_depth"),
                    "status": "skipped_small_groups",
                    "n_low": n_low,
                    "n_high": n_high,
                }
            )
            continue

        cox_row = m1.loc[m1["cancer_type"].eq(cancer) & m1["site"].eq(site)]
        cox_hr = cox_ci_low = cox_ci_high = cox_p = cox_q = None
        if not cox_row.empty:
            x = cox_row.iloc[0]
            cox_hr, cox_ci_low, cox_ci_high = x.get("HR_phospho"), x.get("CI_low"), x.get("CI_high")
            cox_p, cox_q = x.get("p_raw"), x.get("q_bh")

        safe = label.replace("/", "_").replace("|", "_")
        path = out_dir / f"KM_{cancer}_{safe}_{endpoint}.png"
        lp = make_km_plot(
            sample_df,
            cancer_type=cancer,
            site=site,
            site_label=label,
            endpoint=endpoint,
            out_path=path,
            cox_hr=cox_hr,
            cox_ci_low=cox_ci_low,
            cox_ci_high=cox_ci_high,
            cox_p=cox_p,
            cox_q=cox_q,
        )
        # also pdf sibling if png written
        pdf = path.with_suffix(".pdf")
        if path.exists() and not pdf.exists():
            # make_km_plot only writes png; re-save pdf from same figure is hard —
            # leave png as primary; small-multiples covers overview.
            pass
        rows.append(
            {
                "cancer_type": cancer,
                "site": site,
                "site_label": label,
                "endpoint": endpoint,
                "n_events_m1": r.get("n_events_m1"),
                "analysis_depth": r.get("analysis_depth"),
                "status": "plotted" if path.exists() else "failed",
                "logrank_p": lp,
                "n_low": n_low,
                "n_high": n_high,
                "path": str(path) if path.exists() else "",
                "cox_HR": cox_hr,
                "cox_p": cox_p,
                "cox_q": cox_q,
            }
        )
        print(f"  KM {cancer} {label} {endpoint}: logrank={lp}")
    return pd.DataFrame(rows)


def plot_forest_m1(cox: pd.DataFrame, pairs: pd.DataFrame, out_stem: Path) -> None:
    m1 = cox.loc[
        cox["model"].astype(str).eq("univariate")
        & cox["status"].astype(str).eq("success")
    ].copy()
    if "method" in m1.columns:
        m1 = m1.loc[m1["method"].fillna("ordinary").eq("ordinary")]
    if m1.empty:
        print("forest skipped: no successful M1 fits")
        return
    m1 = m1.merge(
        pairs[["cancer_type", "site", "n_events_m1", "analysis_depth"]],
        on=["cancer_type", "site"],
        how="left",
    )
    m1["HR_phospho"] = pd.to_numeric(m1["HR_phospho"], errors="coerce")
    m1["CI_low"] = pd.to_numeric(m1["CI_low"], errors="coerce")
    m1["CI_high"] = pd.to_numeric(m1["CI_high"], errors="coerce")
    m1["p_raw"] = pd.to_numeric(m1["p_raw"], errors="coerce")
    m1["q_bh"] = pd.to_numeric(m1["q_bh"], errors="coerce")
    m1 = m1.dropna(subset=["HR_phospho"]).sort_values(
        ["q_bh", "p_raw", "HR_phospho"], ascending=[True, True, True]
    )
    m1["y_label"] = m1["cancer_type"].astype(str) + " · " + m1["site_label"].astype(str)

    n = len(m1)
    fig_h = max(4.0, 0.38 * n + 1.8)
    fig, ax = plt.subplots(figsize=(8.2, fig_h))
    y = np.arange(n)
    hr = m1["HR_phospho"].to_numpy(dtype=float)
    lo = m1["CI_low"].to_numpy(dtype=float)
    hi = m1["CI_high"].to_numpy(dtype=float)
    # color by BH / raw
    colors = []
    for _, r in m1.iterrows():
        if np.isfinite(r["q_bh"]) and r["q_bh"] <= 0.05:
            colors.append("#C0392B")
        elif np.isfinite(r["p_raw"]) and r["p_raw"] < 0.05:
            colors.append("#E67E22")
        else:
            colors.append("#5D6D7E")
    ax.axvline(1.0, color="#333333", lw=1.0, ls="--", zorder=0)
    ax.errorbar(
        hr,
        y,
        xerr=[hr - lo, hi - hr],
        fmt="none",
        ecolor="#888888",
        elinewidth=1.2,
        capsize=2.5,
        zorder=1,
    )
    ax.scatter(hr, y, c=colors, s=42, zorder=2, edgecolors="white", linewidths=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(m1["y_label"].tolist(), fontsize=8.5)
    ax.set_xlabel("Univariate Cox HR (high vs low hotspot activity)")
    ax.set_title("Hotspot survival — M1 HR forest (all successful fits)")
    ax.set_xscale("log")
    # annotate p on the right
    xmax = float(np.nanmax(hi[np.isfinite(hi)])) if np.isfinite(hi).any() else 3.0
    xmin = float(np.nanmin(lo[np.isfinite(lo)])) if np.isfinite(lo).any() else 0.3
    ax.set_xlim(max(xmin * 0.85, 0.05), xmax * 1.35)
    for yi, (_, r) in enumerate(m1.iterrows()):
        q = r["q_bh"]
        p = r["p_raw"]
        txt = f"p={p:.3g}" if np.isfinite(p) else ""
        if np.isfinite(q):
            txt += f", q={q:.3g}"
        ev = r.get("n_events_m1")
        if pd.notna(ev):
            txt += f" | e={int(ev)}"
        ax.text(xmax * 1.02, yi, txt, va="center", ha="left", fontsize=7.5, color="#444444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # legend
    from matplotlib.lines import Line2D

    leg = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#C0392B", markersize=8, label="BH q≤0.05"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#E67E22", markersize=8, label="raw p<0.05"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#5D6D7E", markersize=8, label="ns"),
    ]
    ax.legend(handles=leg, frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{out_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    m1.to_csv(Path(str(out_stem) + "_table.csv"), index=False)
    print(f"forest → {out_stem}.pdf ({n} pairs)")


def _mini_km_ax(
    ax: plt.Axes,
    sample_df: pd.DataFrame,
    cancer: str,
    site: str,
    label: str,
    endpoint: str,
) -> Optional[float]:
    if KaplanMeierFitter is None or logrank_test is None:
        return None
    time_col, event_col = endpoint_columns(endpoint)
    sub = sample_df.loc[
        sample_df["cancer_type"].eq(cancer) & sample_df["site"].eq(site)
    ].copy()
    if sub.empty:
        ax.axis("off")
        return None
    sub[time_col] = _safe_to_numeric(sub[time_col])
    sub[event_col] = _safe_to_numeric(sub[event_col])
    sub = sub.dropna(subset=[time_col, event_col, "phospho_group"])
    colors = {"low": "#1B6CA8", "high": "#C0392B"}
    logrank_p = np.nan
    try:
        low = sub.loc[sub["phospho_group"].eq("low")]
        high = sub.loc[sub["phospho_group"].eq("high")]
        if len(low) >= 3 and len(high) >= 3:
            lr = logrank_test(
                low[time_col],
                high[time_col],
                event_observed_A=low[event_col],
                event_observed_B=high[event_col],
            )
            logrank_p = float(lr.p_value)
            for group in ("low", "high"):
                g = sub.loc[sub["phospho_group"].eq(group)]
                kmf = KaplanMeierFitter()
                kmf.fit(g[time_col], event_observed=g[event_col])
                kmf.plot_survival_function(
                    ax=ax, color=colors[group], ci_show=False, linewidth=1.6, legend=False
                )
    except Exception:
        pass
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{cancer}\n{label}", fontsize=7.5, pad=2)
    ptxt = f"p={logrank_p:.2g}" if np.isfinite(logrank_p) else ""
    ax.text(0.98, 0.05, ptxt, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="#444")
    ax.tick_params(labelsize=6, length=2)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return logrank_p if np.isfinite(logrank_p) else None


def plot_small_multiples(
    sample_df: pd.DataFrame,
    pairs: pd.DataFrame,
    out_stem: Path,
    *,
    min_group_n: int,
    ncols: int = 4,
) -> None:
    plot_pairs = []
    for _, r in pairs.iterrows():
        cancer, site = str(r["cancer_type"]), str(r["site"])
        sub = sample_df.loc[
            sample_df["cancer_type"].eq(cancer) & sample_df["site"].eq(site)
        ]
        n_low = int((sub["phospho_group"].astype(str) == "low").sum())
        n_high = int((sub["phospho_group"].astype(str) == "high").sum())
        if n_low >= min_group_n and n_high >= min_group_n:
            plot_pairs.append(r)
    if not plot_pairs:
        print("small multiples skipped")
        return
    n = len(plot_pairs)
    nrows = int(np.ceil(n / ncols))
    fig_w = 2.6 * ncols
    fig_h = 2.15 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    for i, r in enumerate(plot_pairs):
        ax = axes[i // ncols][i % ncols]
        endpoint = str(r.get("endpoint", "OS"))
        if endpoint not in {"OS", "PFS"}:
            endpoint = "OS"
        _mini_km_ax(
            ax,
            sample_df,
            str(r["cancer_type"]),
            str(r["site"]),
            str(r["site_label"]),
            endpoint,
        )
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(
        "Hotspot activity high/low vs survival (all pairs with ≥3 per group)",
        fontsize=11,
        y=1.01,
    )
    # shared legend
    from matplotlib.lines import Line2D

    leg = [
        Line2D([0], [0], color="#1B6CA8", lw=2, label="Low activity"),
        Line2D([0], [0], color="#C0392B", lw=2, label="High activity"),
    ]
    fig.legend(handles=leg, loc="upper right", frameon=False, fontsize=8, bbox_to_anchor=(0.99, 1.0))
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{out_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"small multiples → {out_stem}.pdf ({n} panels)")


def main() -> None:
    args = parse_args()
    surv_dir = args.surv_dir
    sample_path = surv_dir / "sample_hotspot_activity_groups.csv"
    if not sample_path.exists():
        raise FileNotFoundError(sample_path)
    sample_df = pd.read_csv(sample_path)
    pairs = _pair_table(surv_dir)
    cox = pd.read_csv(surv_dir / "cox_results_bh_fdr.csv")

    print(f"Pairs to consider: {len(pairs)}")
    km_dir = surv_dir / "km_curves_all"
    manifest = plot_all_km(sample_df, pairs, cox, km_dir, min_group_n=args.min_group_n)
    manifest.to_csv(surv_dir / "km_plot_manifest_all.csv", index=False)
    print(f"KM plotted: {(manifest['status']=='plotted').sum()} / {len(manifest)}")

    plot_forest_m1(cox, pairs, surv_dir / "forest_univariate_HR_all")
    plot_small_multiples(
        sample_df, pairs, surv_dir / "km_small_multiples_overview", min_group_n=args.min_group_n
    )
    print(f"Done → {surv_dir}")


if __name__ == "__main__":
    main()
