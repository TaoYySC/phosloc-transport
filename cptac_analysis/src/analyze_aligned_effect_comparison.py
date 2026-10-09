#!/usr/bin/env python3
"""Part 2: direction-aligned effect comparison across four classes.

1) Association-level E_aligned = ΔE × S_loc × S_reg for all testable associations.
2) Cancer-level control-adjusted effects from figure4c (observed − random directional),
   already on the scale where positive = more concordant than random.
3) Per-class pooled effect (fixed-effect meta), Wilson-style bootstrap 95% CI,
   one-sided empirical P for H0: E_aligned ≤ 0, BH across classes.
4) Between-class comparison via stratified bootstrap over cancers.
5) Main 4-row pooled forest + per-cancer forests for supplement.
"""

from __future__ import annotations

from pathlib import Path

# Repo-relative results root (cptac_analysis/results)
RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

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
OUT_DIR = TWO_ROOT / "aligned_effect_comparison"

# arm, regulation, S_loc, S_reg, short label
ARMS: List[Tuple[str, str, int, int, str]] = [
    ("Import_activate", "activate", +1, +1, "Nuclear accumulation\n× activated targets"),
    ("Import_repress", "repress", +1, -1, "Nuclear accumulation\n× repressed targets"),
    ("Export_activate", "activate", -1, +1, "Cytoplasmic redistribution\n× activated targets"),
    ("Export_repress", "repress", -1, -1, "Cytoplasmic redistribution\n× repressed targets"),
]

N_BOOT = 5000
RNG_SEED = 20260929


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    if not mask.any():
        return out
    pv = p[mask]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out_m = np.empty(n, dtype=float)
    out_m[order] = q
    out[mask] = out_m
    return out


def _fixed_effect_meta(effects: Sequence[float], ses: Sequence[float]) -> Tuple[float, float, float, float]:
    effects_array = np.asarray(effects, dtype=float)
    ses_array = np.asarray(ses, dtype=float)
    valid = np.isfinite(effects_array) & np.isfinite(ses_array) & (ses_array > 0)
    if not np.any(valid):
        return np.nan, np.nan, np.nan, np.nan
    ev = effects_array[valid]
    sv = ses_array[valid]
    w = 1.0 / (sv ** 2)
    pooled = float(np.sum(w * ev) / np.sum(w))
    se = float(np.sqrt(1.0 / np.sum(w)))
    return pooled, pooled - 1.96 * se, pooled + 1.96 * se, se


def _load_association_aligned() -> pd.DataFrame:
    """All testable site×cancer associations with E_aligned = ΔE × S_loc × S_reg."""
    frames = []
    for arm, reg, s_loc, s_reg, label in ARMS:
        path = (
            TWO_ROOT
            / arm
            / "cptac_filtered"
            / f"two_side_high_low_phospho_comparison_by_site_{arm.lower()}.csv"
        )
        df = pd.read_csv(path, low_memory=False)
        df["delta"] = pd.to_numeric(df["delta_high_minus_low"], errors="coerce")
        df["q"] = pd.to_numeric(df["wilcoxon_q_bh"], errors="coerce")
        df["arm"] = arm
        df["category"] = label
        df["S_localization"] = s_loc
        df["S_regulation"] = s_reg
        df["align_sign"] = s_loc * s_reg
        df["E_aligned"] = df["delta"] * df["align_sign"]
        if "evaluable" in df.columns:
            df["evaluable"] = df["evaluable"].astype(bool)
        else:
            df["evaluable"] = df["q"].notna()
        # Association-level summaries use only n>=10 evaluable tests.
        df["testable"] = df["evaluable"] & df["q"].notna() & df["delta"].notna()
        df["bh_sig"] = df["testable"] & df["q"].lt(0.05)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _load_cancer_control_adjusted() -> pd.DataFrame:
    """Cancer-level control-adjusted effects, direction-aligned for plotting/tests.

    figure4c reports effect = mean(obs directional) − mean(random directional).
    For classes whose expected raw ΔE is negative (Import×repress, Export×activate),
    flip effect and CI so that positive always means 'more concordant than random
    on the expected-direction scale' used in the manuscript:

        E_aligned = −E
        CI_aligned = [−CI_upper, −CI_lower]
    """
    rows = []
    for arm, reg, s_loc, s_reg, label in ARMS:
        stats_path = (
            TWO_ROOT
            / arm
            / "figure4"
            / "figure4c_observed_vs_random"
            / "Figure4c_FigureA_gene_level_observed_vs_random_by_cancer_stats.csv"
        )
        stats = pd.read_csv(stats_path)
        direction = "Import" if arm.startswith("Import") else "Export"
        sub = stats[
            stats["direction_short"].astype(str).eq(direction)
            & stats["target_regulation"].astype(str).str.lower().eq(reg)
        ].copy()
        if sub.empty:
            continue
        sub["arm"] = arm
        sub["category"] = label
        sub["align_sign"] = s_loc * s_reg
        # needs_flip: expected raw direction is negative
        sub["needs_sign_flip"] = sub["align_sign"].lt(0)

        e = pd.to_numeric(sub["effect_size_mean_diff"], errors="coerce")
        lo = pd.to_numeric(sub["effect_size_ci_lower"], errors="coerce")
        hi = pd.to_numeric(sub["effect_size_ci_upper"], errors="coerce")
        se = pd.to_numeric(sub["effect_size_se"], errors="coerce")
        miss_se = se.isna() & lo.notna() & hi.notna()
        se = se.copy()
        se.loc[miss_se] = (hi.loc[miss_se] - lo.loc[miss_se]) / (2 * 1.96)

        sub["E_original"] = e
        sub["CI_low_original"] = lo
        sub["CI_high_original"] = hi

        flip = sub["needs_sign_flip"].to_numpy()
        e_al = e.to_numpy(dtype=float).copy()
        lo_al = lo.to_numpy(dtype=float).copy()
        hi_al = hi.to_numpy(dtype=float).copy()
        e_al[flip] = -e_al[flip]
        # flip CI endpoints and swap
        lo_f = lo_al.copy()
        hi_f = hi_al.copy()
        lo_al[flip] = -hi_f[flip]
        hi_al[flip] = -lo_f[flip]

        sub["E_aligned_control"] = e_al
        sub["E_aligned_ci_low"] = lo_al
        sub["E_aligned_ci_high"] = hi_al
        sub["E_aligned_se"] = se.to_numpy(dtype=float)  # SE unchanged under sign flip
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def _bootstrap_pooled(
    effects: np.ndarray,
    ses: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float, float, np.ndarray]:
    """Bootstrap cancers → pooled effect, percentile CI, one-sided empirical P (≤0)."""
    n = len(effects)
    obs, ci_lo_fe, ci_hi_fe, se_fe = _fixed_effect_meta(effects, ses)
    boots = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = _fixed_effect_meta(effects[idx], ses[idx])[0]
    finite = boots[np.isfinite(boots)]
    if len(finite) == 0:
        return obs, ci_lo_fe, ci_hi_fe, np.nan, boots
    # percentile CI
    ci_lo = float(np.quantile(finite, 0.025))
    ci_hi = float(np.quantile(finite, 0.975))
    # one-sided empirical P for H0: E ≤ 0 vs H1: E > 0
    p_emp = float((np.sum(finite <= 0) + 1) / (len(finite) + 1))
    return obs, ci_lo, ci_hi, p_emp, boots


def summarize_control_adjusted(cancer_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    boot_store: Dict[str, np.ndarray] = {}
    for arm, _, s_loc, s_reg, label in ARMS:
        sub = cancer_df[cancer_df["arm"] == arm].copy()
        effects = sub["E_aligned_control"].to_numpy(dtype=float)
        ses = sub["E_aligned_se"].to_numpy(dtype=float)
        valid = np.isfinite(effects) & np.isfinite(ses) & (ses > 0)
        effects, ses = effects[valid], ses[valid]
        pooled_fe, fe_lo, fe_hi, fe_se = _fixed_effect_meta(effects, ses)
        # analytic one-sided normal P as supplement
        if np.isfinite(pooled_fe) and np.isfinite(fe_se) and fe_se > 0:
            p_norm = float(1.0 - norm.cdf(pooled_fe / fe_se))
        else:
            p_norm = np.nan
        pooled, ci_lo, ci_hi, p_emp, boots = _bootstrap_pooled(effects, ses, N_BOOT, rng)
        boot_store[arm] = boots
        rows.append(
            {
                "arm": arm,
                "category": label,
                "n_cancers": int(len(effects)),
                "needs_sign_flip": bool(s_loc * s_reg < 0),
                "pooled_E_aligned": pooled_fe,
                "pooled_se_fixed": fe_se,
                "ci_low_fixed": fe_lo,
                "ci_high_fixed": fe_hi,
                "ci_low_bootstrap": ci_lo,
                "ci_high_bootstrap": ci_hi,
                "p_empirical_gt0": p_emp,
                "p_normal_gt0": p_norm,
            }
        )
    summary = pd.DataFrame(rows)
    summary["p_empirical_q_bh"] = _bh_adjust(summary["p_empirical_gt0"].to_numpy())
    summary["p_normal_q_bh"] = _bh_adjust(summary["p_normal_gt0"].to_numpy())
    return summary, boot_store


def pairwise_bootstrap_compare(boot_store: Dict[str, np.ndarray], summary: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise and overall between-category comparison from bootstrap replicates."""
    arms = [a for a, _, _, _, _ in ARMS]
    labels = {a: lab for a, _, _, _, lab in ARMS}
    # overall: variance of the four pooled means across bootstrap draws
    mats = []
    for arm in arms:
        mats.append(boot_store[arm])
    boot_mat = np.vstack(mats)  # 4 × B
    # drop columns with any nan
    ok = np.all(np.isfinite(boot_mat), axis=0)
    boot_mat = boot_mat[:, ok]
    obs = summary.set_index("arm").loc[arms, "pooled_E_aligned"].to_numpy(dtype=float)
    obs_disp = float(np.nanvar(obs, ddof=0))
    null_disp = np.var(boot_mat - boot_mat.mean(axis=0, keepdims=True), axis=0)  # not right for overall test

    # Better overall test: for each bootstrap, compute between-category SS of pooled estimates;
    # compare observed between-SS to null where we randomly reassign category labels to the
    # vector of four observed pooled values — weak. Instead use pairwise only + omnibus on
    # absolute differences from bootstrap of category-wise pooled values:
    # H0: all equal → under bootstrap of independent category metas, P(max pairwise |diff| as large).
    obs_diffs = []
    pair_rows = []
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            a, b = arms[i], arms[j]
            diff_obs = float(obs[i] - obs[j])
            diff_boot = boot_store[a] - boot_store[b]
            finite = diff_boot[np.isfinite(diff_boot)]
            # two-sided: how often |boot diff| as large as |obs| when centered at 0
            # use percentile: P = mean( |boot - mean(boot)| >= |obs - mean(boot)| ) approx
            # simpler: shift-null p-value
            if len(finite):
                centered = finite - np.mean(finite)
                p = float((np.sum(np.abs(centered) >= abs(diff_obs - np.mean(finite))) + 1) / (len(finite) + 1))
                # cleaner two-sided percentile for H0: diff=0 using bootstrap distribution of diff
                # P = 2 * min( mean(boot<=0), mean(boot>=0) ) clipped
                p_perc = float(2 * min(np.mean(finite <= 0), np.mean(finite >= 0)))
                p_perc = float(min(1.0, p_perc))
                ci_lo = float(np.quantile(finite, 0.025))
                ci_hi = float(np.quantile(finite, 0.975))
            else:
                p_perc, ci_lo, ci_hi = np.nan, np.nan, np.nan
            pair_rows.append(
                {
                    "category_a": labels[a],
                    "category_b": labels[b],
                    "diff_a_minus_b": diff_obs,
                    "boot_ci_low": ci_lo,
                    "boot_ci_high": ci_hi,
                    "p_bootstrap_two_sided": p_perc,
                }
            )
            obs_diffs.append(abs(diff_obs))
    pairwise = pd.DataFrame(pair_rows)
    pairwise["p_bootstrap_q_bh"] = _bh_adjust(pairwise["p_bootstrap_two_sided"].to_numpy())

    # Omnibus: max absolute pairwise difference
    max_obs = float(np.max(obs_diffs)) if obs_diffs else np.nan
    if boot_mat.shape[1] > 0:
        max_boot = []
        for b in range(boot_mat.shape[1]):
            v = boot_mat[:, b]
            md = 0.0
            for i in range(len(v)):
                for j in range(i + 1, len(v)):
                    md = max(md, abs(float(v[i] - v[j])))
            max_boot.append(md)
        max_boot = np.asarray(max_boot)
        # under independent category bootstraps this is not a true null; report descriptive
        # Use permutation of category labels on cancer-level concatenated? Skip — report
        # observed max |diff| and bootstrap distribution of max |diff| as uncertainty.
        overall = pd.DataFrame(
            [
                {
                    "test": "max_abs_pairwise_pooled_diff",
                    "statistic_obs": max_obs,
                    "boot_median": float(np.median(max_boot)),
                    "boot_ci_low": float(np.quantile(max_boot, 0.025)),
                    "boot_ci_high": float(np.quantile(max_boot, 0.975)),
                    "note": (
                        "Between-category comparison uses cancer-stratified bootstrap of "
                        "fixed-effect pooled control-adjusted aligned effects; pairwise "
                        "two-sided percentile P for diff=0, BH-adjusted."
                    ),
                }
            ]
        )
    else:
        overall = pd.DataFrame(
            [{"test": "max_abs_pairwise_pooled_diff", "statistic_obs": max_obs, "note": "bootstrap failed"}]
        )
    return overall, pairwise


def summarize_association_level(assoc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(RNG_SEED + 1)
    for arm, _, _, _, label in ARMS:
        sub = assoc[assoc["arm"].eq(arm) & assoc["testable"]].copy()
        vals = sub["E_aligned"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        n = len(vals)
        mean = float(np.mean(vals)) if n else np.nan
        # bootstrap mean CI + one-sided empirical P
        if n:
            boots = np.array(
                [float(np.mean(vals[rng.integers(0, n, size=n)])) for _ in range(N_BOOT)]
            )
            ci_lo, ci_hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
            p_emp = float((np.sum(boots <= 0) + 1) / (len(boots) + 1))
        else:
            ci_lo = ci_hi = p_emp = np.nan
        rows.append(
            {
                "arm": arm,
                "category": label,
                "n_testable_associations": n,
                "n_bh_significant": int(sub["bh_sig"].sum()),
                "mean_E_aligned": mean,
                "median_E_aligned": float(np.median(vals)) if n else np.nan,
                "boot_ci_low": ci_lo,
                "boot_ci_high": ci_hi,
                "p_empirical_gt0": p_emp,
            }
        )
    out = pd.DataFrame(rows)
    out["p_empirical_q_bh"] = _bh_adjust(out["p_empirical_gt0"].to_numpy())
    return out


def _format_p(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.3f}"


def plot_pooled_forest(summary: pd.DataFrame, out_stem: Path, dpi: int = 300) -> None:
    """Main text: four-row forest of pooled aligned control-adjusted effects."""
    from matplotlib.ticker import MaxNLocator

    s = summary.set_index("arm").loc[[a for a, *_ in ARMS]]
    labels = [lab for *_, lab in ARMS]
    y = np.arange(len(labels), dtype=float)[::-1]
    effect = s["pooled_E_aligned"].to_numpy(dtype=float)
    lo = s["ci_low_bootstrap"].to_numpy(dtype=float)
    hi = s["ci_high_bootstrap"].to_numpy(dtype=float)
    miss = ~np.isfinite(lo) | ~np.isfinite(hi)
    lo[miss] = s["ci_low_fixed"].to_numpy(dtype=float)[miss]
    hi[miss] = s["ci_high_fixed"].to_numpy(dtype=float)[miss]
    p_emp = s["p_empirical_gt0"].to_numpy(dtype=float)
    q_bh = s["p_empirical_q_bh"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1.4, zorder=0)
    ax.errorbar(
        effect,
        y,
        xerr=np.vstack([effect - lo, hi - effect]),
        fmt="D",
        color="#2F4F6F",
        ecolor="#2F4F6F",
        elinewidth=1.7,
        capsize=3.5,
        capthick=1.3,
        markersize=7.5,
        markeredgewidth=0.8,
        zorder=3,
    )

    # Place stats just after each CI tip (same style as earlier version).
    for yi, e, l, h, p, q in zip(y, effect, lo, hi, p_emp, q_bh):
        ax.text(
            max(h, e) + 0.002,
            yi,
            f"{e:.3f}  [{l:.3f}, {h:.3f}]   P={_format_p(p)}  q={_format_p(q)}",
            va="center",
            ha="left",
            fontsize=8,
            color="#333333",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel(
        "Pooled direction-aligned effect relative to matched random genes",
        fontsize=10,
        labelpad=5,
    )
    ax.set_title("Direction-aligned effects across four classes", fontsize=12, pad=7)
    ax.tick_params(axis="x", labelsize=10, length=4, width=1.1)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
    ax.set_ylim(-0.35, len(labels) - 0.65)

    xmin = min(0.0, float(np.nanmin(lo))) - 0.008
    xmax = float(np.nanmax(hi)) + 0.055
    ax.set_xlim(xmin, xmax)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    fig.subplots_adjust(left=0.36, right=0.98, bottom=0.16, top=0.90)

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{out_stem}.{ext}", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_stem}.png")


def plot_per_cancer_forests(cancer_df: pd.DataFrame, out_dir: Path, dpi: int = 300) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for arm, _, _, _, label in ARMS:
        sub = cancer_df[cancer_df["arm"] == arm].copy()
        sub = sub.sort_values("cancer_type")
        if sub.empty:
            continue
        cancers = sub["cancer_type"].astype(str).tolist()
        effect = sub["E_aligned_control"].to_numpy(dtype=float)
        lo = sub["E_aligned_ci_low"].to_numpy(dtype=float)
        hi = sub["E_aligned_ci_high"].to_numpy(dtype=float)
        se = sub["E_aligned_se"].to_numpy(dtype=float)
        y = np.arange(len(cancers))[::-1]

        valid = np.isfinite(effect) & np.isfinite(se) & (se > 0)
        pooled, plo, phi, _ = _fixed_effect_meta(effect[valid], se[valid])

        fig_h = max(4.0, 0.35 * len(cancers) + 1.8)
        fig, ax = plt.subplots(figsize=(7.0, fig_h))
        ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0, zorder=0)
        ax.errorbar(
            effect,
            y,
            xerr=np.vstack([effect - lo, hi - effect]),
            fmt="o",
            color="#C97B49",
            ecolor="#C97B49",
            elinewidth=1.2,
            capsize=3,
            markersize=5.5,
        )
        # pooled diamond below
        y_pool = -1.2
        ax.plot([plo, phi], [y_pool, y_pool], color="#4A2F1F", linewidth=1.4)
        ax.scatter([pooled], [y_pool], marker="D", s=40, color="#4A2F1F", zorder=3)
        ax.text(phi + 0.002, y_pool, f"Pooled {pooled:.3f}", va="center", fontsize=8, color="#4A2F1F")

        ax.set_yticks(list(y) + [y_pool])
        ax.set_yticklabels(cancers + ["Pooled"], fontsize=8)
        ax.set_xlabel("Aligned control-adjusted effect (obs − random)")
        ax.set_title(f"{label}: per-cancer aligned effects", fontsize=10, pad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        stem = out_dir / f"per_cancer_forest_{arm}"
        for ext in ("png", "pdf", "svg"):
            fig.savefig(f"{stem}.{ext}", dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    supp = OUT_DIR / "supplement_per_cancer_forests"
    supp.mkdir(parents=True, exist_ok=True)

    assoc = _load_association_aligned()
    assoc_testable = assoc[assoc["testable"]].copy()
    assoc_testable.to_csv(OUT_DIR / "association_level_E_aligned_all_testable.csv", index=False)
    assoc_summary = summarize_association_level(assoc)
    assoc_summary.to_csv(OUT_DIR / "association_level_E_aligned_summary.csv", index=False)

    cancer_df = _load_cancer_control_adjusted()
    cancer_df.to_csv(OUT_DIR / "cancer_level_E_aligned_control_adjusted.csv", index=False)

    pooled_summary, boot_store = summarize_control_adjusted(cancer_df)
    pooled_summary.to_csv(OUT_DIR / "pooled_E_aligned_control_adjusted_summary.csv", index=False)

    overall, pairwise = pairwise_bootstrap_compare(boot_store, pooled_summary)
    overall.to_csv(OUT_DIR / "between_category_bootstrap_overall.csv", index=False)
    pairwise.to_csv(OUT_DIR / "between_category_bootstrap_pairwise.csv", index=False)

    plot_pooled_forest(pooled_summary, OUT_DIR / "pooled_aligned_effect_forest")
    plot_per_cancer_forests(cancer_df, supp)

    # README note
    note = OUT_DIR / "README_aligned_effect.txt"
    note.write_text(
        "\n".join(
            [
                "Direction-aligned effect analysis (Part 2)",
                "",
                "Association-level:",
                "  E_aligned = delta_high_minus_low × S_localization × S_regulation",
                "  Includes all testable associations (significant and non-significant).",
                "",
                "Control-adjusted (primary for pooled forest):",
                "  Start from figure4c cancer-level effect_size_mean_diff (obs − random).",
                "  For classes with expected-negative raw ΔE (Import×repress, Export×activate):",
                "      E_aligned = −E",
                "      CI_aligned = [−CI_upper, −CI_lower]",
                "  After this flip, positive = more concordant with the expected direction",
                "  than matched random genes (manuscript scale).",
                "",
                "Hypothesis per class: H0 E_aligned ≤ 0 vs H1 E_aligned > 0",
                "  (one-sided empirical bootstrap P; BH across four classes).",
                "",
                "Manuscript wording note:",
                "  Displayed CIs are nominal 95% (not multiplicity-adjusted).",
                "  Prefer describing 'positive pooled effects' in the main text;",
                "  reserve 'statistically significant after BH' for classes with q < 0.05.",
                "  Current BH q for one-sided bootstrap P: first three classes ~0.028,",
                "  Cytoplasmic redistribution × repressed targets q ~0.125.",
                "",
                "Relation to concordance analysis (Part 1):",
                "  Part 1 asks whether BH-significant associations have the expected raw ΔE sign.",
                "  Part 2 asks whether all real targets shift toward the expected direction",
                "  relative to random genes (control-adjusted). These can differ, e.g. Export×activate",
                "  may have many absolute-discordant BH hits yet still show a relative",
                "  expected-direction offset vs random after alignment.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("=== Association-level mean E_aligned (all testable) ===")
    print(assoc_summary.to_string(index=False))
    print("\n=== Control-adjusted pooled E_aligned (primary) ===")
    print(
        pooled_summary[
            [
                "category",
                "needs_sign_flip",
                "n_cancers",
                "pooled_E_aligned",
                "ci_low_bootstrap",
                "ci_high_bootstrap",
                "p_empirical_gt0",
                "p_empirical_q_bh",
            ]
        ].to_string(index=False)
    )
    print("\n=== Pairwise bootstrap ===")
    print(pairwise.to_string(index=False))
    print(f"\nWrote outputs under {OUT_DIR}")


if __name__ == "__main__":
    main()
