#!/usr/bin/env python3
"""Concordant vs discordant direction among two-sided BH-significant associations.

Descriptive concordant fractions with Wilson 95% CIs, per-class binomial tests
vs 0.5, and permutation tests for between-class differences. Categories with
too few BH-significant associations are flagged as insufficient evidence.
"""

from __future__ import annotations

from pathlib import Path

# Repo-relative results root (cptac_analysis/results)
RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

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
OUT_DIR = TWO_ROOT / "concordance_direction_summary"

# Minimum BH-significant directional associations to interpret a category
MIN_N_FOR_INFERENCE = 5
N_PERM = 10000
RNG_SEED = 20260929

# (arm_folder, regulation_csv_suffix, display_label, expected_sign of ΔE)
ARMS: List[Tuple[str, str, str, int]] = [
    ("Import_activate", "activate", "Nuclear accumulation\n× activated targets", +1),
    ("Import_repress", "repress", "Nuclear accumulation\n× repressed targets", -1),
    ("Export_activate", "activate", "Cytoplasmic redistribution\n× activated targets", -1),
    ("Export_repress", "repress", "Cytoplasmic redistribution\n× repressed targets", +1),
]

SHORT_LABELS = {
    "Import_activate": "Nuclear accumulation × activated targets",
    "Import_repress": "Nuclear accumulation × repressed targets",
    "Export_activate": "Cytoplasmic redistribution × activated targets",
    "Export_repress": "Cytoplasmic redistribution × repressed targets",
}


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
    out_masked = np.empty(n, dtype=float)
    out_masked[order] = q
    out[mask] = out_masked
    return out


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion."""
    if n <= 0:
        return float("nan"), float("nan")
    z = float(norm.ppf(1.0 - alpha / 2.0))
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    half = (z / denom) * np.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n))
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def _load_arm(arm: str, regulation: str, expected_sign: int) -> pd.DataFrame:
    path = (
        TWO_ROOT
        / arm
        / "cptac_filtered"
        / f"two_side_high_low_phospho_comparison_by_site_{arm.lower()}.csv"
    )
    df = pd.read_csv(path, low_memory=False)
    df["delta"] = pd.to_numeric(df["delta_high_minus_low"], errors="coerce")
    df["q"] = pd.to_numeric(df["wilcoxon_q_bh"], errors="coerce")
    df["p_raw"] = pd.to_numeric(df["wilcoxon_p_raw"], errors="coerce")
    df["cancer_type"] = df["cancer_type"].astype(str)
    df["site_label"] = df["site_label"].astype(str)
    df["arm"] = arm
    df["category"] = SHORT_LABELS[arm]
    df["expected_sign"] = expected_sign
    if "evaluable" in df.columns:
        df["evaluable"] = df["evaluable"].astype(bool)
    else:
        df["evaluable"] = df["q"].notna()
    # Only n>=10 (evaluable) associations enter tests / BH denominators.
    df["testable"] = df["evaluable"] & df["q"].notna() & df["delta"].notna()
    df["bh_sig"] = df["testable"] & df["q"].lt(0.05)
    df["direction_class"] = np.where(
        ~df["bh_sig"] | df["delta"].isna() | df["delta"].eq(0),
        "n/a",
        np.where(np.sign(df["delta"]) == expected_sign, "concordant", "discordant"),
    )
    return df


def build_long() -> pd.DataFrame:
    return pd.concat(
        [_load_arm(arm, reg, sign) for arm, reg, _, sign in ARMS],
        ignore_index=True,
    )


def summarize(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, _, _, _ in ARMS:
        sub = long_df[long_df["arm"] == arm]
        testable = int(sub["testable"].sum())
        sig = sub[sub["bh_sig"]]
        n_sig = int(len(sig))
        n_conc = int((sig["direction_class"] == "concordant").sum())
        n_disc = int((sig["direction_class"] == "discordant").sum())
        denom = n_conc + n_disc
        frac = float(n_conc / denom) if denom > 0 else np.nan
        ci_lo, ci_hi = wilson_ci(n_conc, denom) if denom > 0 else (np.nan, np.nan)
        insufficient = denom < MIN_N_FOR_INFERENCE
        if denom > 0 and not insufficient:
            binom_p = float(binomtest(n_conc, denom, p=0.5, alternative="greater").pvalue)
        elif denom > 0 and insufficient:
            # still compute for completeness, but flag as not for inference
            binom_p = float(binomtest(n_conc, denom, p=0.5, alternative="greater").pvalue)
        else:
            binom_p = np.nan
        rows.append(
            {
                "arm": arm,
                "category": SHORT_LABELS[arm],
                "n_testable": testable,
                "n_bh_significant": n_sig,
                "n_concordant": n_conc,
                "n_discordant": n_disc,
                "concordant_fraction": frac,
                "wilson_ci_low": ci_lo,
                "wilson_ci_high": ci_hi,
                "count_label": f"{n_conc}/{denom}" if denom > 0 else "0/0",
                "insufficient_evidence": bool(insufficient),
                "binom_p_gt_0.5": binom_p,
                "note": "insufficient evidence — do not interpret fraction"
                if insufficient
                else "",
            }
        )
    summary = pd.DataFrame(rows)
    # BH only among categories with sufficient evidence
    q = np.full(len(summary), np.nan)
    ok = ~summary["insufficient_evidence"].to_numpy()
    q[ok] = _bh_adjust(summary.loc[ok, "binom_p_gt_0.5"].to_numpy())
    summary["binom_q_bh"] = q
    return summary


def _proportion_dispersion(labels: np.ndarray, y: np.ndarray) -> float:
    """Chi-square-like dispersion of concordant rates across category labels."""
    cats = np.unique(labels)
    if len(cats) < 2:
        return 0.0
    overall = float(np.mean(y))
    if overall <= 0 or overall >= 1:
        # still allow separation by category means
        means = [float(np.mean(y[labels == c])) for c in cats]
        return float(np.var(means) * len(y))
    stat = 0.0
    for c in cats:
        mask = labels == c
        n = int(mask.sum())
        if n == 0:
            continue
        phat = float(np.mean(y[mask]))
        stat += n * (phat - overall) ** 2 / (overall * (1.0 - overall))
    return float(stat)


def permutation_category_tests(long_df: pd.DataFrame, n_perm: int = N_PERM) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Permutation tests among categories with sufficient BH-significant n."""
    sig = long_df[
        long_df["bh_sig"] & long_df["direction_class"].isin(["concordant", "discordant"])
    ].copy()
    counts = sig.groupby("arm").size()
    usable_arms = [a for a, _, _, _ in ARMS if int(counts.get(a, 0)) >= MIN_N_FOR_INFERENCE]
    excluded = [SHORT_LABELS[a] for a, _, _, _ in ARMS if a not in usable_arms]

    if len(usable_arms) < 2:
        overall = pd.DataFrame(
            [
                {
                    "test": "permutation_dispersion_across_categories",
                    "n_perm": n_perm,
                    "statistic_obs": np.nan,
                    "p_value": np.nan,
                    "categories_included": "",
                    "categories_excluded": "; ".join(excluded),
                    "note": "need ≥2 categories with sufficient n",
                }
            ]
        )
        return overall, pd.DataFrame()

    sub = sig[sig["arm"].isin(usable_arms)].copy()
    y = (sub["direction_class"] == "concordant").to_numpy(dtype=float)
    labels = sub["arm"].to_numpy()
    obs = _proportion_dispersion(labels, y)

    rng = np.random.default_rng(RNG_SEED)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        null[i] = _proportion_dispersion(rng.permutation(labels), y)
    # two-sided: as or more extreme dispersion
    p_overall = float((np.sum(null >= obs) + 1) / (n_perm + 1))

    overall = pd.DataFrame(
        [
            {
                "test": "permutation_dispersion_across_categories",
                "n_perm": n_perm,
                "statistic_obs": obs,
                "p_value": p_overall,
                "categories_included": "; ".join(SHORT_LABELS[a] for a in usable_arms),
                "categories_excluded": "; ".join(excluded) if excluded else "",
                "note": (
                    f"Permute category labels among BH-significant associations; "
                    f"exclude categories with n < {MIN_N_FOR_INFERENCE}"
                ),
            }
        ]
    )

    # Pairwise absolute difference in proportions
    pair_rows = []
    for i in range(len(usable_arms)):
        for j in range(i + 1, len(usable_arms)):
            a, b = usable_arms[i], usable_arms[j]
            ya = y[labels == a]
            yb = y[labels == b]
            na, nb = len(ya), len(yb)
            obs_diff = abs(float(ya.mean()) - float(yb.mean()))
            pooled = np.concatenate([ya, yb])
            null_d = np.empty(n_perm, dtype=float)
            for k in range(n_perm):
                perm = rng.permutation(pooled)
                null_d[k] = abs(float(perm[:na].mean()) - float(perm[na:].mean()))
            p_pair = float((np.sum(null_d >= obs_diff) + 1) / (n_perm + 1))
            pair_rows.append(
                {
                    "category_a": SHORT_LABELS[a],
                    "category_b": SHORT_LABELS[b],
                    "frac_a": float(ya.mean()),
                    "frac_b": float(yb.mean()),
                    "abs_diff": obs_diff,
                    "n_a": na,
                    "n_b": nb,
                    "permutation_p": p_pair,
                }
            )
    pairwise = pd.DataFrame(pair_rows)
    if not pairwise.empty:
        pairwise["permutation_q_bh"] = _bh_adjust(pairwise["permutation_p"].to_numpy())
    return overall, pairwise


def plot_fraction_ci(summary: pd.DataFrame, out_stem: Path, dpi: int = 300) -> None:
    arms = [a for a, _, _, _ in ARMS]
    s = summary.set_index("arm").loc[arms]
    # Horizontal layout: long category labels on y, denser filled panel.
    y = np.arange(len(arms), dtype=float)[::-1]
    frac = s["concordant_fraction"].to_numpy(dtype=float)
    lo = s["wilson_ci_low"].to_numpy(dtype=float)
    hi = s["wilson_ci_high"].to_numpy(dtype=float)
    xerr = np.vstack([frac - lo, hi - frac])
    insuff = s["insufficient_evidence"].to_numpy(dtype=bool)
    labels = [
        "Nuclear accumulation\n× activated targets",
        "Nuclear accumulation\n× repressed targets",
        "Cytoplasmic redistribution\n× activated targets",
        "Cytoplasmic redistribution\n× repressed targets",
    ]

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.axvline(0.5, color="#888888", linestyle="--", linewidth=1.8, zorder=0)

    ok = ~insuff
    if ok.any():
        ax.errorbar(
            frac[ok],
            y[ok],
            xerr=xerr[:, ok],
            fmt="o",
            color="#2F4F6F",
            ecolor="#2F4F6F",
            elinewidth=3.0,
            capsize=7,
            capthick=2.6,
            markersize=16,
            markeredgewidth=1.3,
            zorder=3,
            label="Concordant fraction (Wilson 95% CI)",
        )
    if insuff.any():
        ax.errorbar(
            frac[insuff],
            y[insuff],
            xerr=xerr[:, insuff],
            fmt="o",
            color="#9AA4AE",
            ecolor="#9AA4AE",
            elinewidth=2.6,
            capsize=7,
            capthick=2.4,
            markersize=16,
            markerfacecolor="white",
            markeredgewidth=2.4,
            zorder=3,
            label="Insufficient evidence",
        )

    for i, arm in enumerate(arms):
        row = s.loc[arm]
        count = str(row["count_label"])
        yi = float(y[i])
        xi = float(row["concordant_fraction"])
        hi_i = float(row["wilson_ci_high"])
        ax.text(
            min(hi_i + 0.03, 0.98),
            yi + 0.18,
            count,
            ha="left",
            va="center",
            fontsize=14,
            color="#222222",
        )
        if bool(row["insufficient_evidence"]):
            ax.text(
                0.82,
                yi - 0.22,
                "insufficient evidence",
                ha="left",
                va="center",
                fontsize=12,
                color="#6B7280",
                style="italic",
            )

    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.55, len(arms) - 0.45)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=13)
    ax.set_xlabel("Concordant fraction", fontsize=15, labelpad=8)
    ax.tick_params(axis="x", labelsize=13, length=5, width=1.2)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.set_title(
        "Direction concordance among two-sided BH-significant associations\n"
        "(Wilson 95% CI; q < 0.05)",
        fontsize=15,
        pad=12,
    )
    ax.legend(
        frameon=False,
        loc="lower right",
        fontsize=12,
        handlelength=1.6,
        borderaxespad=0.4,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.3)
    ax.spines["bottom"].set_linewidth(1.3)
    fig.subplots_adjust(left=0.34, right=0.98, bottom=0.12, top=0.88)

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{out_stem}.{ext}", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_stem}.png")



def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long_df = build_long()
    long_df[long_df["bh_sig"]].to_csv(
        OUT_DIR / "bh_significant_associations_with_direction.csv", index=False
    )
    long_df.to_csv(OUT_DIR / "all_associations_with_direction_flags.csv", index=False)

    summary = summarize(long_df)
    summary.to_csv(OUT_DIR / "concordance_summary_by_category.csv", index=False)

    expect = pd.DataFrame(
        [
            {
                "arm": arm,
                "category": SHORT_LABELS[arm],
                "expected_delta_sign": "ΔE > 0" if sign > 0 else "ΔE < 0",
                "expected_sign": sign,
            }
            for arm, _, _, sign in ARMS
        ]
    )
    expect.to_csv(OUT_DIR / "expected_direction_lookup.csv", index=False)

    overall_perm, pairwise_perm = permutation_category_tests(long_df)
    overall_perm.to_csv(OUT_DIR / "concordance_permutation_overall.csv", index=False)
    pairwise_perm.to_csv(OUT_DIR / "concordance_permutation_pairwise.csv", index=False)

    plot_fraction_ci(summary, OUT_DIR / "concordance_fraction_point_ci")

    # remove superseded stacked-bar / chi2 outputs if present
    for obsolete in (
        "concordance_fraction_stacked_bar.png",
        "concordance_fraction_stacked_bar.pdf",
        "concordance_fraction_stacked_bar.svg",
        "concordance_contingency_and_chi2.csv",
        "concordance_pairwise_fisher.csv",
    ):
        p = OUT_DIR / obsolete
        if p.exists():
            p.unlink()

    show_cols = [
        "category",
        "n_concordant",
        "n_discordant",
        "concordant_fraction",
        "wilson_ci_low",
        "wilson_ci_high",
        "insufficient_evidence",
        "binom_p_gt_0.5",
        "binom_q_bh",
    ]
    print(summary[show_cols].to_string(index=False))
    print("\nPermutation overall:")
    print(overall_perm.to_string(index=False))
    print("\nPermutation pairwise:")
    print(pairwise_perm.to_string(index=False) if len(pairwise_perm) else "(none)")
    print(f"\nWrote outputs under {OUT_DIR}")


if __name__ == "__main__":
    main()
