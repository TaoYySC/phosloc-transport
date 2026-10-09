#!/usr/bin/env python3
"""Part 3: Known-containing vs Predicted-only comparison (revised).

Panels (supplement figure):
  a) BH-significant fraction with Wilson 95% CI + denominators
  b) Concordant fraction among BH-significant (NA if none) + Wilson CI
  c) Pooled direction-aligned control-adjusted effects by evidence type
  d) Known − Predicted difference with bootstrap 95% CI

Evidence mapping is strict: known_containing / known_independent only.
Aligned effects are computed at cancer×hotspot level, then aggregated across
cancers (cluster bootstrap by cancer).
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
from scipy.stats import fisher_exact, norm

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
OUT_DIR = TWO_ROOT / "known_vs_predicted_comparison"

ARMS: List[Tuple[str, str, int, int, str]] = [
    ("Import_activate", "activate", +1, +1, "Nuclear accumulation\n× activated targets"),
    ("Import_repress", "repress", +1, -1, "Nuclear accumulation\n× repressed targets"),
    ("Export_activate", "activate", -1, +1, "Cytoplasmic redistribution\n× activated targets"),
    ("Export_repress", "repress", -1, -1, "Cytoplasmic redistribution\n× repressed targets"),
]

EVIDENCE_ORDER = ["Known-containing", "Predicted-only"]
EVIDENCE_MAP = {
    "known_containing": "Known-containing",
    "known_independent": "Predicted-only",
}
N_BOOT = 5000
RNG_SEED = 20260929
MIN_N_FOR_FRACTION = 1  # still show point if n>=1; NA only when denominator 0


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


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    z = float(norm.ppf(1.0 - alpha / 2.0))
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    half = (z / denom) * np.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n))
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def _fixed_effect_meta(effects: Sequence[float], ses: Sequence[float]) -> Tuple[float, float, float, float]:
    effects_array = np.asarray(effects, dtype=float)
    ses_array = np.asarray(ses, dtype=float)
    valid = np.isfinite(effects_array) & np.isfinite(ses_array) & (ses_array > 0)
    if not np.any(valid):
        # fall back to unweighted mean if no usable SE
        ev = effects_array[np.isfinite(effects_array)]
        if len(ev) == 0:
            return np.nan, np.nan, np.nan, np.nan
        m = float(np.mean(ev))
        se = float(np.std(ev, ddof=1) / np.sqrt(len(ev))) if len(ev) > 1 else np.nan
        if not np.isfinite(se) or se <= 0:
            return m, np.nan, np.nan, np.nan
        return m, m - 1.96 * se, m + 1.96 * se, se
    ev, sv = effects_array[valid], ses_array[valid]
    w = 1.0 / (sv ** 2)
    pooled = float(np.sum(w * ev) / np.sum(w))
    se = float(np.sqrt(1.0 / np.sum(w)))
    return pooled, pooled - 1.96 * se, pooled + 1.96 * se, se


def map_evidence_type(series: pd.Series) -> pd.Series:
    raw = series.astype(object).where(series.notna(), other=np.nan)
    unknown = sorted({str(v) for v in raw.dropna().unique() if str(v) not in EVIDENCE_MAP})
    if unknown:
        raise ValueError(
            "Unexpected hotspot_class values (strict mapping failed): "
            + ", ".join(unknown)
            + ". Update EVIDENCE_MAP."
        )
    if raw.isna().any():
        raise ValueError("hotspot_class contains missing values; refuse to coerce to Predicted-only.")
    return raw.map(EVIDENCE_MAP)


def load_associations() -> pd.DataFrame:
    frames = []
    class_counts = []
    for arm, reg, s_loc, s_reg, label in ARMS:
        path = (
            TWO_ROOT
            / arm
            / "cptac_filtered"
            / f"two_side_high_low_phospho_comparison_by_site_{arm.lower()}.csv"
        )
        df = pd.read_csv(path, low_memory=False)
        vc = df["hotspot_class"].value_counts(dropna=False).rename_axis("hotspot_class").reset_index(name="n")
        vc["arm"] = arm
        class_counts.append(vc)
        print(f"[{arm}] hotspot_class value_counts:")
        print(df["hotspot_class"].value_counts(dropna=False).to_string())
        df["delta"] = pd.to_numeric(df["delta_high_minus_low"], errors="coerce")
        df["q"] = pd.to_numeric(df["wilcoxon_q_bh"], errors="coerce")
        df["arm"] = arm
        df["category"] = label
        df["align_sign"] = s_loc * s_reg
        df["evidence_type"] = map_evidence_type(df["hotspot_class"])
        df["E_aligned"] = df["delta"] * df["align_sign"]
        if "evaluable" in df.columns:
            df["evaluable"] = df["evaluable"].astype(bool)
        else:
            df["evaluable"] = df["q"].notna()
        # Significant-rate denominators: only n>=10 evaluable associations.
        df["testable"] = df["evaluable"] & df["q"].notna() & df["delta"].notna()
        df["bh_sig"] = df["testable"] & df["q"].lt(0.05)
        df["concordant"] = np.where(
            ~df["bh_sig"] | df["delta"].isna() | df["delta"].eq(0),
            np.nan,
            (np.sign(df["delta"]) == df["align_sign"]).astype(float),
        )
        frames.append(df)
    counts = pd.concat(class_counts, ignore_index=True)
    counts.to_csv(OUT_DIR / "hotspot_class_value_counts.csv", index=False)
    return pd.concat(frames, ignore_index=True)


def analyze_significant_fraction(assoc: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for arm, _, _, _, label in ARMS:
        for ev in EVIDENCE_ORDER:
            sub = assoc[assoc["arm"].eq(arm) & assoc["evidence_type"].eq(ev)]
            n_test = int(sub["testable"].sum())
            n_sig = int(sub["bh_sig"].sum())
            frac = float(n_sig / n_test) if n_test > 0 else np.nan
            lo, hi = wilson_ci(n_sig, n_test) if n_test > 0 else (np.nan, np.nan)
            rows.append(
                {
                    "arm": arm,
                    "category": label,
                    "evidence_type": ev,
                    "n_testable": n_test,
                    "n_bh_significant": n_sig,
                    "significant_fraction": frac,
                    "wilson_ci_low": lo,
                    "wilson_ci_high": hi,
                    "count_label": f"{n_sig}/{n_test}" if n_test > 0 else "NA",
                    "plottable": n_test > 0,
                }
            )
    by_cat = pd.DataFrame(rows)

    fisher_rows = []
    tab = []
    for ev in EVIDENCE_ORDER:
        sub = assoc[assoc["evidence_type"].eq(ev) & assoc["testable"]]
        tab.append([int(sub["bh_sig"].sum()), int((~sub["bh_sig"]).sum())])
    oddsr, p = fisher_exact(np.array(tab), alternative="two-sided")
    fisher_rows.append(
        {
            "scope": "overall",
            "category": "all",
            "odds_ratio": float(oddsr),
            "fisher_p": float(p),
            "known_sig": tab[0][0],
            "known_nonsig": tab[0][1],
            "pred_sig": tab[1][0],
            "pred_nonsig": tab[1][1],
            "note": "exploratory; associations not independent across cancers/hotspots",
        }
    )
    for arm, _, _, _, label in ARMS:
        tab = []
        for ev in EVIDENCE_ORDER:
            sub = assoc[assoc["arm"].eq(arm) & assoc["evidence_type"].eq(ev) & assoc["testable"]]
            tab.append([int(sub["bh_sig"].sum()), int((~sub["bh_sig"]).sum())])
        mat = np.array(tab, dtype=float)
        if mat.sum() == 0 or (mat.sum(axis=1) == 0).any() or (mat.sum(axis=0) == 0).any():
            oddsr, p = np.nan, np.nan
        else:
            oddsr, p = fisher_exact(mat, alternative="two-sided")
        fisher_rows.append(
            {
                "scope": "category",
                "category": label.replace("\n", " "),
                "odds_ratio": float(oddsr) if pd.notna(oddsr) else np.nan,
                "fisher_p": float(p) if pd.notna(p) else np.nan,
                "known_sig": int(mat[0, 0]),
                "known_nonsig": int(mat[0, 1]),
                "pred_sig": int(mat[1, 0]),
                "pred_nonsig": int(mat[1, 1]),
                "note": "exploratory Fisher exact",
            }
        )
    fisher = pd.DataFrame(fisher_rows)
    cat = fisher["scope"].eq("category")
    fisher.loc[cat, "fisher_q_bh"] = _bh_adjust(fisher.loc[cat, "fisher_p"].to_numpy())
    return by_cat, fisher


def analyze_concordant_fraction(assoc: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for arm, _, _, _, label in ARMS:
        for ev in EVIDENCE_ORDER:
            sub = assoc[assoc["arm"].eq(arm) & assoc["evidence_type"].eq(ev) & assoc["bh_sig"]]
            n = int(len(sub))
            n_conc = int((sub["concordant"] == 1).sum())
            n_disc = int((sub["concordant"] == 0).sum())
            if n == 0:
                frac = lo = hi = np.nan
                label_n = "NA"
                plottable = False
                note = "No BH-significant associations"
            else:
                frac = float(n_conc / n)
                lo, hi = wilson_ci(n_conc, n)
                label_n = f"{n_conc}/{n}"
                plottable = True
                note = "Insufficient evidence" if n < 5 else ""
            rows.append(
                {
                    "arm": arm,
                    "category": label,
                    "evidence_type": ev,
                    "n_bh_significant": n,
                    "n_concordant": n_conc,
                    "n_discordant": n_disc,
                    "concordant_fraction": frac,
                    "wilson_ci_low": lo,
                    "wilson_ci_high": hi,
                    "count_label": label_n,
                    "plottable": plottable,
                    "note": note,
                }
            )
    by_cat = pd.DataFrame(rows)

    fisher_rows = []
    tab = []
    for ev in EVIDENCE_ORDER:
        sub = assoc[assoc["evidence_type"].eq(ev) & assoc["bh_sig"]]
        tab.append([int((sub["concordant"] == 1).sum()), int((sub["concordant"] == 0).sum())])
    mat = np.array(tab, dtype=float)
    if mat.sum() == 0 or (mat.sum(axis=1) == 0).any() or (mat.sum(axis=0) == 0).any():
        oddsr, p = np.nan, np.nan
    else:
        oddsr, p = fisher_exact(mat, alternative="two-sided")
    fisher_rows.append(
        {
            "scope": "overall",
            "category": "all",
            "odds_ratio": float(oddsr) if pd.notna(oddsr) else np.nan,
            "fisher_p": float(p) if pd.notna(p) else np.nan,
            "known_concordant": int(mat[0, 0]) if mat.size else 0,
            "known_discordant": int(mat[0, 1]) if mat.size else 0,
            "pred_concordant": int(mat[1, 0]) if mat.size else 0,
            "pred_discordant": int(mat[1, 1]) if mat.size else 0,
            "note": "exploratory; associations not independent",
        }
    )
    for arm, _, _, _, label in ARMS:
        tab = []
        for ev in EVIDENCE_ORDER:
            sub = assoc[assoc["arm"].eq(arm) & assoc["evidence_type"].eq(ev) & assoc["bh_sig"]]
            tab.append([int((sub["concordant"] == 1).sum()), int((sub["concordant"] == 0).sum())])
        mat = np.array(tab, dtype=float)
        if mat.sum() == 0 or (mat.sum(axis=1) == 0).any() or (mat.sum(axis=0) == 0).any():
            oddsr, p = np.nan, np.nan
        else:
            oddsr, p = fisher_exact(mat, alternative="two-sided")
        fisher_rows.append(
            {
                "scope": "category",
                "category": label.replace("\n", " "),
                "odds_ratio": float(oddsr) if pd.notna(oddsr) else np.nan,
                "fisher_p": float(p) if pd.notna(p) else np.nan,
                "known_concordant": int(mat[0, 0]),
                "known_discordant": int(mat[0, 1]),
                "pred_concordant": int(mat[1, 0]),
                "pred_discordant": int(mat[1, 1]),
                "note": "exploratory Fisher exact",
            }
        )
    fisher = pd.DataFrame(fisher_rows)
    cat = fisher["scope"].eq("category")
    fisher.loc[cat, "fisher_q_bh"] = _bh_adjust(fisher.loc[cat, "fisher_p"].to_numpy())
    return by_cat, fisher


def compute_site_level_effects(assoc: pd.DataFrame) -> pd.DataFrame:
    """Control-adjusted aligned effect per cancer × hotspot (site), by evidence type."""
    site_map = assoc.drop_duplicates(["arm", "site_label"])[
        ["arm", "site_label", "evidence_type"]
    ]
    rows = []
    for arm, reg, s_loc, s_reg, label in ARMS:
        gene_path = (
            TWO_ROOT
            / arm
            / "figure4"
            / "figure4c_observed_vs_random"
            / "Figure4c_FigureA_gene_level_observed_and_matched_random_logfc.csv"
        )
        direction = "Import" if arm.startswith("Import") else "Export"
        usecols = [
            "source",
            "direction_short",
            "target_regulation",
            "cancer_type",
            "site_label",
            "directional_logFC",
        ]
        gene = pd.read_csv(gene_path, usecols=usecols, low_memory=False)
        gene = gene[
            gene["direction_short"].astype(str).eq(direction)
            & gene["target_regulation"].astype(str).str.lower().eq(reg)
        ].copy()
        gene["directional_logFC"] = pd.to_numeric(gene["directional_logFC"], errors="coerce")
        gene["site_label"] = gene["site_label"].astype(str)
        gene["cancer_type"] = gene["cancer_type"].astype(str)
        sm = site_map[site_map["arm"].eq(arm)]
        gene = gene.merge(sm, on="site_label", how="inner")
        needs_flip = (s_loc * s_reg) < 0

        for (ev, cancer, site), g in gene.groupby(
            ["evidence_type", "cancer_type", "site_label"], sort=False
        ):
            obs = g.loc[g["source"].eq("Observed targets"), "directional_logFC"].to_numpy(dtype=float)
            rnd = g.loc[g["source"].eq("Matched random genes"), "directional_logFC"].to_numpy(dtype=float)
            obs = obs[np.isfinite(obs)]
            rnd = rnd[np.isfinite(rnd)]
            if len(obs) < 3 or len(rnd) < 3:
                continue
            effect = float(np.mean(obs) - np.mean(rnd))
            # conservative SE treating gene values as approximately independent within site
            se = float(np.sqrt(np.var(obs, ddof=1) / len(obs) + np.var(rnd, ddof=1) / len(rnd)))
            if needs_flip:
                effect = -effect
            rows.append(
                {
                    "arm": arm,
                    "category": label,
                    "evidence_type": ev,
                    "cancer_type": cancer,
                    "site_label": site,
                    "n_observed_genes": int(len(obs)),
                    "n_random_genes": int(len(rnd)),
                    "E_aligned_control": effect,
                    "SE": se if se > 0 else np.nan,
                    "needs_sign_flip": needs_flip,
                }
            )
    return pd.DataFrame(rows)


def pool_by_cancer_then_meta(site_eff: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate site effects within cancer (IVW), then cancer-cluster bootstrap meta."""
    rng = np.random.default_rng(RNG_SEED)
    # cancer-level: inverse-variance weighted mean of hotspot effects (downweights tiny n)
    cancer_rows = []
    for keys, sub in site_eff.groupby(["arm", "category", "evidence_type", "cancer_type"], sort=False):
        arm, label, ev, cancer = keys
        vals = sub["E_aligned_control"].to_numpy(dtype=float)
        ses = sub["SE"].to_numpy(dtype=float)
        n_obs = sub["n_observed_genes"].to_numpy(dtype=float)
        ok = np.isfinite(vals)
        vals, ses, n_obs = vals[ok], ses[ok], n_obs[ok]
        if len(vals) == 0:
            continue
        ivw_ok = np.isfinite(ses) & (ses > 0)
        if ivw_ok.any():
            w = 1.0 / (ses[ivw_ok] ** 2)
            mean = float(np.sum(w * vals[ivw_ok]) / np.sum(w))
            se = float(np.sqrt(1.0 / np.sum(w)))
            agg = "ivw_by_site_se"
        else:
            # fallback: weight by observed gene count
            w = np.maximum(n_obs, 1.0)
            mean = float(np.sum(w * vals) / np.sum(w))
            se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan
            agg = "n_obs_weighted"
        cancer_rows.append(
            {
                "arm": arm,
                "category": label,
                "evidence_type": ev,
                "cancer_type": cancer,
                "n_hotspots": int(len(vals)),
                "E_aligned_control": mean,
                "SE": se,
                "aggregation": agg,
            }
        )
    cancer_df = pd.DataFrame(cancer_rows)
    cancer_df.to_csv(OUT_DIR / "C_cancer_level_from_hotspot_aggregation.csv", index=False)

    summary_rows = []
    diff_rows = []
    for arm, _, _, _, label in ARMS:
        boots_ev: Dict[str, np.ndarray] = {}
        for ev in EVIDENCE_ORDER:
            sub = cancer_df[cancer_df["arm"].eq(arm) & cancer_df["evidence_type"].eq(ev)]
            effects = sub["E_aligned_control"].to_numpy(dtype=float)
            ses = sub["SE"].to_numpy(dtype=float)
            cancers = sub["cancer_type"].astype(str).to_numpy()
            valid = np.isfinite(effects)
            effects, ses, cancers = effects[valid], ses[valid], cancers[valid]
            n = len(effects)
            if n == 0:
                summary_rows.append(
                    {
                        "arm": arm,
                        "category": label,
                        "evidence_type": ev,
                        "n_cancers": 0,
                        "n_hotspot_cancer_units": 0,
                        "pooled_E_aligned": np.nan,
                        "ci_low_bootstrap": np.nan,
                        "ci_high_bootstrap": np.nan,
                        "p_empirical_gt0": np.nan,
                        "note": "no cancer×hotspot units",
                    }
                )
                boots_ev[ev] = np.full(N_BOOT, np.nan)
                continue

            # Cancer-cluster bootstrap of the mean across cancers (equal cancer weight)
            pooled = float(np.mean(effects))
            boots = np.empty(N_BOOT, dtype=float)
            for b in range(N_BOOT):
                idx = rng.integers(0, n, size=n)  # resample cancers
                boots[b] = float(np.mean(effects[idx]))
            finite = boots[np.isfinite(boots)]
            ci_lo = float(np.quantile(finite, 0.025))
            ci_hi = float(np.quantile(finite, 0.975))
            p_emp = float((np.sum(finite <= 0) + 1) / (len(finite) + 1))
            boots_ev[ev] = boots
            n_units = int(
                site_eff[site_eff["arm"].eq(arm) & site_eff["evidence_type"].eq(ev)].shape[0]
            )
            summary_rows.append(
                {
                    "arm": arm,
                    "category": label,
                    "evidence_type": ev,
                    "n_cancers": n,
                    "n_hotspot_cancer_units": n_units,
                    "pooled_E_aligned": pooled,
                    "ci_low_bootstrap": ci_lo,
                    "ci_high_bootstrap": ci_hi,
                    "p_empirical_gt0": p_emp,
                    "note": (
                        "site-level control-adjusted effects; IVW within cancer; "
                        "cancer-cluster bootstrap across cancers"
                    ),
                }
            )

        if all(ev in boots_ev for ev in EVIDENCE_ORDER):
            k = next(r for r in summary_rows if r["arm"] == arm and r["evidence_type"] == "Known-containing")
            p = next(r for r in summary_rows if r["arm"] == arm and r["evidence_type"] == "Predicted-only")
            if np.isfinite(k["pooled_E_aligned"]) and np.isfinite(p["pooled_E_aligned"]):
                diff_obs = float(k["pooled_E_aligned"] - p["pooled_E_aligned"])
            else:
                diff_obs = np.nan
            diff_boot = boots_ev["Known-containing"] - boots_ev["Predicted-only"]
            finite = diff_boot[np.isfinite(diff_boot)]
            if len(finite):
                p_diff = float(min(1.0, 2 * min(np.mean(finite <= 0), np.mean(finite >= 0))))
                d_lo, d_hi = float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))
            else:
                p_diff = d_lo = d_hi = np.nan
            diff_rows.append(
                {
                    "arm": arm,
                    "category": label,
                    "diff_known_minus_pred": diff_obs,
                    "boot_ci_low": d_lo,
                    "boot_ci_high": d_hi,
                    "p_bootstrap_two_sided": p_diff,
                    "ci_excludes_zero": bool(
                        np.isfinite(d_lo) and np.isfinite(d_hi) and (d_lo > 0 or d_hi < 0)
                    ),
                }
            )

    summary = pd.DataFrame(summary_rows)
    for ev in EVIDENCE_ORDER:
        m = summary["evidence_type"].eq(ev)
        summary.loc[m, "p_empirical_q_bh_within_evidence"] = _bh_adjust(
            summary.loc[m, "p_empirical_gt0"].to_numpy()
        )
    diffs = pd.DataFrame(diff_rows)
    if not diffs.empty:
        diffs["p_bootstrap_q_bh"] = _bh_adjust(diffs["p_bootstrap_two_sided"].to_numpy())
    return summary, diffs


def _category_positions() -> Tuple[List[str], np.ndarray]:
    labels = [lab for *_, lab in ARMS]
    return labels, np.arange(len(labels))[::-1]


def plot_fraction_points(
    ax,
    df: pd.DataFrame,
    value_col: str,
    title: str,
    ylabel: str,
    show_legend: bool,
) -> None:
    categories, y_base = _category_positions()
    offset = {"Known-containing": 0.15, "Predicted-only": -0.15}
    colors = {"Known-containing": "#2F4F6F", "Predicted-only": "#C97B49"}

    ax.axvline(0.5, color="#AAAAAA", linestyle=":", linewidth=0.8, zorder=0)
    for ev in EVIDENCE_ORDER:
        ys, xs, elo, ehi = [], [], [], []
        for i, lab in enumerate(categories):
            row = df[df["category"].eq(lab) & df["evidence_type"].eq(ev)]
            if row.empty:
                continue
            r = row.iloc[0]
            yi = y_base[i] + offset[ev]
            if not bool(r["plottable"]) or not np.isfinite(r[value_col]):
                msg = "NA — no BH-significant associations" if value_col.startswith("concordant") else "NA"
                ax.text(
                    0.02,
                    yi,
                    msg,
                    va="center",
                    ha="left",
                    fontsize=6.5,
                    color="#888888",
                    style="italic",
                )
                continue
            e = float(r[value_col])
            lo = float(r["wilson_ci_low"])
            hi = float(r["wilson_ci_high"])
            ys.append(yi)
            xs.append(e)
            elo.append(max(0.0, e - lo))
            ehi.append(max(0.0, hi - e))
            ax.text(hi + 0.02, yi, str(r["count_label"]), va="center", ha="left", fontsize=6.5, color="#333333")
            if r.get("note") == "Insufficient evidence":
                ax.text(
                    e,
                    yi + 0.22,
                    "insufficient evidence",
                    va="bottom",
                    ha="center",
                    fontsize=5.5,
                    color="#888888",
                    style="italic",
                )
        if xs:
            ax.errorbar(
                xs,
                ys,
                xerr=np.vstack([elo, ehi]),
                fmt="o",
                color=colors[ev],
                ecolor=colors[ev],
                elinewidth=1.2,
                capsize=3,
                markersize=5.5,
                label=ev if show_legend else None,
                zorder=3,
            )
    ax.set_yticks(y_base)
    ax.set_yticklabels(categories, fontsize=7)
    ax.set_xlim(-0.05, 1.35)
    ax.set_xlabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_legend:
        ax.legend(frameon=False, fontsize=7, loc="lower right")


def plot_effect_panel(ax, summary: pd.DataFrame, title: str, show_legend: bool) -> None:
    categories, y_base = _category_positions()
    offset = {"Known-containing": 0.15, "Predicted-only": -0.15}
    colors = {"Known-containing": "#2F4F6F", "Predicted-only": "#C97B49"}
    ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0, zorder=0)
    for ev in EVIDENCE_ORDER:
        ys, xs, elo, ehi = [], [], [], []
        for i, lab in enumerate(categories):
            row = summary[summary["category"].eq(lab) & summary["evidence_type"].eq(ev)]
            if row.empty or not np.isfinite(row.iloc[0]["pooled_E_aligned"]):
                continue
            r = row.iloc[0]
            e = float(r["pooled_E_aligned"])
            lo = float(r["ci_low_bootstrap"])
            hi = float(r["ci_high_bootstrap"])
            ys.append(y_base[i] + offset[ev])
            xs.append(e)
            elo.append(max(0.0, e - lo))
            ehi.append(max(0.0, hi - e))
        if xs:
            ax.errorbar(
                xs,
                ys,
                xerr=np.vstack([elo, ehi]),
                fmt="o",
                color=colors[ev],
                ecolor=colors[ev],
                elinewidth=1.2,
                capsize=3,
                markersize=5.5,
                label=ev if show_legend else None,
                zorder=3,
            )
    ax.set_yticks(y_base)
    ax.set_yticklabels(categories, fontsize=7)
    ax.set_xlabel("Pooled direction-aligned effect\nrelative to matched random genes", fontsize=8)
    ax.set_title(title, fontsize=9, pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_legend:
        ax.legend(frameon=False, fontsize=7, loc="lower right")


def plot_diff_panel(ax, diffs: pd.DataFrame, title: str) -> None:
    categories, y_base = _category_positions()
    ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0, zorder=0)
    ys, xs, elo, ehi = [], [], [], []
    for i, lab in enumerate(categories):
        row = diffs[diffs["category"].eq(lab)]
        if row.empty or not np.isfinite(row.iloc[0]["diff_known_minus_pred"]):
            continue
        r = row.iloc[0]
        e = float(r["diff_known_minus_pred"])
        lo = float(r["boot_ci_low"])
        hi = float(r["boot_ci_high"])
        ys.append(y_base[i])
        xs.append(e)
        elo.append(max(0.0, e - lo))
        ehi.append(max(0.0, hi - e))
        q = r.get("p_bootstrap_q_bh", np.nan)
        ax.text(
            hi + 0.002,
            y_base[i],
            f"q={q:.3f}" if np.isfinite(q) else "",
            va="center",
            ha="left",
            fontsize=6.5,
            color="#555555",
        )
    if xs:
        ax.errorbar(
            xs,
            ys,
            xerr=np.vstack([elo, ehi]),
            fmt="D",
            color="#4A2F1F",
            ecolor="#4A2F1F",
            elinewidth=1.3,
            capsize=3,
            markersize=6,
            zorder=3,
        )
    ax.set_yticks(y_base)
    ax.set_yticklabels(categories, fontsize=7)
    ax.set_xlabel("Known − Predicted difference\n(bootstrap 95% CI)", fontsize=8)
    ax.set_title(title, fontsize=9, pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_supplement_four_panels(
    sig_frac: pd.DataFrame,
    conc_frac: pd.DataFrame,
    pooled: pd.DataFrame,
    diffs: pd.DataFrame,
    out_stem: Path,
    dpi: int = 300,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    plot_fraction_points(
        axes[0, 0],
        sig_frac,
        "significant_fraction",
        "a. BH-significant fraction",
        "Significant fraction (Wilson 95% CI)",
        show_legend=True,
    )
    plot_fraction_points(
        axes[0, 1],
        conc_frac,
        "concordant_fraction",
        "b. Concordant fraction among BH-significant",
        "Concordant fraction (Wilson 95% CI)",
        show_legend=False,
    )
    plot_effect_panel(axes[1, 0], pooled, "c. Pooled aligned effects by evidence type", show_legend=True)
    plot_diff_panel(axes[1, 1], diffs, "d. Known − Predicted difference")
    fig.suptitle(
        "Known-containing vs Predicted-only hotspots (supplement)\n"
        "Fisher tests are exploratory; effect CIs use cancer-cluster bootstrap.",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{out_stem}.{ext}", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_interpretation() -> None:
    text = """
INTERPRETATION GUIDE (Part 3)

Safe summary sentence:
Predicted only hotspots showed transcriptional support primarily within the
nuclear accumulation and activated target class, whereas evidence in the other
localization and regulatory classes was weaker or more heterogeneous.

Do NOT claim predicted-only matches known-containing across all four classes.

Panel guidance:
- a: compare significant fractions (denominators matter). Nuclear×activated
  predicted-only has low/no BH hits — do not equate to known significant rate.
- b: NA when no BH-significant associations (not 0% concordance).
- c: positive pooled aligned effect for predicted-only supports distributed
  overall directional support even without many BH hits.
- d: only claim Known vs Predicted differ when bootstrap difference CI excludes
  zero AND BH q is significant. Positive predicted-only effect alone does not
  imply equivalence to known-containing.

Methods notes:
- Evidence mapping is strict: known_containing → Known-containing;
  known_independent → Predicted-only; other values raise an error.
- Fisher exact tests are exploratory (cancer×hotspot associations are not
  independent). A mixed logistic sensitivity analysis
  (BH_sig ~ evidence + category + (1|cancer) + (1|hotspot)) is recommended
  if formal inference is needed.
- Aligned effects: control-adjusted effect per cancer×hotspot (obs − random),
  inverse-variance weighted within cancer, then cancer-cluster bootstrap across
  cancers. Direct Known−Predicted comparisons use panel d / difference table.
""".strip()
    (OUT_DIR / "INTERPRETATION.md").write_text(text + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assoc = load_associations()
    assoc.to_csv(OUT_DIR / "associations_with_evidence_type.csv", index=False)

    sig_frac, sig_fisher = analyze_significant_fraction(assoc)
    sig_frac.to_csv(OUT_DIR / "A_significant_fraction_by_evidence.csv", index=False)
    sig_fisher.to_csv(OUT_DIR / "A_significant_fraction_fisher.csv", index=False)

    conc_frac, conc_fisher = analyze_concordant_fraction(assoc)
    conc_frac.to_csv(OUT_DIR / "B_concordant_fraction_by_evidence.csv", index=False)
    conc_fisher.to_csv(OUT_DIR / "B_concordant_fraction_fisher.csv", index=False)

    print("\nComputing cancer×hotspot control-adjusted effects...")
    site_eff = compute_site_level_effects(assoc)
    site_eff.to_csv(OUT_DIR / "C_site_level_aligned_effects.csv", index=False)

    pooled, diffs = pool_by_cancer_then_meta(site_eff)
    pooled.to_csv(OUT_DIR / "C_pooled_aligned_effects_by_evidence.csv", index=False)
    diffs.to_csv(OUT_DIR / "C_known_minus_predicted_diffs.csv", index=False)

    plot_supplement_four_panels(
        sig_frac,
        conc_frac,
        pooled,
        diffs,
        OUT_DIR / "supplement_known_vs_predicted_four_panels",
    )
    # keep standalone grouped forest as panel-c style export
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    plot_effect_panel(ax, pooled, "Pooled aligned effects by evidence type", show_legend=True)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            f"{OUT_DIR / 'C_grouped_forest_known_vs_predicted'}.{ext}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)

    write_interpretation()

    # remove obsolete bar chart if present
    for obsolete in (
        "AB_significant_and_concordant_fractions.png",
        "AB_significant_and_concordant_fractions.pdf",
        "AB_significant_and_concordant_fractions.svg",
    ):
        p = OUT_DIR / obsolete
        if p.exists():
            p.unlink()

    print("\n=== A significant fraction ===")
    print(sig_frac[["category", "evidence_type", "count_label", "significant_fraction", "wilson_ci_low", "wilson_ci_high"]].to_string(index=False))
    print("\n=== B concordant fraction ===")
    print(conc_frac[["category", "evidence_type", "count_label", "concordant_fraction", "note"]].to_string(index=False))
    print("\n=== C pooled effects ===")
    print(pooled[["category", "evidence_type", "n_cancers", "pooled_E_aligned", "ci_low_bootstrap", "ci_high_bootstrap", "p_empirical_gt0", "p_empirical_q_bh_within_evidence"]].to_string(index=False))
    print("\n=== D Known − Predicted ===")
    print(diffs.to_string(index=False))
    print(f"\nWrote outputs under {OUT_DIR}")


if __name__ == "__main__":
    main()
