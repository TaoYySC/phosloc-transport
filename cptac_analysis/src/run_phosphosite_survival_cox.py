#!/usr/bin/env python3
"""
CPTAC phosphosite survival analysis (v4).

Primary evidence:
  Model 1 Univariate Cox:     phospho
  Model 2 Protein-adjusted:   phospho + TF protein

Supportive clinical:
  Model 3s: phospho + TF protein + stage_binary (Early I/II vs Advanced III/IV)
  only when events are sufficient (event-adaptive depth)

Sensitivity:
  Firth Cox (bias-reduced) and bootstrap HR-direction stability on primary hits

Event-adaptive depth (pre-specified):
  <10 events  : no Cox (descriptive only)
  10-19       : Model 1 only
  20-29       : Model 1 + Model 2
  >=30        : Model 1 + Model 2 + Model 3s (reduced clinical)

Phospho high/low MUST match boxplot split (median_nonmissing).
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import inv, LinAlgError
from scipy.optimize import minimize

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test
except Exception:  # pragma: no cover
    CoxPHFitter = None  # type: ignore[assignment]
    KaplanMeierFitter = None  # type: ignore[assignment]
    logrank_test = None  # type: ignore[assignment]

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from run_import_target_regulation_analysis import TargetRegulationBoxplotPipeline, TempoConfig

warnings.filterwarnings("ignore", category=RuntimeWarning)

DEFAULT_BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SIGNIFICANT_CSV = (
    DEFAULT_BASE_DIR
    / "results/import_target_regulation/high_low_phospho_boxplots/all_significant_sites_combined/"
    / "all_cancers_Import_activate_bh_significant_sites_table.csv"
)
DEFAULT_POINTS_CSV = (
    DEFAULT_BASE_DIR
    / "results/import_target_regulation/high_low_phospho_boxplots/"
    / "target_gene_high_low_expression_points_plotted.csv"
)
EXPECTED_PHOSPHO_SPLIT_MODE = "median_nonmissing"
DEFAULT_CANCER_ENDPOINTS: Dict[str, str] = {
    "BRCA": "OS", "CCRCC": "OS", "COAD": "OS", "GBM": "OS", "HNSCC": "OS",
    "LSCC": "OS", "LUAD": "OS", "OV": "PFS", "PDAC": "OS", "UCEC": "PFS",
}


@dataclass(frozen=True)
class CoxLayerSpec:
    model_name: str
    covariates: Tuple[str, ...]
    role: str  # primary / supportive


def bh_fdr(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if valid.sum() == 0:
        return q
    p_valid = p[valid]
    m = len(p_valid)
    order = np.argsort(p_valid)
    ranks = np.arange(1, m + 1, dtype=float)
    q_valid = (p_valid[order] * m) / ranks
    q_valid = np.minimum.accumulate(q_valid[::-1])[::-1]
    q_out = np.full(m, np.nan, dtype=float)
    q_out[order] = q_valid
    q[valid] = q_out
    return q


def _safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_survival(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=0).replace({"NA": np.nan, "": np.nan})
    if "case_id" not in df.columns:
        raise ValueError(f"Missing case_id in {path}")
    return df.set_index("case_id")


def load_phenotype_meta(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=0, index_col=0).replace({"NA": np.nan, "": np.nan})
    if "data_type" in df.index:
        df = df.drop(index=["data_type"])
    return df


def normalize_ensembl_id_for_proteins(idx: Iterable[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for raw in idx:
        if not isinstance(raw, str):
            continue
        base = raw.split(".", 1)[0]
        if base not in mapping:
            mapping[base] = raw
    return mapping


def extract_tf_protein_values(
    *,
    cancer_protein_df: pd.DataFrame,
    tf_gene_id: str,
    samples: Sequence[str],
) -> pd.Series:
    if tf_gene_id in cancer_protein_df.index:
        row_key = tf_gene_id
    else:
        mapping = normalize_ensembl_id_for_proteins(cancer_protein_df.index)
        if tf_gene_id not in mapping:
            return pd.Series(index=list(samples), dtype=float, data=np.nan)
        row_key = mapping[tf_gene_id]
    out = pd.to_numeric(cancer_protein_df.loc[row_key, list(samples)], errors="coerce")
    out.index = list(samples)
    return out


def _normalize_label(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.upper() == "NA":
        return None
    return text


def _stage_or_grade_to_advanced(value: object) -> float:
    text = _normalize_label(value)
    if text is None:
        return np.nan
    upper = text.upper()
    if "IV" in upper or "STAGE 4" in upper:
        return 1.0
    if "III" in upper or "STAGE 3" in upper:
        return 1.0
    if "II" in upper or "STAGE 2" in upper:
        return 0.0
    if "I" in upper or "STAGE 1" in upper:
        return 0.0
    if "G4" in upper or "G3" in upper or "POOR" in upper:
        return 1.0
    if "G2" in upper or "G1" in upper or "WELL" in upper or "MODER" in upper:
        return 0.0
    digits = pd.Series([text]).str.extract(r"(\d+)", expand=False).iloc[0]
    if pd.notna(digits):
        num = float(digits)
        if num >= 3:
            return 1.0
        if num in (1, 2):
            return 0.0
    return np.nan


def build_clinical_covariates(
    phenotype_df: pd.DataFrame,
    samples: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Keep stage_binary for supportive Model 3s; also keep age/sex for optional use."""
    sample_index = pd.Index(list(samples), name="case_id")
    out = pd.DataFrame(index=sample_index)
    meta: Dict[str, str] = {"clinical_mode": "stage_binary_early_advanced"}

    age = _safe_to_numeric(
        phenotype_df["Age"] if "Age" in phenotype_df.columns else pd.Series(index=phenotype_df.index, data=np.nan)
    ).reindex(sample_index)
    out["age"] = age

    sex_raw = (
        phenotype_df["Sex"] if "Sex" in phenotype_df.columns else pd.Series(index=phenotype_df.index, data=np.nan)
    ).reindex(sample_index)
    out["sex_raw"] = sex_raw
    out["sex_male"] = np.where(
        sex_raw.astype(str).eq("Male"), 1.0, np.where(sex_raw.astype(str).eq("Female"), 0.0, np.nan)
    )

    stage_raw = (
        phenotype_df["Stage"] if "Stage" in phenotype_df.columns else pd.Series(index=phenotype_df.index, data=np.nan)
    ).reindex(sample_index)
    grade_raw = (
        phenotype_df["Histologic_Grade"]
        if "Histologic_Grade" in phenotype_df.columns
        else pd.Series(index=phenotype_df.index, data=np.nan)
    ).reindex(sample_index)
    out["stage_raw"] = stage_raw
    out["grade_raw"] = grade_raw

    stage_adv = stage_raw.map(_stage_or_grade_to_advanced)
    missing = stage_adv.isna()
    if missing.any():
        stage_adv.loc[missing] = grade_raw.loc[missing].map(_stage_or_grade_to_advanced)
    out["stage_advanced"] = stage_adv
    if stage_adv.notna().sum() == 0:
        meta["clinical_mode"] = "no_stage_available"
    return out, meta


def get_phospho_split_from_pipeline(
    pipeline: TargetRegulationBoxplotPipeline,
    *,
    site: str,
    df_phospho: pd.DataFrame,
    df_rna: pd.DataFrame,
    df_protein: pd.DataFrame,
) -> Dict[str, object]:
    return pipeline._get_phospho_high_low_samples(
        site, df_phospho, df_rna, df_protein=df_protein, phospho_value_mode=pipeline.config.phospho_value_mode
    )


def extract_site_phospho_values(
    *,
    df_phospho: pd.DataFrame,
    df_rna: pd.DataFrame,
    site: str,
    samples: Sequence[str],
) -> pd.Series:
    matched_samples = [s for s in df_rna.columns if s in df_phospho.columns]
    if site not in df_phospho.index:
        return pd.Series(index=list(samples), dtype=float, data=np.nan)
    return pd.to_numeric(df_phospho.loc[site, matched_samples], errors="coerce").reindex(samples)


def validate_split_against_expected(
    split_info: Dict[str, object],
    expected: pd.Series,
    *,
    median_tol: float = 1e-6,
) -> Tuple[bool, str]:
    if str(expected.get("phospho_split_mode", "")) != EXPECTED_PHOSPHO_SPLIT_MODE:
        return False, "unexpected_phospho_split_mode_in_points_csv"
    if split_info.get("status") != "success":
        return False, str(split_info.get("status", "split_failed"))
    rec_n_low = int(split_info.get("n_low_samples", -1))
    rec_n_high = int(split_info.get("n_high_samples", -1))
    if rec_n_low != int(expected["n_low_samples"]) or rec_n_high != int(expected["n_high_samples"]):
        return False, "split_mismatch_expected_vs_reconstructed"
    rec_median = split_info.get("median_cutoff", np.nan)
    if np.isfinite(rec_median) and abs(float(rec_median) - float(expected["median_cutoff"])) > median_tol:
        return False, "split_mismatch_expected_vs_reconstructed"
    return True, "success"


def endpoint_columns(endpoint: str) -> Tuple[str, str]:
    if endpoint == "OS":
        return "OS_days", "OS_event"
    if endpoint == "PFS":
        return "PFS_days", "PFS_event"
    raise ValueError(endpoint)


def count_endpoint_events(survival_df: pd.DataFrame, samples: Sequence[str], endpoint: str) -> Tuple[int, int]:
    time_col, event_col = endpoint_columns(endpoint)
    if time_col not in survival_df.columns or event_col not in survival_df.columns:
        return 0, 0
    sub = survival_df.reindex(list(samples))[[time_col, event_col]].copy()
    sub[time_col] = _safe_to_numeric(sub[time_col])
    sub[event_col] = _safe_to_numeric(sub[event_col])
    sub = sub.dropna()
    if sub.empty:
        return 0, 0
    return int(len(sub)), int(sub[event_col].sum())


def choose_site_primary_endpoint(
    survival_df: pd.DataFrame,
    samples: Sequence[str],
    *,
    cancer_type: str,
    cancer_fallback: Dict[str, str],
) -> Tuple[str, Dict[str, object]]:
    n_os, e_os = count_endpoint_events(survival_df, samples, "OS")
    n_pfs, e_pfs = count_endpoint_events(survival_df, samples, "PFS")
    meta: Dict[str, object] = {"os_n": n_os, "os_events": e_os, "pfs_n": n_pfs, "pfs_events": e_pfs}
    if e_os == 0 and e_pfs == 0:
        meta["selection_rule"] = "fallback_zero_events"
        return cancer_fallback.get(cancer_type, "OS"), meta
    if e_pfs > e_os:
        meta["selection_rule"] = "more_events_pfs"
        return "PFS", meta
    if e_os > e_pfs:
        meta["selection_rule"] = "more_events_os"
        return "OS", meta
    meta["selection_rule"] = "tie_prefer_os"
    return "OS", meta


def analysis_depth_from_events(n_events: int, min_events_m1: int = 10) -> str:
    """
    Pre-specified event-adaptive depth.
    <min_events_m1: none; min_events_m1-19: M1; 20-29: M1+M2; >=30: M1+M2+M3s
    """
    if n_events < min_events_m1:
        return "none"
    if n_events < 20:
        return "m1"
    if n_events < 30:
        return "m1_m2"
    return "m1_m2_m3s"


def allowed_models_for_depth(depth: str) -> List[str]:
    if depth == "none":
        return []
    if depth == "m1":
        return ["univariate"]
    if depth == "m1_m2":
        return ["univariate", "protein_adjusted"]
    if depth == "m1_m2_m3s":
        return ["univariate", "protein_adjusted", "clinical_supportive"]
    return []


def fit_ordinary_cox(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    covariates: Sequence[str],
    *,
    penalizer: float = 0.0,
) -> Dict[str, object]:
    if CoxPHFitter is None:
        raise RuntimeError("lifelines is required")
    use_cols = [time_col, event_col] + list(covariates)
    sub = df[use_cols].dropna()
    if sub.empty:
        return {"status": "insufficient_data_after_na_drop"}
    events = int(sub[event_col].sum())
    if events <= 0:
        return {"status": "no_events"}
    cph = CoxPHFitter(penalizer=penalizer)
    try:
        cph.fit(sub, duration_col=time_col, event_col=event_col, show_progress=False)
    except Exception as exc:
        return {"status": "cox_fit_error", "error": str(exc)}
    if "phospho_binary" not in cph.params_.index:
        return {"status": "missing_phospho_coef"}
    beta = float(cph.params_["phospho_binary"])
    ci = cph.confidence_intervals_.loc["phospho_binary"]
    return {
        "status": "success",
        "n": int(len(sub)),
        "events": events,
        "HR_phospho": float(np.exp(beta)),
        "CI_low": float(np.exp(ci.iloc[0])),
        "CI_high": float(np.exp(ci.iloc[1])),
        "p_raw": float(cph.summary.loc["phospho_binary", "p"]),
        "n_covariates": int(len(covariates)),
        "n_dropped_by_na": int(df.shape[0] - sub.shape[0]),
        "method": "ordinary" if penalizer == 0 else f"ridge_penalizer_{penalizer}",
    }


def _cox_partial_ll_and_info(
    beta: np.ndarray,
    X: np.ndarray,
    event: np.ndarray,
    time: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Breslow partial log-likelihood and observed information for Cox model.
    Samples must be sorted by time ascending.
    """
    p = beta.shape[0]
    eta = X @ beta
    # numerically stable risk-set accumulation from the end
    exp_eta = np.exp(np.clip(eta, -20, 20))
    # reverse cumulative sum of exp(eta)
    risk_sum = np.cumsum(exp_eta[::-1])[::-1]
    ll = 0.0
    score = np.zeros(p)
    info = np.zeros((p, p))

    # Process unique event times
    event_idx = np.where(event == 1)[0]
    for i in event_idx:
        t = time[i]
        # risk set: time >= t
        risk = time >= t
        w = exp_eta[risk]
        s0 = w.sum()
        if s0 <= 0:
            continue
        xr = X[risk]
        s1 = (xr * w[:, None]).sum(axis=0)
        s2 = (xr.T * w) @ xr
        ll += float(eta[i] - np.log(s0))
        mean = s1 / s0
        score += X[i] - mean
        info += s2 / s0 - np.outer(mean, mean)
    return ll, info


def fit_firth_cox(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    covariates: Sequence[str],
) -> Dict[str, object]:
    """
    Firth-penalized Cox: maximize l(beta) + 0.5 * log|I(beta)|.
    Suitable as sensitivity for small-sample / rare-event settings.
    """
    use_cols = [time_col, event_col] + list(covariates)
    sub = df[use_cols].dropna().copy()
    if sub.empty:
        return {"status": "insufficient_data_after_na_drop"}
    events = int(sub[event_col].sum())
    if events <= 0:
        return {"status": "no_events"}
    if "phospho_binary" not in covariates:
        return {"status": "missing_phospho"}

    sub = sub.sort_values(time_col, kind="mergesort")
    X = sub[list(covariates)].to_numpy(dtype=float)
    time = sub[time_col].to_numpy(dtype=float)
    event = sub[event_col].to_numpy(dtype=float)
    p = X.shape[1]
    phospho_idx = list(covariates).index("phospho_binary")

    def objective(beta: np.ndarray) -> float:
        ll, info = _cox_partial_ll_and_info(beta, X, event, time)
        try:
            sign, logdet = np.linalg.slogdet(info + 1e-8 * np.eye(p))
            if sign <= 0:
                return 1e6
            return -(ll + 0.5 * logdet)
        except LinAlgError:
            return 1e6

    # start from ordinary MLE if possible
    start = np.zeros(p)
    ord_res = fit_ordinary_cox(df, time_col, event_col, covariates)
    if ord_res.get("status") == "success":
        # rough start: logHR on phospho, zeros elsewhere
        start[phospho_idx] = float(np.log(max(ord_res["HR_phospho"], 1e-6)))

    try:
        opt = minimize(objective, start, method="BFGS", options={"maxiter": 200, "disp": False})
    except Exception as exc:
        return {"status": "firth_fit_error", "error": str(exc)}

    if not opt.success and not np.isfinite(opt.fun):
        return {"status": "firth_fit_error", "error": str(opt.message)}

    beta = opt.x
    ll, info = _cox_partial_ll_and_info(beta, X, event, time)
    try:
        cov = inv(info + 1e-8 * np.eye(p))
        se = float(np.sqrt(max(cov[phospho_idx, phospho_idx], 0.0)))
    except LinAlgError:
        se = np.nan

    b = float(beta[phospho_idx])
    hr = float(np.exp(b))
    if np.isfinite(se) and se > 0:
        ci_low = float(np.exp(b - 1.96 * se))
        ci_high = float(np.exp(b + 1.96 * se))
        from scipy.stats import norm
        p_raw = float(2 * (1 - norm.cdf(abs(b / se))))
    else:
        ci_low = ci_high = p_raw = np.nan

    return {
        "status": "success",
        "n": int(len(sub)),
        "events": events,
        "HR_phospho": hr,
        "CI_low": ci_low,
        "CI_high": ci_high,
        "p_raw": p_raw,
        "n_covariates": p,
        "method": "firth",
    }


def bootstrap_hr_direction(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    covariates: Sequence[str],
    *,
    n_boot: int = 500,
    seed: int = 42,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    use_cols = [time_col, event_col] + list(covariates)
    sub = df[use_cols].dropna()
    if sub.empty or int(sub[event_col].sum()) <= 0:
        return {"status": "insufficient_data", "n_boot_success": 0}

    hrs: List[float] = []
    n = len(sub)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot = sub.iloc[idx]
        if int(boot[event_col].sum()) <= 0:
            continue
        res = fit_ordinary_cox(boot, time_col, event_col, covariates)
        if res.get("status") == "success" and np.isfinite(res.get("HR_phospho", np.nan)):
            hrs.append(float(res["HR_phospho"]))

    if not hrs:
        return {"status": "no_successful_boot", "n_boot_success": 0}
    arr = np.asarray(hrs, dtype=float)
    return {
        "status": "success",
        "n_boot_success": int(len(arr)),
        "hr_median": float(np.median(arr)),
        "hr_ci_low": float(np.quantile(arr, 0.025)),
        "hr_ci_high": float(np.quantile(arr, 0.975)),
        "frac_hr_gt1": float(np.mean(arr > 1.0)),
        "frac_hr_lt1": float(np.mean(arr < 1.0)),
    }


def apply_stratified_fdr(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    out = df.copy()
    if "q_bh" not in out.columns:
        out["q_bh"] = np.nan
    group_cols = ["cancer_type", "model", "endpoint"] if scope == "per_cancer" else ["model", "endpoint"]
    for _, sub_idx in out.groupby(group_cols).groups.items():
        sub = out.loc[sub_idx]
        success_mask = sub["status"].eq("success")
        if "method" in sub.columns:
            success_mask = success_mask & sub["method"].fillna("ordinary").eq("ordinary")
        if not success_mask.any():
            continue
        p = sub.loc[success_mask, "p_raw"].astype(float)
        q = bh_fdr(p.tolist())
        out.loc[sub.index[success_mask], "q_bh"] = q
    return out


def _p_to_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def make_km_plot(
    sample_df: pd.DataFrame,
    *,
    cancer_type: str,
    site: str,
    site_label: str,
    endpoint: str,
    out_path: Path,
    cox_hr: Optional[float] = None,
    cox_ci_low: Optional[float] = None,
    cox_ci_high: Optional[float] = None,
    cox_p: Optional[float] = None,
    cox_q: Optional[float] = None,
) -> Optional[float]:
    if KaplanMeierFitter is None or logrank_test is None:
        return None
    time_col, event_col = endpoint_columns(endpoint)
    sub = sample_df.loc[sample_df["cancer_type"].eq(cancer_type) & sample_df["site"].eq(site)].copy()
    if sub.empty:
        return None
    sub[time_col] = _safe_to_numeric(sub[time_col])
    sub[event_col] = _safe_to_numeric(sub[event_col])
    sub = sub.dropna(subset=[time_col, event_col, "phospho_group"])
    if not {"low", "high"}.issubset(set(sub["phospho_group"].astype(str))):
        return None
    low = sub.loc[sub["phospho_group"].eq("low")]
    high = sub.loc[sub["phospho_group"].eq("high")]
    if len(low) < 3 or len(high) < 3:
        return None
    try:
        lr = logrank_test(low[time_col], high[time_col], event_observed_A=low[event_col], event_observed_B=high[event_col])
        logrank_p = float(lr.p_value)
    except Exception:
        logrank_p = np.nan

    colors = {"low": "#1B6CA8", "high": "#C0392B"}

    fig, ax = plt.subplots(figsize=(6.4, 4.6), facecolor="white")
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.12)

    tmax = float(sub[time_col].max())
    for group, label in [("low", "Low phospho"), ("high", "High phospho")]:
        g = sub.loc[sub["phospho_group"].eq(group)]
        kmf = KaplanMeierFitter()
        kmf.fit(g[time_col], event_observed=g[event_col], label=f"{label} (n={len(g)})")
        kmf.plot_survival_function(
            ax=ax, color=colors[group], ci_show=True, ci_alpha=0.14, linewidth=2.4
        )

    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, tmax * 1.02)
    ax.set_ylabel("Survival probability")
    ax.set_xlabel("Time (days)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.35)
    ax.set_axisbelow(True)

    fig.text(
        0.545,
        0.955,
        f"{cancer_type}  ·  {str(site_label).replace('_', ' ')}",
        ha="center",
        va="top",
        fontsize=13,
        fontweight="regular",
        color="#222222",
    )

    # Prefer BH q, then Cox p, then log-rank p for asterisk annotation
    sig_p = np.nan
    for candidate in (cox_q, cox_p, logrank_p):
        if candidate is not None and np.isfinite(float(candidate)):
            sig_p = float(candidate)
            break
    stars = _p_to_stars(sig_p)
    if stars:
        ax.text(
            0.98,
            0.98,
            stars,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=16,
            fontweight="bold",
            color="#222222",
        )

    handles, labels = ax.get_legend_handles_labels()
    keep, seen = [], set()
    for h, lab in zip(handles, labels):
        key = lab.split(" (n=")[0]
        if key in {"Low phospho", "High phospho"} and key not in seen:
            keep.append((h, lab))
            seen.add(key)
    if keep:
        ax.legend([k[0] for k in keep], [k[1] for k in keep], loc="lower left", frameon=False)

    step = 500 if tmax >= 1200 else 250
    times = list(np.arange(0, ax.get_xlim()[1] + 1e-9, step))
    ax.set_xticks(times)

    # silence unused CI args (kept for call-site compatibility)
    _ = (cox_hr, cox_ci_low, cox_ci_high)

    fig.savefig(out_path, dpi=350, bbox_inches="tight", facecolor="white")
    if out_path.suffix.lower() == ".png":
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return logrank_p


def build_evidence_table(adj_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    keys = adj_df[["cancer_type", "site", "site_label"]].drop_duplicates()
    rows: List[Dict[str, object]] = []
    for _, key in keys.iterrows():
        c, site, label = key["cancer_type"], key["site"], key["site_label"]
        sub = adj_df.loc[adj_df["cancer_type"].eq(c) & adj_df["site"].eq(site)]
        row: Dict[str, object] = {"cancer_type": c, "site": site, "site_label": label}
        for model in ["univariate", "protein_adjusted", "clinical_supportive"]:
            msub = sub.loc[sub["model"].eq(model)]
            if "method" in sub.columns:
                msub = msub.loc[msub["method"].fillna("ordinary").eq("ordinary")]
            if msub.empty:
                row[f"{model}_status"] = "missing"
                for k in ["endpoint", "HR", "p_raw", "q_bh", "events", "n", "analysis_depth"]:
                    row[f"{model}_{k}"] = np.nan
                continue
            r = msub.iloc[0]
            row[f"{model}_status"] = r.get("status")
            row[f"{model}_endpoint"] = r.get("endpoint")
            row[f"{model}_HR"] = r.get("HR_phospho")
            row[f"{model}_p_raw"] = r.get("p_raw")
            row[f"{model}_q_bh"] = r.get("q_bh")
            row[f"{model}_events"] = r.get("events")
            row[f"{model}_n"] = r.get("n")
            row[f"{model}_analysis_depth"] = r.get("analysis_depth")
            row["primary_endpoint"] = r.get("endpoint")
            row["analysis_depth"] = r.get("analysis_depth")

        def sig(model: str, use_q: bool) -> bool:
            if row.get(f"{model}_status") != "success":
                return False
            val = row.get(f"{model}_q_bh" if use_q else f"{model}_p_raw")
            try:
                return float(val) <= 0.05 if use_q else float(val) < 0.05
            except Exception:
                return False

        row["m1_raw_sig"] = sig("univariate", False)
        row["m1_bh_sig"] = sig("univariate", True)
        row["m2_raw_sig"] = sig("protein_adjusted", False)
        row["m2_bh_sig"] = sig("protein_adjusted", True)
        row["m3s_raw_sig"] = sig("clinical_supportive", False)
        row["m3s_bh_sig"] = sig("clinical_supportive", True)

        if row["m1_bh_sig"] and row["m2_bh_sig"] and row["m3s_bh_sig"]:
            row["evidence_class"] = "M1_M2_M3s"
        elif row["m1_bh_sig"] and row["m2_bh_sig"]:
            row["evidence_class"] = "M1_M2"
        elif row["m1_bh_sig"]:
            row["evidence_class"] = "M1_only"
        elif row["m2_bh_sig"]:
            row["evidence_class"] = "M2_only"
        else:
            row["evidence_class"] = "none"
        rows.append(row)

    evidence = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "layer": "M1_univariate_primary",
                "n_success": int((evidence.get("univariate_status") == "success").sum()) if len(evidence) else 0,
                "n_raw_p_lt_0.05": int(evidence["m1_raw_sig"].sum()) if len(evidence) else 0,
                "n_bh_q_le_0.05": int(evidence["m1_bh_sig"].sum()) if len(evidence) else 0,
            },
            {
                "layer": "M2_protein_adjusted_primary",
                "n_success": int((evidence.get("protein_adjusted_status") == "success").sum()) if len(evidence) else 0,
                "n_raw_p_lt_0.05": int(evidence["m2_raw_sig"].sum()) if len(evidence) else 0,
                "n_bh_q_le_0.05": int(evidence["m2_bh_sig"].sum()) if len(evidence) else 0,
            },
            {
                "layer": "M3s_clinical_supportive_stage_binary",
                "n_success": int((evidence.get("clinical_supportive_status") == "success").sum()) if len(evidence) else 0,
                "n_raw_p_lt_0.05": int(evidence["m3s_raw_sig"].sum()) if len(evidence) else 0,
                "n_bh_q_le_0.05": int(evidence["m3s_bh_sig"].sum()) if len(evidence) else 0,
            },
        ]
    )
    return evidence, summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v4 event-adaptive Cox + Firth/bootstrap sensitivity")
    p.add_argument("--significant-csv", type=Path, default=DEFAULT_SIGNIFICANT_CSV)
    p.add_argument("--points-csv", type=Path, default=DEFAULT_POINTS_CSV)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_BASE_DIR / "results/survival_analysis/phosphosite_cox_3tier_v4",
    )
    p.add_argument("--endpoint-mode", choices=["per_site", "cancer_specific"], default="per_site")
    p.add_argument("--fdr-scope", choices=["per_cancer", "global_model_endpoint"], default="per_cancer")
    p.add_argument("--min-group-samples", type=int, default=3)
    p.add_argument(
        "--min-events-m1",
        type=int,
        default=10,
        help="Minimum events to allow univariate (M1) Cox; default 10 (EPV≈10).",
    )
    p.add_argument("--ridge-penalizer", type=float, default=0.1)
    p.add_argument("--n-bootstrap", type=int, default=500)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument("--make-forest-plots", action="store_true")
    p.add_argument("--no-km-plots", action="store_true")
    p.add_argument("--max-sites", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    km_dir = out_dir / "km_curves"
    km_dir.mkdir(parents=True, exist_ok=True)
    sens_dir = out_dir / "sensitivity"
    sens_dir.mkdir(parents=True, exist_ok=True)

    if CoxPHFitter is None:
        raise RuntimeError("lifelines required")

    config = TempoConfig(
        phospho_split_mode=EXPECTED_PHOSPHO_SPLIT_MODE,
        phospho_value_mode="site_abundance",
        exclude_zero_phospho_for_split=False,
        min_group_samples=args.min_group_samples,
    )
    pipeline = TargetRegulationBoxplotPipeline(config)

    df_sig = pd.read_csv(args.significant_csv)
    if "target_regulation" in df_sig.columns:
        df_sig = df_sig.loc[df_sig["target_regulation"].astype(str).eq("activate")].copy()
    df_points = pd.read_csv(args.points_csv)
    split_map = df_points.drop_duplicates(subset=["cancer_type", "site"]).set_index(["cancer_type", "site"])

    cache: Dict[str, Dict[str, object]] = {}

    def load_cancer(cancer_type: str) -> None:
        if cancer_type in cache:
            return
        src = Path(config.linkedomics_base) / cancer_type
        cache[cancer_type] = {
            "phospho": pipeline.load_phospho(cancer_type),
            "rna": pipeline.load_rna(cancer_type),
            "protein": pipeline.load_protein(cancer_type),
            "phenotype": load_phenotype_meta(src / f"{cancer_type}_meta.txt"),
            "survival": load_survival(src / f"{cancer_type}_survival.txt"),
        }

    site_entries = (
        df_sig.drop_duplicates(subset=["cancer_type", "site"])[["cancer_type", "site", "site_label"]]
        .to_dict(orient="records")
    )
    if args.max_sites > 0:
        site_entries = site_entries[: args.max_sites]

    sample_rows: List[pd.DataFrame] = []
    result_rows: List[Dict[str, object]] = []
    endpoint_rows: List[Dict[str, object]] = []
    depth_rows: List[Dict[str, object]] = []

    layers = [
        CoxLayerSpec("univariate", ("phospho_binary",), "primary"),
        CoxLayerSpec("protein_adjusted", ("phospho_binary", "tf_protein"), "primary"),
        CoxLayerSpec("clinical_supportive", ("phospho_binary", "tf_protein", "stage_advanced"), "supportive"),
    ]

    for entry in site_entries:
        cancer_type = str(entry["cancer_type"])
        site = str(entry["site"])
        site_label = str(entry.get("site_label", site))

        if (cancer_type, site) not in split_map.index:
            result_rows.append(
                {"cancer_type": cancer_type, "site": site, "site_label": site_label, "model": "skip", "status": "missing_split_map"}
            )
            continue

        expected = split_map.loc[(cancer_type, site)]
        tf_gene_id = str(expected["tf_gene_id"])
        tf_name = str(expected.get("tf_name", ""))
        load_cancer(cancer_type)
        df_phospho = cache[cancer_type]["phospho"]  # type: ignore
        df_rna = cache[cancer_type]["rna"]  # type: ignore
        df_protein = cache[cancer_type]["protein"]  # type: ignore
        phenotype = cache[cancer_type]["phenotype"]  # type: ignore
        survival = cache[cancer_type]["survival"]  # type: ignore

        split_info = get_phospho_split_from_pipeline(
            pipeline, site=site, df_phospho=df_phospho, df_rna=df_rna, df_protein=df_protein
        )
        ok, qc = validate_split_against_expected(split_info, expected)
        if not ok:
            result_rows.append(
                {
                    "cancer_type": cancer_type,
                    "site": site,
                    "site_label": site_label,
                    "model": "skip",
                    "status": qc,
                }
            )
            continue

        low_samples = list(split_info["low_samples"])
        high_samples = list(split_info["high_samples"])
        all_samples = low_samples + high_samples

        if args.endpoint_mode == "cancer_specific":
            endpoint = DEFAULT_CANCER_ENDPOINTS.get(cancer_type, "OS")
            ep_meta = {"selection_rule": "cancer_specific"}
        else:
            endpoint, ep_meta = choose_site_primary_endpoint(
                survival, all_samples, cancer_type=cancer_type, cancer_fallback=DEFAULT_CANCER_ENDPOINTS
            )
        endpoint_rows.append(
            {"cancer_type": cancer_type, "site": site, "site_label": site_label, "primary_endpoint": endpoint, **ep_meta}
        )

        phospho_log2 = extract_site_phospho_values(df_phospho=df_phospho, df_rna=df_rna, site=site, samples=all_samples)
        tf_protein = extract_tf_protein_values(cancer_protein_df=df_protein, tf_gene_id=tf_gene_id, samples=all_samples)
        clinical_df, clinical_meta = build_clinical_covariates(phenotype, all_samples)

        # sample table
        rows = []
        for grp, samples in [("low", low_samples), ("high", high_samples)]:
            for s in samples:
                row = {
                    "cancer_type": cancer_type,
                    "site": site,
                    "site_label": site_label,
                    "tf_name": tf_name,
                    "tf_gene_id": tf_gene_id,
                    "primary_endpoint": endpoint,
                    "sample": s,
                    "phospho_group": grp,
                    "phospho_binary": 0 if grp == "low" else 1,
                    "phospho_log2": phospho_log2.get(s, np.nan),
                    "tf_protein": tf_protein.get(s, np.nan),
                }
                if s in clinical_df.index:
                    for col in clinical_df.columns:
                        row[col] = clinical_df.loc[s, col]
                if s in survival.index:
                    for col in ["OS_days", "OS_event", "PFS_days", "PFS_event"]:
                        if col in survival.columns:
                            row[col] = survival.loc[s, col]
                rows.append(row)
        sample_rows.append(pd.DataFrame(rows))

        base = pd.DataFrame(
            {"phospho_binary": [0] * len(low_samples) + [1] * len(high_samples), "tf_protein": tf_protein.values},
            index=all_samples,
        ).join(clinical_df, how="left")

        time_col, event_col = endpoint_columns(endpoint)
        df = base.join(survival[[time_col, event_col]], how="left")
        df[time_col] = _safe_to_numeric(df[time_col])
        df[event_col] = _safe_to_numeric(df[event_col])

        # events for depth: complete case on phospho only
        n_events_m1 = int(df[[time_col, event_col, "phospho_binary"]].dropna()[event_col].sum())
        depth = analysis_depth_from_events(n_events_m1, min_events_m1=args.min_events_m1)
        allowed = set(allowed_models_for_depth(depth))
        depth_rows.append(
            {
                "cancer_type": cancer_type,
                "site": site,
                "site_label": site_label,
                "endpoint": endpoint,
                "events_m1": n_events_m1,
                "analysis_depth": depth,
                "allowed_models": ",".join(sorted(allowed)) if allowed else "none",
            }
        )

        for layer in layers:
            if layer.model_name not in allowed:
                result_rows.append(
                    {
                        "cancer_type": cancer_type,
                        "site": site,
                        "site_label": site_label,
                        "tf_gene_id": tf_gene_id,
                        "model": layer.model_name,
                        "endpoint": endpoint,
                        "analysis_role": layer.role,
                        "analysis_depth": depth,
                        "status": "skipped_by_event_depth",
                        "events": n_events_m1,
                        "method": "ordinary",
                    }
                )
                continue

            covs = [c for c in layer.covariates if c in df.columns]
            if layer.model_name == "clinical_supportive" and "stage_advanced" not in covs:
                result_rows.append(
                    {
                        "cancer_type": cancer_type,
                        "site": site,
                        "site_label": site_label,
                        "model": layer.model_name,
                        "endpoint": endpoint,
                        "analysis_role": layer.role,
                        "analysis_depth": depth,
                        "status": "skipped_no_stage",
                        "method": "ordinary",
                        "clinical_mode": clinical_meta.get("clinical_mode"),
                    }
                )
                continue

            fit = fit_ordinary_cox(df, time_col, event_col, covs)
            result_rows.append(
                {
                    "cancer_type": cancer_type,
                    "site": site,
                    "site_label": site_label,
                    "tf_gene_id": tf_gene_id,
                    "model": layer.model_name,
                    "endpoint": endpoint,
                    "analysis_role": layer.role,
                    "analysis_depth": depth,
                    "method": fit.get("method", "ordinary"),
                    "status": fit.get("status"),
                    "HR_phospho": fit.get("HR_phospho", np.nan),
                    "CI_low": fit.get("CI_low", np.nan),
                    "CI_high": fit.get("CI_high", np.nan),
                    "p_raw": fit.get("p_raw", np.nan),
                    "q_bh": np.nan,
                    "n": fit.get("n", np.nan),
                    "events": fit.get("events", n_events_m1),
                    "n_covariates": fit.get("n_covariates", len(covs)),
                    "clinical_mode": clinical_meta.get("clinical_mode"),
                    "n_dropped_by_na": fit.get("n_dropped_by_na", np.nan),
                    "cox_error": fit.get("error", np.nan),
                }
            )

    sample_df = pd.concat(sample_rows, ignore_index=True) if sample_rows else pd.DataFrame()
    sample_df.to_csv(out_dir / "sample_phospho_groups.csv", index=False)
    pd.DataFrame(endpoint_rows).to_csv(out_dir / "site_primary_endpoint_manifest.csv", index=False)
    pd.DataFrame(depth_rows).to_csv(out_dir / "event_adaptive_depth_manifest.csv", index=False)

    results_df = pd.DataFrame(result_rows)
    for col in ["HR_phospho", "CI_low", "CI_high", "p_raw", "q_bh", "n", "events"]:
        if col not in results_df.columns:
            results_df[col] = np.nan
    results_df.to_csv(out_dir / "cox_results_raw.csv", index=False)

    # FDR only on ordinary primary+supportive fits that succeeded
    adj = apply_stratified_fdr(results_df, args.fdr_scope)
    adj.to_csv(out_dir / "cox_results_bh_fdr.csv", index=False)

    evidence, summary = build_evidence_table(adj)
    evidence.to_csv(out_dir / "evidence_upgrade_table.csv", index=False)
    summary.to_csv(out_dir / "layer_significance_summary.csv", index=False)
    try:
        with pd.ExcelWriter(out_dir / "evidence_upgrade_table.xlsx") as w:
            evidence.to_excel(w, sheet_name="evidence_by_site", index=False)
            summary.to_excel(w, sheet_name="layer_summary", index=False)
            pd.DataFrame(depth_rows).to_excel(w, sheet_name="event_depth", index=False)
    except Exception:
        pass

    # Sensitivity on M1/M2 ordinary success with raw p<0.10 or BH hits
    sens_targets = adj.loc[
        adj["status"].eq("success")
        & adj["method"].fillna("ordinary").eq("ordinary")
        & adj["model"].isin(["univariate", "protein_adjusted"])
        & (
            (adj["p_raw"] < 0.10)
            | (adj["q_bh"] <= 0.05)
        )
    ].copy()

    sens_rows: List[Dict[str, object]] = []
    for _, hit in sens_targets.iterrows():
        cancer_type = str(hit["cancer_type"])
        site = str(hit["site"])
        site_label = str(hit["site_label"])
        endpoint = str(hit["endpoint"])
        model = str(hit["model"])
        time_col, event_col = endpoint_columns(endpoint)
        covs = ["phospho_binary"] if model == "univariate" else ["phospho_binary", "tf_protein"]

        # rebuild df from sample table
        sub_s = sample_df.loc[sample_df["cancer_type"].eq(cancer_type) & sample_df["site"].eq(site)].copy()
        if sub_s.empty:
            continue
        df = sub_s.set_index("sample")[["phospho_binary", "tf_protein", time_col, event_col]].copy()
        df[time_col] = _safe_to_numeric(df[time_col])
        df[event_col] = _safe_to_numeric(df[event_col])

        ridge = fit_ordinary_cox(df, time_col, event_col, covs, penalizer=args.ridge_penalizer)
        firth = fit_firth_cox(df, time_col, event_col, covs)
        boot = bootstrap_hr_direction(
            df, time_col, event_col, covs, n_boot=args.n_bootstrap, seed=args.bootstrap_seed
        )

        sens_rows.append(
            {
                "cancer_type": cancer_type,
                "site": site,
                "site_label": site_label,
                "endpoint": endpoint,
                "model": model,
                "ordinary_HR": hit.get("HR_phospho"),
                "ordinary_p": hit.get("p_raw"),
                "ordinary_q": hit.get("q_bh"),
                "ridge_status": ridge.get("status"),
                "ridge_HR": ridge.get("HR_phospho"),
                "ridge_p": ridge.get("p_raw"),
                "firth_status": firth.get("status"),
                "firth_HR": firth.get("HR_phospho"),
                "firth_p": firth.get("p_raw"),
                "firth_CI_low": firth.get("CI_low"),
                "firth_CI_high": firth.get("CI_high"),
                "boot_status": boot.get("status"),
                "boot_n_success": boot.get("n_boot_success"),
                "boot_hr_median": boot.get("hr_median"),
                "boot_hr_ci_low": boot.get("hr_ci_low"),
                "boot_hr_ci_high": boot.get("hr_ci_high"),
                "boot_frac_hr_gt1": boot.get("frac_hr_gt1"),
                "boot_frac_hr_lt1": boot.get("frac_hr_lt1"),
            }
        )

    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(sens_dir / "firth_ridge_bootstrap_sensitivity.csv", index=False)

    # KM for BH hits
    if not args.no_km_plots and not sample_df.empty and not evidence.empty:
        hits = evidence.loc[evidence["m1_bh_sig"].fillna(False) | evidence["m2_bh_sig"].fillna(False) | evidence["m3s_bh_sig"].fillna(False)]
        km_rows = []
        for _, hit in hits.iterrows():
            safe = str(hit["site_label"]).replace("/", "_").replace("|", "_")
            ep = str(hit.get("primary_endpoint", "OS"))
            if ep not in {"OS", "PFS"}:
                ep = "OS"
            path = km_dir / f"KM_{hit['cancer_type']}_{safe}_{ep}.png"
            cox_row = adj.loc[
                adj["cancer_type"].eq(hit["cancer_type"])
                & adj["site"].eq(hit["site"])
                & adj["model"].eq("univariate")
                & adj["status"].eq("success")
            ]
            if "method" in adj.columns:
                cox_row = cox_row.loc[cox_row["method"].fillna("ordinary").eq("ordinary")]
            cox_hr = cox_ci_low = cox_ci_high = cox_p = cox_q = None
            if not cox_row.empty:
                r0 = cox_row.iloc[0]
                cox_hr, cox_ci_low, cox_ci_high = r0.get("HR_phospho"), r0.get("CI_low"), r0.get("CI_high")
                cox_p, cox_q = r0.get("p_raw"), r0.get("q_bh")
            lp = make_km_plot(
                sample_df,
                cancer_type=str(hit["cancer_type"]),
                site=str(hit["site"]),
                site_label=str(hit["site_label"]),
                endpoint=ep,
                out_path=path,
                cox_hr=cox_hr,
                cox_ci_low=cox_ci_low,
                cox_ci_high=cox_ci_high,
                cox_p=cox_p,
                cox_q=cox_q,
            )
            km_rows.append({"cancer_type": hit["cancer_type"], "site_label": hit["site_label"], "endpoint": ep, "logrank_p": lp, "path": str(path)})
        if km_rows:
            pd.DataFrame(km_rows).to_csv(out_dir / "km_plot_manifest.csv", index=False)

    print(f"[OK] v4 output: {out_dir}")
    print(summary.to_string(index=False))
    print(f"[OK] Sensitivity rows: {len(sens_df)} -> {sens_dir / 'firth_ridge_bootstrap_sensitivity.csv'}")
    if len(sens_df):
        print(sens_df[["cancer_type", "site_label", "model", "ordinary_HR", "firth_HR", "boot_frac_hr_gt1"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
