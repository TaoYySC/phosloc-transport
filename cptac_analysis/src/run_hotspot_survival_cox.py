#!/usr/bin/env python3
"""Hotspot-level CPTAC survival Cox (v4 logic, activity exposure).

Definition (locked): Known/HC-only hotspots are clustered first; Cox exposure is
hotspot activity (default mean_z of members measured in that cancer).

Same event-adaptive Cox / Firth / bootstrap / KM framework as
run_phosphosite_survival_cox.py, but the exposure is hotspot activity
injected via HotspotTargetRegulationPipeline.load_phospho (site key ENSG|HS_...).

Primary cohort: BH-significant cancer×hotspot pairs (Import or Export × activate/repress)
from a target-regulation CPTAC run. Use --target-regulation to keep activate, repress, or all.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from run_import_target_regulation_analysis import TempoConfig
from run_hotspot_target_regulation_analysis import (
    HotspotTargetRegulationPipeline,
    load_hotspot_tables,
)
from run_phosphosite_survival_cox import (  # noqa: E402
    CoxLayerSpec,
    CoxPHFitter,
    EXPECTED_PHOSPHO_SPLIT_MODE,
    apply_stratified_fdr,
    bootstrap_hr_direction,
    build_clinical_covariates,
    build_evidence_table,
    analysis_depth_from_events,
    allowed_models_for_depth,
    choose_site_primary_endpoint,
    endpoint_columns,
    extract_site_phospho_values,
    extract_tf_protein_values,
    fit_firth_cox,
    fit_ordinary_cox,
    get_phospho_split_from_pipeline,
    load_phenotype_meta,
    load_survival,
    make_km_plot,
    validate_split_against_expected,
    _safe_to_numeric,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

DEFAULT_BASE = Path(__file__).resolve().parents[1]
# Canonical: Known/HC-only hotspot catalogs + CPTAC under ..._anchor_only (d=10).
DEFAULT_HOTSPOT_RESULTS = (
    DEFAULT_BASE
    / "results/import_target_regulation_hotspot_v11_147pos_d3_platt_anchor_only"
)
DEFAULT_SIGNIFICANT_CSV = (
    DEFAULT_HOTSPOT_RESULTS
    / "figure4/figure4b_nuclear_hotspot_targets"
    / "all_cancers_Import_activate_bh_significant_sites_table.csv"
)
DEFAULT_POINTS_CSV = DEFAULT_HOTSPOT_RESULTS / "all_target_gene_mean_expression_points.csv"
DEFAULT_HOTSPOT_DIR = (
    DEFAULT_BASE
    / "results/import_target_regulation_hotspot_v11_147pos_d3_platt/hotspots"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_BASE
    / "results/survival_analysis/hotspot_cox_v11_147pos_d3_platt_anchor_only_mean_z"
)
DEFAULT_SITE_EVIDENCE = (
    DEFAULT_BASE
    / "results/survival_analysis/phosphosite_cox_v11_147pos_d3_platt/evidence_upgrade_table.csv"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--significant-csv", type=Path, default=DEFAULT_SIGNIFICANT_CSV)
    p.add_argument("--points-csv", type=Path, default=DEFAULT_POINTS_CSV)
    p.add_argument("--hotspot-dir", type=Path, default=DEFAULT_HOTSPOT_DIR)
    p.add_argument("--distance", type=int, default=10)
    p.add_argument("--activity", choices=["mean_z", "max", "direction_weighted"], default="mean_z")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--site-evidence-csv", type=Path, default=DEFAULT_SITE_EVIDENCE)
    p.add_argument("--endpoint-mode", choices=["per_site", "cancer_specific"], default="per_site")
    p.add_argument("--fdr-scope", choices=["per_cancer", "global_model_endpoint"], default="per_cancer")
    p.add_argument("--min-group-samples", type=int, default=3)
    p.add_argument("--min-events-m1", type=int, default=10)
    p.add_argument("--ridge-penalizer", type=float, default=0.1)
    p.add_argument("--n-bootstrap", type=int, default=500)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument("--no-km-plots", action="store_true")
    p.add_argument("--max-sites", type=int, default=0)
    p.add_argument("--skip-split-validation", action="store_true",
                   help="If reconstructed activity median differs slightly from points CSV, still proceed")
    p.add_argument(
        "--target-regulation",
        choices=["activate", "repress", "all"],
        default="activate",
        help="Keep only this target_regulation from significant CSV (default activate; use all to keep as-is)",
    )
    return p.parse_args()


def _attach_hotspot_meta(expected: pd.Series) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for col in (
        "hotspot_id",
        "hotspot_label",
        "hotspot_class",
        "n_measured_members",
        "n_members",
        "activity_mode",
    ):
        if col in expected.index:
            out[col] = expected.get(col)
    return out


def compare_to_site_evidence(hotspot_evidence: pd.DataFrame, site_evidence_path: Path, out_path: Path) -> None:
    """Join hotspot Cox evidence to unit-site evidence by cancer + TF name when possible."""
    if hotspot_evidence.empty or not site_evidence_path.exists():
        return
    site_ev = pd.read_csv(site_evidence_path)
    if site_ev.empty:
        return
    hs = hotspot_evidence.copy()
    # TF from hotspot label prefix before first _
    hs["tf_name"] = hs["site_label"].astype(str).str.split("_", n=1).str[0]
    site_ev = site_ev.copy()
    site_ev["tf_name"] = site_ev["site_label"].astype(str).str.split("_", n=1).str[0]
    cols_keep = [
        c
        for c in site_ev.columns
        if c.startswith("univariate_") or c.startswith("protein_adjusted_") or c in {"evidence_class", "site", "site_label"}
    ]
    site_slim = site_ev[["cancer_type", "tf_name", "site", "site_label", "evidence_class"] + [
        c for c in cols_keep if c not in {"site", "site_label", "evidence_class"}
    ]].copy()
    site_slim = site_slim.rename(
        columns={
            "site": "unit_site",
            "site_label": "unit_site_label",
            "evidence_class": "unit_evidence_class",
            "univariate_HR": "unit_univariate_HR",
            "univariate_p_raw": "unit_univariate_p_raw",
            "univariate_q_bh": "unit_univariate_q_bh",
            "protein_adjusted_HR": "unit_protein_adjusted_HR",
            "protein_adjusted_p_raw": "unit_protein_adjusted_p_raw",
            "protein_adjusted_q_bh": "unit_protein_adjusted_q_bh",
        }
    )
    # one row per cancer×TF: keep most significant unit univariate p if multiple sites
    if "unit_univariate_p_raw" in site_slim.columns:
        site_slim = site_slim.sort_values("unit_univariate_p_raw").drop_duplicates(
            ["cancer_type", "tf_name"], keep="first"
        )
    else:
        site_slim = site_slim.drop_duplicates(["cancer_type", "tf_name"], keep="first")

    cmp = hs.merge(site_slim, on=["cancer_type", "tf_name"], how="left")
    cmp.to_csv(out_path, index=False)


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

    hotspots, members = load_hotspot_tables(args.hotspot_dir, args.distance)
    config = TempoConfig(
        phospho_split_mode=EXPECTED_PHOSPHO_SPLIT_MODE,
        phospho_value_mode="site_abundance",
        exclude_zero_phospho_for_split=False,
        min_group_samples=args.min_group_samples,
    )
    # Prefer LinkedOmics paths from hotspot run_config if present
    run_cfg = args.points_csv.parent / "run_config.json"
    if run_cfg.exists():
        meta = json.loads(run_cfg.read_text(encoding="utf-8"))
        for key in (
            "linkedomics_base",
            "chip_dir",
            "signed_regulon_path",
            "idmapping_path",
            "prediction_output_dir",
            "import_prediction_filename",
            "export_prediction_filename",
            "known_positive_path",
        ):
            if key in meta and hasattr(config, key):
                setattr(config, key, meta[key])

    pipeline = HotspotTargetRegulationPipeline(
        config=config,
        hotspots=hotspots,
        members=members,
        activity_mode=args.activity,
        cluster_distance=args.distance,
    )

    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "analysis": "hotspot_survival_cox_v4",
                "activity_mode": args.activity,
                "cluster_distance": args.distance,
                "significant_csv": str(args.significant_csv),
                "points_csv": str(args.points_csv),
                "hotspot_dir": str(args.hotspot_dir),
                "endpoint_mode": args.endpoint_mode,
                "fdr_scope": args.fdr_scope,
                "min_events_m1": args.min_events_m1,
                "target_regulation": args.target_regulation,
                "exposure": (
                    "hotspot activity row in load_phospho "
                    f"(mode={args.activity}); binary high/low via median_nonmissing"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    df_sig = pd.read_csv(args.significant_csv)
    if "target_regulation" in df_sig.columns and args.target_regulation != "all":
        df_sig = df_sig.loc[
            df_sig["target_regulation"].astype(str).eq(args.target_regulation)
        ].copy()
    df_points = pd.read_csv(args.points_csv)
    split_map = df_points.drop_duplicates(subset=["cancer_type", "site"]).set_index(
        ["cancer_type", "site"]
    )

    cache: Dict[str, Dict[str, object]] = {}

    def load_cancer(cancer_type: str) -> None:
        if cancer_type in cache:
            return
        src = Path(config.linkedomics_base) / cancer_type
        cache[cancer_type] = {
            "phospho": pipeline.load_phospho(cancer_type),  # includes hotspot activity rows
            "rna": pipeline.load_rna(cancer_type),
            "protein": pipeline.load_protein(cancer_type),
            "phenotype": load_phenotype_meta(src / f"{cancer_type}_meta.txt"),
            "survival": load_survival(src / f"{cancer_type}_survival.txt"),
        }

    label_col = "site_label"
    if "hotspot_label" in df_sig.columns and df_sig["hotspot_label"].notna().any():
        df_sig["site_label"] = df_sig["hotspot_label"].fillna(df_sig.get("site_label"))
    site_entries = (
        df_sig.drop_duplicates(subset=["cancer_type", "site"])[
            [c for c in ["cancer_type", "site", label_col, "hotspot_id", "hotspot_class"] if c in df_sig.columns]
        ]
        .to_dict(orient="records")
    )
    if args.max_sites > 0:
        site_entries = site_entries[: args.max_sites]

    print(
        f"Hotspot Cox: {len(site_entries)} cancer×hotspot pairs | "
        f"activity={args.activity} | out={out_dir}"
    )

    sample_rows: List[pd.DataFrame] = []
    result_rows: List[Dict[str, object]] = []
    endpoint_rows: List[Dict[str, object]] = []
    depth_rows: List[Dict[str, object]] = []

    layers = [
        CoxLayerSpec("univariate", ("phospho_binary",), "primary"),
        CoxLayerSpec("protein_adjusted", ("phospho_binary", "tf_protein"), "primary"),
        CoxLayerSpec(
            "clinical_supportive", ("phospho_binary", "tf_protein", "stage_advanced"), "supportive"
        ),
    ]

    for entry in site_entries:
        cancer_type = str(entry["cancer_type"])
        site = str(entry["site"])
        site_label = str(entry.get("site_label", site))

        if (cancer_type, site) not in split_map.index:
            result_rows.append(
                {
                    "cancer_type": cancer_type,
                    "site": site,
                    "site_label": site_label,
                    "model": "skip",
                    "status": "missing_split_map",
                }
            )
            continue

        expected = split_map.loc[(cancer_type, site)]
        if isinstance(expected, pd.DataFrame):
            expected = expected.iloc[0]
        tf_gene_id = str(expected["tf_gene_id"])
        hs_meta = _attach_hotspot_meta(expected)

        load_cancer(cancer_type)
        df_phospho = cache[cancer_type]["phospho"]  # type: ignore
        df_rna = cache[cancer_type]["rna"]  # type: ignore
        df_protein = cache[cancer_type]["protein"]  # type: ignore
        phenotype = cache[cancer_type]["phenotype"]  # type: ignore
        survival = cache[cancer_type]["survival"]  # type: ignore

        if site not in df_phospho.index:
            result_rows.append(
                {
                    "cancer_type": cancer_type,
                    "site": site,
                    "site_label": site_label,
                    "model": "skip",
                    "status": "hotspot_activity_missing_in_phospho",
                    **hs_meta,
                }
            )
            continue

        split_info = get_phospho_split_from_pipeline(
            pipeline, site=site, df_phospho=df_phospho, df_rna=df_rna, df_protein=df_protein
        )
        ok, qc = validate_split_against_expected(split_info, expected)
        if not ok and not args.skip_split_validation:
            result_rows.append(
                {
                    "cancer_type": cancer_type,
                    "site": site,
                    "site_label": site_label,
                    "model": "skip",
                    "status": qc,
                    **hs_meta,
                }
            )
            continue
        if not ok and args.skip_split_validation:
            print(f"  warn {cancer_type} {site_label}: {qc} (continuing)")

        low_samples = list(split_info["low_samples"])
        high_samples = list(split_info["high_samples"])
        all_samples = low_samples + high_samples

        if args.endpoint_mode == "cancer_specific":
            from run_phosphosite_survival_cox import DEFAULT_CANCER_ENDPOINTS

            endpoint = DEFAULT_CANCER_ENDPOINTS.get(cancer_type, "OS")
            ep_meta = {"selection_rule": "cancer_specific"}
        else:
            from run_phosphosite_survival_cox import DEFAULT_CANCER_ENDPOINTS

            endpoint, ep_meta = choose_site_primary_endpoint(
                survival,
                all_samples,
                cancer_type=cancer_type,
                cancer_fallback=DEFAULT_CANCER_ENDPOINTS,
            )
        endpoint_rows.append(
            {
                "cancer_type": cancer_type,
                "site": site,
                "site_label": site_label,
                "primary_endpoint": endpoint,
                **ep_meta,
                **hs_meta,
            }
        )

        # Continuous exposure = hotspot activity (z / max-z), not raw site abundance
        activity_values = extract_site_phospho_values(
            df_phospho=df_phospho, df_rna=df_rna, site=site, samples=all_samples
        )
        tf_protein = extract_tf_protein_values(
            cancer_protein_df=df_protein, tf_gene_id=tf_gene_id, samples=all_samples
        )
        clinical_df, clinical_meta = build_clinical_covariates(phenotype, all_samples)

        rows = []
        for grp, samples in [("low", low_samples), ("high", high_samples)]:
            for s in samples:
                row = {
                    "cancer_type": cancer_type,
                    "site": site,
                    "site_label": site_label,
                    "sample": s,
                    "phospho_group": grp,
                    "phospho_binary": 0 if grp == "low" else 1,
                    "phospho_log2": activity_values.get(s, np.nan),  # activity score
                    "activity_mode": args.activity,
                    "tf_gene_id": tf_gene_id,
                    "tf_protein": tf_protein.get(s, np.nan),
                    **hs_meta,
                }
                if s in survival.index:
                    for col in ("OS_days", "OS_event", "PFS_days", "PFS_event"):
                        if col in survival.columns:
                            row[col] = survival.loc[s, col]
                if s in clinical_df.index:
                    for col in clinical_df.columns:
                        row[col] = clinical_df.loc[s, col]
                rows.append(row)
        sample_df_i = pd.DataFrame(rows)
        sample_rows.append(sample_df_i)

        time_col, event_col = endpoint_columns(endpoint)
        df = sample_df_i.set_index("sample").copy()
        df[time_col] = _safe_to_numeric(df[time_col])
        df[event_col] = _safe_to_numeric(df[event_col])
        df["phospho_binary"] = pd.to_numeric(df["phospho_binary"], errors="coerce")
        df["tf_protein"] = pd.to_numeric(df["tf_protein"], errors="coerce")

        n_events_m1 = int(df[[time_col, event_col, "phospho_binary"]].dropna()[event_col].sum())
        depth = analysis_depth_from_events(n_events_m1, min_events_m1=args.min_events_m1)
        depth_rows.append(
            {
                "cancer_type": cancer_type,
                "site": site,
                "site_label": site_label,
                "endpoint": endpoint,
                "n_events_m1": n_events_m1,
                "analysis_depth": depth,
                **hs_meta,
            }
        )
        allowed = set(allowed_models_for_depth(depth))

        for layer in layers:
            if layer.model_name not in allowed:
                result_rows.append(
                    {
                        "cancer_type": cancer_type,
                        "site": site,
                        "site_label": site_label,
                        "model": layer.model_name,
                        "endpoint": endpoint,
                        "analysis_role": layer.role,
                        "analysis_depth": depth,
                        "status": "skipped_by_event_depth",
                        "method": "ordinary",
                        **hs_meta,
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
                        **hs_meta,
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
                    "activity_mode": args.activity,
                    **hs_meta,
                }
            )

    sample_df = pd.concat(sample_rows, ignore_index=True) if sample_rows else pd.DataFrame()
    sample_df.to_csv(out_dir / "sample_hotspot_activity_groups.csv", index=False)
    pd.DataFrame(endpoint_rows).to_csv(out_dir / "hotspot_primary_endpoint_manifest.csv", index=False)
    pd.DataFrame(depth_rows).to_csv(out_dir / "event_adaptive_depth_manifest.csv", index=False)

    results_df = pd.DataFrame(result_rows)
    for col in ["HR_phospho", "CI_low", "CI_high", "p_raw", "q_bh", "n", "events"]:
        if col not in results_df.columns:
            results_df[col] = np.nan
    results_df.to_csv(out_dir / "cox_results_raw.csv", index=False)

    adj = apply_stratified_fdr(results_df, args.fdr_scope)
    adj.to_csv(out_dir / "cox_results_bh_fdr.csv", index=False)

    evidence, summary = build_evidence_table(adj)
    evidence.to_csv(out_dir / "evidence_upgrade_table.csv", index=False)
    summary.to_csv(out_dir / "layer_significance_summary.csv", index=False)
    compare_to_site_evidence(evidence, args.site_evidence_csv, out_dir / "hotspot_vs_unit_site_evidence.csv")

    try:
        with pd.ExcelWriter(out_dir / "evidence_upgrade_table.xlsx") as w:
            evidence.to_excel(w, sheet_name="evidence_by_hotspot", index=False)
            summary.to_excel(w, sheet_name="layer_summary", index=False)
            pd.DataFrame(depth_rows).to_excel(w, sheet_name="event_depth", index=False)
    except Exception:
        pass

    sens_targets = adj.loc[
        adj["status"].eq("success")
        & adj["method"].fillna("ordinary").eq("ordinary")
        & adj["model"].isin(["univariate", "protein_adjusted"])
        & ((adj["p_raw"] < 0.10) | (adj["q_bh"] <= 0.05))
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

    if not args.no_km_plots and not sample_df.empty and not evidence.empty:
        hits = evidence.loc[
            evidence["m1_bh_sig"].fillna(False)
            | evidence["m2_bh_sig"].fillna(False)
            | evidence["m3s_bh_sig"].fillna(False)
        ]
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
            km_rows.append(
                {
                    "cancer_type": hit["cancer_type"],
                    "site_label": hit["site_label"],
                    "endpoint": ep,
                    "logrank_p": lp,
                    "path": str(path),
                }
            )
        if km_rows:
            pd.DataFrame(km_rows).to_csv(out_dir / "km_plot_manifest.csv", index=False)

    print(f"[OK] hotspot Cox output: {out_dir}")
    print(summary.to_string(index=False))
    print(f"[OK] Sensitivity rows: {len(sens_df)}")
    n_skip = int((results_df.get("status", pd.Series(dtype=str)) == "missing_split_map").sum()) if not results_df.empty else 0
    n_split_fail = (
        int(results_df["status"].astype(str).str.contains("split_mismatch|unexpected_phospho").sum())
        if not results_df.empty and "status" in results_df.columns
        else 0
    )
    print(f"[QC] skip missing_split={n_skip} split_mismatch={n_split_fail}")


if __name__ == "__main__":
    main()
