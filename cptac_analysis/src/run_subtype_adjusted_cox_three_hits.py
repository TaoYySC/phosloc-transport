#!/usr/bin/env python3
"""Subtype-adjusted Cox for three priority hotspot hits.

Pairs (same median High/Low as km_curves survival run):
  - CCRCC STAT3_Y686-T708  | subtype = BAP1 mutation (molecular class)
  - PDAC  IRF9_S131-S139   | subtype = Moffitt-like basal vs classical (RNA)
  - LUAD  STAT3_Y686-T708  | subtype = KRAS mutation (driver class)

Models (event-aware):
  M1:  phospho_binary
  M2:  phospho_binary + tf_protein
  M3s: phospho_binary + tf_protein + subtype_binary
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from run_phosphosite_survival_cox import (  # noqa: E402
    CoxPHFitter,
    fit_ordinary_cox,
)

BASE = Path(__file__).resolve().parents[1]
LOKB = BASE / "data/source/1.cpatac/LinkedOmicsKB"
SURV = (
    BASE
    / "results/survival_analysis/hotspot_cox_v11_147pos_d3_platt_mixed_pred_filter_mean_z"
)
OUT = SURV / "subtype_adjusted_cox"
IDMAP = BASE / "data/source/3.idmapping/HUMAN_9606_idmapping_selected.tab.gz"

# Compact Moffitt tumor-intrinsic gene panels with GRCh38 Ensembl IDs.
MOFFITT_BASAL = {
    "S100A2": "ENSG00000196754",
    "KRT6A": "ENSG00000205420",
    "KRT5": "ENSG00000186081",
    "KRT14": "ENSG00000186847",
    "KRT17": "ENSG00000128435",
    "LAMC2": "ENSG00000058085",
    "SPRR1B": "ENSG00000169469",
    "SPRR3": "ENSG00000163209",
    "FABP5": "ENSG00000164687",
    "LY6D": "ENSG00000167656",
    "DST": "ENSG00000151914",
    "ITGA6": "ENSG00000091409",
    "COL17A1": "ENSG00000065618",
    "SERPINB3": "ENSG00000057149",
    "SERPINB4": "ENSG00000206075",
    "SCEL": "ENSG00000136155",
    "DHRS9": "ENSG00000122545",
    "TRIM29": "ENSG00000137634",
}
MOFFITT_CLASSICAL = {
    "BTNL8": "ENSG00000105048",
    "FAM3D": "ENSG00000198670",
    "LGALS4": "ENSG00000171747",
    "REG4": "ENSG00000134193",
    "AGR2": "ENSG00000106541",
    "AGR3": "ENSG00000173467",
    "TFF1": "ENSG00000160182",
    "TFF2": "ENSG00000160190",
    "TFF3": "ENSG00000160180",
    "GPX2": "ENSG00000176153",
    "CDH17": "ENSG00000079112",
    "MUC13": "ENSG00000173702",
    "MUC17": "ENSG00000169876",
    "MYO7B": "ENSG00000169994",
    "ANXA10": "ENSG00000109511",
    "EPS8L3": "ENSG00000198758",
    "CAPN8": "ENSG00000203697",
    "VIL1": "ENSG00000127831",
}

PAIRS = [
    {
        "cancer_type": "CCRCC",
        "site_label": "STAT3_Y686-T708",
        "endpoint": "OS",
        "subtype_name": "BAP1_mutation",
        "subtype_high_label": "BAP1_mut",
        "subtype_low_label": "BAP1_wt",
    },
    {
        "cancer_type": "PDAC",
        "site_label": "IRF9_S131-S139",
        "endpoint": "OS",
        "subtype_name": "Moffitt_basal_vs_classical",
        "subtype_high_label": "basal-like",
        "subtype_low_label": "classical",
    },
    {
        "cancer_type": "LUAD",
        "site_label": "STAT3_Y686-T708",
        "endpoint": "OS",
        "subtype_name": "KRAS_mutation",
        "subtype_high_label": "KRAS_mut",
        "subtype_low_label": "KRAS_wt",
    },
]


def resolve_rna_rows(rna: pd.DataFrame, ensembl_by_symbol: Dict[str, str]) -> List[str]:
    ens_base_to_row = {idx.split(".")[0]: idx for idx in rna.index.astype(str)}
    rows = []
    for ens in ensembl_by_symbol.values():
        if ens in ens_base_to_row:
            rows.append(ens_base_to_row[ens])
    return rows


def moffitt_assign(cancer: str, samples: Sequence[str], _sym2ens: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    rna_path = LOKB / cancer / f"{cancer}_RNAseq_gene_RSEM_coding_UQ_1500_log2_Tumor.txt"
    rna = pd.read_csv(rna_path, sep="\t", index_col=0)
    basal_rows = resolve_rna_rows(rna, MOFFITT_BASAL)
    class_rows = resolve_rna_rows(rna, MOFFITT_CLASSICAL)
    if len(basal_rows) < 5 or len(class_rows) < 5:
        raise RuntimeError(
            f"Too few Moffitt genes mapped: basal={len(basal_rows)} classical={len(class_rows)}"
        )
    # z-score genes across samples, then mean
    def panel_score(rows: List[str]) -> pd.Series:
        sub = rna.loc[rows].apply(pd.to_numeric, errors="coerce")
        z = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1).replace(0, np.nan), axis=0)
        return z.mean(axis=0, skipna=True)

    basal = panel_score(basal_rows)
    classical = panel_score(class_rows)
    score = basal - classical
    score = score.reindex(list(samples))
    # assign by sign relative to cohort median among non-missing
    med = float(score.dropna().median())
    subtype = pd.Series(index=score.index, dtype=object)
    subtype.loc[score.notna() & (score >= med)] = "basal-like"
    subtype.loc[score.notna() & (score < med)] = "classical"
    binary = pd.Series(np.nan, index=score.index, dtype=float)
    binary.loc[subtype.eq("basal-like")] = 1.0
    binary.loc[subtype.eq("classical")] = 0.0
    out = pd.DataFrame(
        {
            "sample": score.index,
            "subtype_label": subtype.values,
            "subtype_binary": binary.values,
            "subtype_score": score.values,
            "n_basal_genes": len(basal_rows),
            "n_classical_genes": len(class_rows),
            "score_median_cutoff": med,
        }
    )
    return out


def mutation_assign(
    cancer: str,
    samples: Sequence[str],
    mut_col: str,
    high_label: str,
    low_label: str,
) -> pd.DataFrame:
    meta = pd.read_csv(LOKB / cancer / f"{cancer}_meta.txt", sep="\t", index_col=0)
    if "data_type" in meta.index:
        meta = meta.drop(index=["data_type"])
    if mut_col not in meta.columns:
        raise KeyError(f"{mut_col} not in {cancer} meta")
    vals = pd.to_numeric(meta[mut_col], errors="coerce").reindex(list(samples))
    label = pd.Series(index=vals.index, dtype=object)
    label.loc[vals.eq(1)] = high_label
    label.loc[vals.eq(0)] = low_label
    return pd.DataFrame(
        {
            "sample": list(samples),
            "subtype_label": label.values,
            "subtype_binary": vals.astype(float).values,
            "subtype_score": vals.astype(float).values,
            "n_basal_genes": np.nan,
            "n_classical_genes": np.nan,
            "score_median_cutoff": np.nan,
        }
    )


def fisher_or_counts(df: pd.DataFrame) -> Dict[str, object]:
    tab = pd.crosstab(df["phospho_group"], df["subtype_label"])
    # ensure 2x2 orientation if possible
    out: Dict[str, object] = {"table": tab.to_dict()}
    if tab.shape == (2, 2):
        a = float(tab.iloc[0, 0])
        b = float(tab.iloc[0, 1])
        c = float(tab.iloc[1, 0])
        d = float(tab.iloc[1, 1])
        # odds ratio High in col1 vs col0 — interpret via labels
        from scipy.stats import fisher_exact

        oddsr, p = fisher_exact([[a, b], [c, d]])
        out.update(
            {
                "fisher_odds_ratio": oddsr,
                "fisher_p": p,
                "table_index": list(tab.index.astype(str)),
                "table_columns": list(tab.columns.astype(str)),
            }
        )
    else:
        out.update({"fisher_odds_ratio": np.nan, "fisher_p": np.nan})
    return out


def fit_layers(df: pd.DataFrame, time_col: str, event_col: str) -> List[Dict[str, object]]:
    rows = []
    specs = [
        ("univariate", ["phospho_binary"]),
        ("protein_adjusted", ["phospho_binary", "tf_protein"]),
        ("subtype_adjusted", ["phospho_binary", "tf_protein", "subtype_binary"]),
    ]
    for name, covs in specs:
        use = [c for c in covs if c in df.columns]
        sub = df[[time_col, event_col] + use].dropna().copy()
        n_events = int(pd.to_numeric(sub[event_col], errors="coerce").fillna(0).sum()) if len(sub) else 0
        if CoxPHFitter is None:
            rows.append({"model": name, "status": "lifelines_missing", "n": len(sub), "events": n_events})
            continue
        if len(sub) < 10 or n_events < 5 or sub["phospho_binary"].nunique() < 2:
            rows.append(
                {
                    "model": name,
                    "status": "insufficient_data",
                    "n": int(len(sub)),
                    "events": n_events,
                    "n_covariates": len(use),
                }
            )
            continue
        if name == "subtype_adjusted" and sub["subtype_binary"].nunique() < 2:
            rows.append(
                {
                    "model": name,
                    "status": "subtype_no_variation",
                    "n": int(len(sub)),
                    "events": n_events,
                    "n_covariates": len(use),
                }
            )
            continue
        fit = fit_ordinary_cox(sub, time_col, event_col, use)
        row = {
            "model": name,
            "status": fit.get("status"),
            "method": fit.get("method", "ordinary"),
            "HR_phospho": fit.get("HR_phospho"),
            "CI_low": fit.get("CI_low"),
            "CI_high": fit.get("CI_high"),
            "p_raw": fit.get("p_raw"),
            "n": fit.get("n", len(sub)),
            "events": fit.get("events", n_events),
            "n_covariates": len(use),
            "cox_error": fit.get("error"),
        }
        if name == "subtype_adjusted" and fit.get("status") == "success":
            cph = CoxPHFitter()
            cph.fit(sub[[time_col, event_col] + use], duration_col=time_col, event_col=event_col)
            if "subtype_binary" in cph.summary.index:
                row["HR_subtype"] = float(np.exp(cph.params_["subtype_binary"]))
                row["p_subtype"] = float(cph.summary.loc["subtype_binary", "p"])
            if "tf_protein" in cph.summary.index:
                row["HR_tf_protein"] = float(np.exp(cph.params_["tf_protein"]))
                row["p_tf_protein"] = float(cph.summary.loc["tf_protein", "p"])
        rows.append(row)
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    samples_all = pd.read_csv(SURV / "sample_hotspot_activity_groups.csv")

    assign_rows = []
    enrich_rows = []
    cox_rows = []
    definitions = {}

    for pair in PAIRS:
        ca = pair["cancer_type"]
        site = pair["site_label"]
        endpoint = pair["endpoint"]
        d = samples_all[(samples_all["cancer_type"] == ca) & (samples_all["site_label"] == site)].copy()
        if d.empty:
            raise RuntimeError(f"No samples for {ca} {site}")

        if pair["subtype_name"] == "Moffitt_basal_vs_classical":
            sub = moffitt_assign(ca, d["sample"].tolist())
            definitions[f"{ca}:{site}"] = {
                "subtype": pair["subtype_name"],
                "rule": "basal_score - classical_score; >= cohort median => basal-like",
                "basal_genes": list(MOFFITT_BASAL.keys()),
                "classical_genes": list(MOFFITT_CLASSICAL.keys()),
                "n_basal_mapped": int(sub["n_basal_genes"].iloc[0]),
                "n_classical_mapped": int(sub["n_classical_genes"].iloc[0]),
                "median_cutoff": float(sub["score_median_cutoff"].iloc[0]),
            }
        elif pair["subtype_name"] == "BAP1_mutation":
            sub = mutation_assign(
                ca, d["sample"].tolist(), "BAP1_mutation", pair["subtype_high_label"], pair["subtype_low_label"]
            )
            definitions[f"{ca}:{site}"] = {
                "subtype": pair["subtype_name"],
                "source": f"{ca}_meta.txt",
                "rule": "BAP1_mutation 1=mut, 0=wt",
            }
        elif pair["subtype_name"] == "KRAS_mutation":
            sub = mutation_assign(
                ca, d["sample"].tolist(), "KRAS_mutation", pair["subtype_high_label"], pair["subtype_low_label"]
            )
            definitions[f"{ca}:{site}"] = {
                "subtype": pair["subtype_name"],
                "source": f"{ca}_meta.txt",
                "rule": "KRAS_mutation 1=mut, 0=wt",
            }
        else:
            raise ValueError(pair["subtype_name"])

        merged = d.merge(sub, on="sample", how="left")
        merged["subtype_name"] = pair["subtype_name"]
        merged["subtype_high_label"] = pair["subtype_high_label"]
        merged["subtype_low_label"] = pair["subtype_low_label"]
        assign_rows.append(merged)

        # enrichment on samples with both phospho group and subtype
        ee = merged.dropna(subset=["phospho_group", "subtype_label"]).copy()
        fisher = fisher_or_counts(ee)
        enrich_rows.append(
            {
                "cancer_type": ca,
                "site_label": site,
                "subtype_name": pair["subtype_name"],
                "n_samples": int(len(ee)),
                "n_high": int((ee["phospho_group"] == "high").sum()),
                "n_low": int((ee["phospho_group"] == "low").sum()),
                "subtype_counts": json.dumps(ee["subtype_label"].value_counts(dropna=False).to_dict()),
                "crosstab": json.dumps(fisher.get("table", {})),
                "table_index": json.dumps(fisher.get("table_index", [])),
                "table_columns": json.dumps(fisher.get("table_columns", [])),
                "fisher_odds_ratio": fisher.get("fisher_odds_ratio"),
                "fisher_p": fisher.get("fisher_p"),
            }
        )

        time_col = f"{endpoint}_days"
        event_col = f"{endpoint}_event"
        cox_df = merged.dropna(subset=[time_col, event_col, "phospho_binary"]).copy()
        cox_df[event_col] = pd.to_numeric(cox_df[event_col], errors="coerce")
        cox_df[time_col] = pd.to_numeric(cox_df[time_col], errors="coerce")
        cox_df = cox_df.dropna(subset=[time_col, event_col])
        for row in fit_layers(cox_df, time_col, event_col):
            row.update(
                {
                    "cancer_type": ca,
                    "site_label": site,
                    "endpoint": endpoint,
                    "subtype_name": pair["subtype_name"],
                    "subtype_high_label": pair["subtype_high_label"],
                    "subtype_low_label": pair["subtype_low_label"],
                }
            )
            cox_rows.append(row)

    assign_df = pd.concat(assign_rows, ignore_index=True)
    assign_df.to_csv(OUT / "sample_subtype_assignments.csv", index=False)
    pd.DataFrame(enrich_rows).to_csv(OUT / "phospho_subtype_enrichment.csv", index=False)
    cox_df_out = pd.DataFrame(cox_rows)
    cox_df_out.to_csv(OUT / "subtype_adjusted_cox_results.csv", index=False)
    (OUT / "subtype_definitions.json").write_text(json.dumps(definitions, indent=2), encoding="utf-8")

    # concise comparison table: M1 vs M3 HR
    pivot_rows = []
    for pair in PAIRS:
        ca, site = pair["cancer_type"], pair["site_label"]
        sub = cox_df_out[(cox_df_out.cancer_type == ca) & (cox_df_out.site_label == site)]
        row = {"cancer_type": ca, "site_label": site, "subtype_name": pair["subtype_name"]}
        for m in ["univariate", "protein_adjusted", "subtype_adjusted"]:
            r = sub[sub.model == m]
            if r.empty:
                continue
            r0 = r.iloc[0]
            row[f"{m}_status"] = r0.get("status")
            row[f"{m}_HR"] = r0.get("HR_phospho")
            row[f"{m}_p"] = r0.get("p_raw")
            row[f"{m}_n"] = r0.get("n")
            row[f"{m}_events"] = r0.get("events")
            if m == "subtype_adjusted":
                row["HR_subtype"] = r0.get("HR_subtype")
                row["p_subtype"] = r0.get("p_subtype")
        er = [e for e in enrich_rows if e["cancer_type"] == ca and e["site_label"] == site][0]
        row["fisher_OR"] = er["fisher_odds_ratio"]
        row["fisher_p"] = er["fisher_p"]
        pivot_rows.append(row)
    pd.DataFrame(pivot_rows).to_csv(OUT / "summary_M1_M2_M3subtype.csv", index=False)

    readme = f"""# Subtype-adjusted Cox (three priority hits)

Source survival groups: `{SURV.name}/sample_hotspot_activity_groups.csv`
(same median_nonmissing High/Low on mean_z hotspot activity).

## Subtype definitions
- **CCRCC STAT3**: BAP1 mutation from LinkedOmics meta (`BAP1_mut` vs `BAP1_wt`)
- **PDAC IRF9**: Moffitt-like basal vs classical from tumor RNA
  (basal panel mean z − classical panel mean z; ≥ cohort median → basal-like)
- **LUAD STAT3**: KRAS mutation from LinkedOmics meta (`KRAS_mut` vs `KRAS_wt`)

## Models
- univariate: phospho_binary
- protein_adjusted: phospho_binary + tf_protein
- subtype_adjusted: phospho_binary + tf_protein + subtype_binary

## Outputs
- `sample_subtype_assignments.csv`
- `phospho_subtype_enrichment.csv`
- `subtype_adjusted_cox_results.csv`
- `summary_M1_M2_M3subtype.csv`
- `subtype_definitions.json`
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print("Wrote", OUT)
    print(pd.DataFrame(pivot_rows).to_string(index=False))


if __name__ == "__main__":
    main()
