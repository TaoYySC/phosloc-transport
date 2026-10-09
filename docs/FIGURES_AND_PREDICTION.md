# Figures, supplementary figures, and prediction scripts

This document maps manuscript panels to scripts in the monorepo. Script names reflect **function**; panel labels are in each script's header comment.

All paths are relative to each subproject root (`functional/` or `import_export/`).

Default commands use the manuscript v11 Stage 1 and D3 Stage 2 runs. Earlier runs are retained as legacy versions.

## Setup

1. Install dependencies from the repo root: `pip install -r requirements.txt`
2. Optional: uncomment and install `pyensembl` in [`requirements.txt`](../requirements.txt) (required for Stage 3)
3. Provide data under `functional/data/`, `import_export/data/`, and `cptac_analysis/data/source/` — see **[DATA.md](../DATA.md)**.

Plot scripts read bundled inputs from `data/precomputed/` and `data/features/`.  
Inference uses `data/model_artifacts/`. New figures are written to `results/`.

## Functional transport classifier (`functional/scripts/`)

| Panel | Script | Main inputs | Output directory |
|-------|--------|-------------|------------------|
| Figure 1c,d,e; Supp. Fig. 1a,b | `plot_dataset_description.py` | `data/dataset_phos_site/TF_positive_phos_site_0608.csv`, `data/TF_family/TF_Information.txt` | `results/0_dataset_description/` |
| Figure 2b | `plot_model_ablation_comparison.py` | `data/precomputed/` metrics (v11 / legacy) | `results/2_1_functional_classifier_results/model_ablation_comparison/` |
| Figure 2c | `benchmark_funcphos_str_seq.py` | `data/precomputed/` (fixed test + FuncPhos scores) | `results/2_1_functional_classifier_results/benchmark_results/` |
| Supp. Fig. 2b | `plot_functional_score_distribution.py` | `data/precomputed/.../predictions/`, site tables | `results/2_1_functional_classifier_results/distribution/` |
| Supp. Fig. 2c,d,e | `plot_functional_validation_scores.py` | `v11_147pos_5_folds_ensemble_predictions.csv`, cluster table | `results/2_1_functional_classifier_results/single_model_rank_eval_5_folds_ensemble/` |
| Supp. Fig. 3a–i | `plot_functional_feature_panel.py` | `data/precomputed/.../predictions/`, `data/features/` | `results/2_1_functional_classifier_results/feature_boxplot_stacked_barplot/functional_selected_panel/` |
| **Prediction** | `predict_functional_transport.py` | v11 model artifacts, site CSV, FASTA | `results/2_1_functional_classifier_results/predictions/` |
| **Training** | `1_1_run_experiment.py` | Experiment YAML, cluster CSV, embeddings/PDB | `results/run_*/Functional_Transport/` |

Figure S2e uses unadjusted empirical one-sided permutation P values for significance annotations (1,000 permutations; *P < 0.05).

Table S3 reports unadjusted empirical one-sided P values from 1,000 within-protein permutations with a plus-one correction.

Example:

```bash
cd functional
python scripts/plot_model_ablation_comparison.py
python scripts/predict_functional_transport.py --device cpu
```

## Import/export direction classifier (`import_export/scripts/`)

| Panel | Script | Main inputs | Output directory |
|-------|--------|-------------|------------------|
| Figure 3b | `plot_import_export_model_performance.py` | D3 / legacy metrics in `data/precomputed/` | `results/1_transport_classifier_results/model_performance/` |
| Figure 3c | `calculate_joint_direction_score.py` | `v11_147pos_5_folds_ensemble_predictions.csv`, D3 Platt per-fold predictions | `results/1_transport_classifier_results/joint_score_v11_147pos_d3_platt/` |
| Supp. Fig. 4a,b | `plot_import_export_score_distribution.py` | OOF + functional ensemble predictions in `data/precomputed/` | `results/1_transport_classifier_results/` |
| Figure 3d; Supp. Fig. 4d | `plot_import_export_feature_panel.py` | `joint_score_v11_147pos_d3_platt/`, `../functional/data/features/` | `results/4_1_feature_boxplot_stacked_barplot/importexport_selected_panel_no_negative/` |
| **Prediction** | `predict_import_export_direction.py` | D3 fold artifacts + Platt calibrator | `results/1_transport_classifier_results/d3_kpls_gauto_predictions_platt/` |
| **Training** | `run_import_export_experiment.py` | `import_export_ie147_D3_kpls_gauto.yaml` | `results/run_20260904_134712_ie147_R3D_D3_kpls_gauto/Import_vs_Export/` |

Example:

```bash
cd import_export
python scripts/plot_import_export_model_performance.py
python scripts/calculate_joint_direction_score.py
python scripts/predict_import_export_direction.py --device cpu
```

## CPTAC phospho-hotspot analysis (`cptac_analysis/scripts/`)

All paths are relative to `cptac_analysis/`. Requires `pyensembl` and a populated `data/source/` directory (see [cptac_analysis/data/README.md](../cptac_analysis/data/README.md)).

**Manuscript primary analysis is hotspot-level** (`build_hotspots_mixed_pred_filter.py`).

- Figure 4c: representative phospho-hotspots in STAT1 (T699 to S708), AR (S647 to T653) and CUX1 (Y1209 to S1218).
- Figure 5: hotspot-associated target-gene expression across four categories defined by localization direction and target regulation (Nuclear accumulation × activate/repress and cytoplasmic redistribution × activate/repress).

The primary analysis uses two-sided Wilcoxon signed-rank tests on paired target-gene expression summaries. Associations require at least 10 paired target genes. BH correction is applied within each cancer type and localization direction × target regulation category.

| Panel / output | Script | Main inputs | Output directory |
|----------------|--------|-------------|------------------|
| Hotspot catalog (Mixed/Predicted; **Methods**) | `build_hotspots_mixed_pred_filter.py` | FuncTransport+Direction summary (v11) | `results/.../hotspots_mixed_pred_filter/` |
| Pure nuclear/cyto gates | `reannotate_hotspot_pure_direction.py` | mixed_pred_filter catalog | `.../hotspots_mixed_pred_filter_pure_direction/` |
| CPTAC hotspot scan (regulon_only, two-sided) | `run_hotspot_target_regulation_analysis.py` / `run_four_arm_twosided_pipeline.sh` | pure-direction catalog, `data/source/` | `results/hotspot_mixed_pred_filter_pure_regulon_only_twosided_20260929/` |
| Figure **4A–C** / S5 | `run_figure4_spatial_direction_panels.py` | summary + mixed_pred_filter catalog | `results/figure4_spatial_direction_panels_v11_147pos_d3_platt/` |
| Figure **5** four-arm heatmap | `plot_two_sided_four_arm_significance_heatmap.py` | two-sided four-arm results | `.../combined_four_arm_heatmap/` |
| Figure **5b** concordance | `analyze_two_sided_concordance.py` | four-arm CPTAC tables | `.../concordance_direction_summary/` |
| Table **S4** associations | `analyze_two_sided_concordance.py` | concordance flags CSV | `../supplement/Supplemental_Table_4.xlsx` |
| Table **S3** co-regulatory enrichment | `../functional/scripts/plot_functional_validation_scores.py` | reported co-regulatory (cluster-sheet) sites | `../supplement/Supplemental_Table_3.xlsx` |
| Hotspot Cox / KM | `run_hotspot_survival_cox.py` | BH-significant cancer×hotspot pairs | `results/survival_analysis/hotspot_cox_*` |
| Legacy unit-site scan | `run_import_target_regulation_analysis.py` / `plot_phosphosite_across_cancers.py` | unit-site CPTAC outputs | `results/import_target_regulation/` |

Table S4 lists all evaluable cancer–hotspot associations, including raw P values, BH-adjusted q values and BH significance flags.

Example (catalog + Figure 5 + Figure 4c):

```bash
cd cptac_analysis
bash scripts/run_four_arm_twosided_pipeline.sh
python scripts/run_figure4_spatial_direction_panels.py --only 4c
```

See [cptac_analysis/README.md](../cptac_analysis/README.md) for Methods catalog rules, Figure 5b counts, and legacy unit-site commands.
