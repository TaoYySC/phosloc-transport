# Figures, supplementary figures, and prediction scripts

This document maps manuscript panels to scripts in the monorepo. Script names reflect **function**; panel labels are in each script's header comment.

All paths are relative to each subproject root (`functional/` or `import_export/`).

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
| Figure 2b | `plot_model_ablation_comparison.py` | `data/precomputed/run_20260610_*/*/metrics_all_folds.csv` | `results/2_1_functional_classifier_results/model_ablation_comparison/` |
| Figure 2c | `benchmark_funcphos_str_seq.py` | `data/precomputed/` (fixed test + FuncPhos scores) | `results/2_1_functional_classifier_results/benchmark_results/` |
| Supp. Fig. 2b | `plot_functional_score_distribution.py` | `data/precomputed/.../predictions/`, site tables | `results/2_1_functional_classifier_results/distribution/` |
| Supp. Fig. 2c,d,e | `plot_functional_validation_scores.py` | `data/precomputed/.../predictions/`, cluster table | `results/2_1_functional_classifier_results/single_model_rank_eval_5_folds_ensemble/` |
| Supp. Fig. 3a–i | `plot_functional_feature_panel.py` | `data/precomputed/.../predictions/`, `data/features/` | `results/2_1_functional_classifier_results/feature_boxplot_stacked_barplot/functional_selected_panel/` |
| **Prediction** | `predict_functional_transport.py` | `data/model_artifacts/.../artifacts/`, site CSV, FASTA | `results/2_1_functional_classifier_results/predictions/` |
| **Training** | `1_1_run_experiment.py` | Experiment YAML, cluster CSV, embeddings/PDB | `results/run_*/Functional_Transport/` |

Example:

```bash
cd functional
python scripts/plot_model_ablation_comparison.py
python scripts/predict_functional_transport.py --device cpu
```

## Import/export direction classifier (`import_export/scripts/`)

| Panel | Script | Main inputs | Output directory |
|-------|--------|-------------|------------------|
| Figure 3b | `plot_import_export_model_performance.py` | `data/precomputed/.../metrics_all_runs.csv` | `results/1_transport_classifier_results/model_performance/` |
| Figure 3c | `calculate_joint_direction_score.py` | `functional/data/precomputed/...`, IE per-fold predictions in `data/precomputed/` | `results/1_transport_classifier_results/joint_score/` |
| Supp. Fig. 4a,b | `plot_import_export_score_distribution.py` | OOF + functional ensemble predictions in `data/precomputed/` | `results/1_transport_classifier_results/esm_window_only_supcon_ce_import_pos_score_distribution_platt/` |
| Figure 3d; Supp. Fig. 4d | `plot_import_export_feature_panel.py` | `data/precomputed/.../joint_score/`, `../functional/data/features/` | `results/4_1_feature_boxplot_stacked_barplot/importexport_selected_panel_no_negative/` |
| **Prediction** | `predict_import_export_direction.py` | `data/model_artifacts/.../fold_artifacts/`, Platt calibrator | `results/1_transport_classifier_results/esm_window_only_import_pos_predictions/` |
| **Training** | `run_import_export_experiment.py` | Experiment YAML, cluster CSV, embeddings | `results/run_*/Import_vs_Export/` |

Example:

```bash
cd import_export
python scripts/plot_import_export_model_performance.py
python scripts/calculate_joint_direction_score.py
python scripts/predict_import_export_direction.py --device cpu
```

## CPTAC phospho-hotspot analysis (`cptac_analysis/scripts/`)

All paths are relative to `cptac_analysis/`. Requires `pyensembl` and a populated `data/source/` directory (see [cptac_analysis/data/README.md](../cptac_analysis/data/README.md)).

**Manuscript primary analysis is hotspot-level.** Figure 4 covers spatial / direction
definition (4A–C) and CPTAC association panels; Figure 5 is the four-arm two-sided
concordance heatmap. Supp. Fig. 2e enrichment stars use **unadjusted permutation P**
(`* P < 0.05`); BH q values are listed in Table S3.

| Panel / output | Script | Main inputs | Output directory |
|----------------|--------|-------------|------------------|
| Hotspot catalog (Mixed/Predicted) | `build_hotspots_mixed_pred_filter.py` | FuncTransport+Direction summary (v11) | `results/.../hotspots_mixed_pred_filter/` |
| Pure nuclear/cyto gates | `reannotate_hotspot_pure_direction.py` | mixed_pred_filter catalog | `.../hotspots_mixed_pred_filter_pure_direction/` |
| CPTAC hotspot scan | `run_hotspot_target_regulation_analysis.py` | pure-direction catalog, `data/source/` | per-arm trees under `results/hotspot_*` |
| Figure **4b–4f** | `plot_hotspot_figure4.py` | hotspot CPTAC results | `results/hotspot_mixed_pred_filter_pure_mean_z_median_20260914/figure4/` |
| Figure **4A–C** / S5 | `run_figure4_spatial_direction_panels.py` | summary + mixed_pred_filter catalog | `results/figure4_spatial_direction_panels_v11_147pos_d3_platt/` |
| Figure **5** four-arm heatmap | `plot_two_sided_four_arm_significance_heatmap.py` | two-sided four-arm results | `results/hotspot_mixed_pred_filter_pure_regulon_only_twosided_20260929/combined_four_arm_heatmap/` |
| Table **S4** associations | `analyze_two_sided_concordance.py` | concordance flags CSV | `../supplement/Supplemental_Table_4.xlsx` |
| Table **S3** co-regulatory enrichment | `../functional/scripts/plot_functional_validation_scores.py` | reported co-regulatory (cluster-sheet) sites | `../supplement/Supplemental_Table_3.xlsx` |
| Hotspot Cox / KM | `run_hotspot_survival_cox.py` | BH-significant cancer×hotspot pairs | `results/survival_analysis/hotspot_cox_*` |
| Legacy unit-site scan | `run_import_target_regulation_analysis.py` | predicted import sites | `results/import_target_regulation/` |

Example (catalog + Figure 5):

```bash
cd cptac_analysis

python scripts/build_hotspots_mixed_pred_filter.py
python scripts/reannotate_hotspot_pure_direction.py
python scripts/analyze_two_sided_concordance.py
python scripts/plot_two_sided_four_arm_significance_heatmap.py
python scripts/run_figure4_spatial_direction_panels.py
```

See [cptac_analysis/README.md](../cptac_analysis/README.md) for catalog rules, Figure 5b counts, and legacy unit-site commands.
