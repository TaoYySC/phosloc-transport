# Figures, supplementary figures, and prediction scripts

This document maps manuscript panels to scripts in the monorepo. Script names reflect **function**; panel labels are in each script's header comment.

All paths are relative to each subproject root (`functional/` or `import_export/`).

## Setup

1. Install dependencies from the repo root: `pip install -r requirements.txt`
2. Optional for Stage 3: uncomment `pyensembl` in `requirements.txt`, then rerun `pip install -r requirements.txt`
3. Provide data under `functional/data/`, `import_export/data/`, and `cptac_analysis/data/source/` — see **[DATA.md](../DATA.md)**.

Plot scripts read bundled inputs from `data/precomputed/` and `data/features/`.  
Inference uses `data/model_artifacts/`. New figures are written to `results/`.

## Localization-Regulatory Classifier (`functional/scripts/`)

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

## Localization Direction Classifier (`import_export/scripts/`)

| Panel | Script | Main inputs | Output directory |
|-------|--------|-------------|------------------|
| Figure 3b | `plot_import_export_model_performance.py` | `data/precomputed/.../metrics_all_runs.csv` | `results/1_transport_classifier_results/model_performance/` |
| Figure 3c | `calculate_joint_direction_score.py` | `functional/data/precomputed/...`, IE per-fold predictions in `data/precomputed/` | `results/1_transport_classifier_results/joint_score/` |
| Supp. Fig. 4a,b | `plot_import_export_score_distribution.py` | OOF + Localization-Regulatory Classifier ensemble predictions in `data/precomputed/` | `results/1_transport_classifier_results/esm_window_only_supcon_ce_import_pos_score_distribution_platt/` |
| Figure 3d; Supp. Fig. 4d | `plot_import_export_feature_panel.py` | `data/precomputed/.../joint_score/`, `../functional/data/features/` | `results/4_1_feature_boxplot_stacked_barplot/importexport_selected_panel_no_negative/` |
| **Prediction** | `predict_import_export_direction.py` | `data/model_artifacts/.../fold_artifacts/`, Platt calibrator | `results/1_transport_classifier_results/esm_window_only_import_pos_predictions/` |
| **Training** | `run_import_export_experiment.py` | Experiment YAML, cluster CSV, embeddings | `results/run_*/Import_vs_Export/` |

Example:

```bash
cd import_export
python scripts/plot_import_export_model_performance.py
python scripts/calculate_joint_direction_score.py --functional_score_threshold 0.6 --min_vote 4
python scripts/predict_import_export_direction.py --device cpu
```

## CPTAC target-regulation analysis (`cptac_analysis/scripts/`)

All paths are relative to `cptac_analysis/`. Requires `pyensembl` and a populated `data/source/` directory (see [cptac_analysis/data/README.md](../cptac_analysis/data/README.md)).

| Panel / output | Script | Main inputs | Output directory |
|----------------|--------|-------------|------------------|
| Integrated analysis | `run_import_target_regulation_analysis.py` | `data/source/`, stable import predictions, known positives | `results/import_target_regulation/` |
| Per-site across cancers | `plot_phosphosite_across_cancers.py` | `results/import_target_regulation/high_low_phospho_boxplots/*.csv` | `results/phosphosite_across_cancers_boxplots/` |
| Combined significant sites | `plot_significant_sites_combined.py` | same boxplot tables | `results/import_target_regulation/high_low_phospho_boxplots/all_significant_sites_combined/` |

Example:

```bash
cd cptac_analysis
export PYTHONPATH="${PWD}/scripts:${PYTHONPATH}"

python scripts/run_import_target_regulation_analysis.py
python scripts/plot_phosphosite_across_cancers.py --site-labels STAT3_Y705 STAT3_S701
python scripts/plot_significant_sites_combined.py
```
