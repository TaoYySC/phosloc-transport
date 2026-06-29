# Import/export direction classifier

Stage 2 training and inference pipeline for the PhosLoc-Transport repository.

This subproject trains a binary classifier that predicts **nuclear import versus nuclear export** among annotated transport-positive transcription factor phosphosites (Import as the positive class). It does **not** classify functional transport activity against background phosphosites; that task is handled by the [`functional/`](../functional/) subproject.

## Finalized model

| Field | Value |
|-------|-------|
| Task | Import vs. export direction classification |
| Label convention | Import = 1, Export = 0 |
| Feature set | `import_export_esm_window_only_supcon_ce_import_pos` - ESM-2 local window (21) with PLS-reduced window embeddings |
| Model | `supcon_ce` with supervised contrastive loss and cross-entropy loss |
| Window size | 21 |
| Original run directory | `results/run_20260612_125646_esm_window_only_supcon_ce_import_pos/Import_vs_Export/` |
| Run metadata | [`configs/runs/esm_window_only_supcon_ce_import_pos_run_meta.json`](configs/runs/esm_window_only_supcon_ce_import_pos_run_meta.json) |

## Predict new sites

Run this model on transport-positive candidate sites, typically after Stage 1 functional prediction. The default prediction input is `../functional/data/dataset_phos_site/tf_all_phos_site_for_prediction.csv`. For custom inputs, provide at least `ACC_ID` and `POSITION`; `INDEX` is recommended as a stable site identifier.

```bash
cd import_export
export PYTHONPATH="${PWD}:${PYTHONPATH}"

python scripts/predict_import_export_direction.py \
  --input_csv ../functional/data/dataset_phos_site/tf_all_phos_site_for_prediction.csv \
  --output_csv results/1_transport_classifier_results/esm_window_only_import_pos_predictions/custom_import_export_predictions.csv \
  --device cpu \
  --save_dropped_csv results/1_transport_classifier_results/esm_window_only_import_pos_predictions/custom_dropped_rows.csv
```

Important options:

| Option | Description |
|--------|-------------|
| `--input_csv` | Input phosphosite table |
| `--fasta_path` | FASTA used to attach `FULL_SEQUENCE` |
| `--run_dir` | Saved fold artifacts and Platt calibrator |
| `--output_csv` | Ensemble prediction CSV path |
| `--device` | Use `cuda` or `cpu` |
| `--threshold` | Override decision threshold on `mean_prob_positive` |
| `--save_dropped_csv` | Save rows dropped during preprocessing |
| `--use_platt` / `--no-use_platt` | Enable or disable Platt calibration when the calibrator exists |

Main output columns include `mean_prob_import`, `std_prob_import`, `mean_prob_export`, `std_prob_export`, `threshold`, `positive_class`, `pred_label`, `pred_direction`, `feature_set`, and `model_name`. The script also writes:

| Output | Description |
|--------|-------------|
| `*_per_fold.csv` | Wide table of fold-level import/export probabilities |
| `*_run_meta.json` | Input paths, feature set, calibration status, threshold, and preprocessing counts |
| dropped-row CSV | Optional table containing sites removed for missing sequence, invalid position, non-STY residue, or missing ESM embedding |

## Joint score and stable predictions

`scripts/calculate_joint_direction_score.py` combines Stage 1 functional ensemble scores with Stage 2 import/export predictions. The stable prediction files used by feature-panel plots and the CPTAC analysis are stored under:

```text
data/precomputed/1_transport_classifier_results/joint_score/
```

The reference stable files use probability and fold-vote filters encoded in the filename, for example `predicted_import_stable_gt0p6_vote4.csv` means sites passing the selected score threshold (`gt0p6`) and at least four supporting fold votes (`vote4`).

## Train

```bash
cd import_export
export PYTHONPATH="${PWD}:${PYTHONPATH}"

python scripts/run_import_export_experiment.py \
  --experiment_cfg configs/experiments/import_export_esm_window_only_supcon_ce_import_pos.yaml \
  --output_tag esm_window_only_supcon_ce_import_pos
```

## Config files

| File | Description |
|------|-------------|
| `configs/experiments/import_export_esm_window_only_supcon_ce_import_pos.yaml` | Experiment entry point: data paths, linked config files, and runtime settings (`device`, `output_dir`) |
| `configs/split.yaml` | 5-fold stratified group cross-validation on annotated import/export positives |
| `configs/train_supcon_ce_window_only.yaml` | `supcon_ce` architecture, SupCon+CE optimization, and training settings |
| `configs/feature_sets_esm_window_only_supcon_ce.yaml` | Feature block definitions used by `import_export_esm_window_only_supcon_ce_import_pos`: ESM-2 window embeddings with PLS reduction |
| `configs/runs/esm_window_only_supcon_ce_import_pos_run_meta.json` | Snapshot of metrics, fold selection, and paths from the finalized run |

## Data

Large feature files, model artifacts, and intermediate inputs are **not** tracked in Git. Prepare or symlink the required files under `import_export/data/` before training or prediction; some inputs are shared with or copied from `functional/data/`. See [`data/README.md`](data/README.md) and [`../DATA.md`](../DATA.md) for the expected directory layout.

Required prediction resources:

| Resource | Default path |
|----------|--------------|
| Input site CSV | `../functional/data/dataset_phos_site/tf_all_phos_site_for_prediction.csv` |
| FASTA | `data/fasta/transcription_fasta.fasta` |
| ESM embeddings | `data/TF_esm_embedding/` |
| Model artifacts | `data/model_artifacts/run_20260612_125646_esm_window_only_supcon_ce_import_pos/Import_vs_Export/` |
| Platt calibrator | `data/model_artifacts/run_20260612_125646_esm_window_only_supcon_ce_import_pos/Import_vs_Export/platt_calibrator.json` |

## Outputs

Training writes model checkpoints, fold-level metrics, cross-validation summaries, and run metadata to the configured results directory (default: `results/`). The finalized run is stored at:

```text
results/run_20260612_125646_esm_window_only_supcon_ce_import_pos/Import_vs_Export/
```

Prediction writes ensemble and per-fold tables to:

```text
results/1_transport_classifier_results/esm_window_only_import_pos_predictions/
```

## Notes and limitations

- This model assumes the input sites are plausible transport-regulatory candidates.
- The reference positive class is import; if a future run uses export as the positive class, interpret `mean_prob_positive` through the `positive_class` column.
- Platt-calibrated probabilities are used when the saved calibrator is available and `--use_platt` is enabled.

## Related documentation

| Resource | Description |
|----------|-------------|
| [../docs/TRAINING_RUNS.md](../docs/TRAINING_RUNS.md) | Full reproduction details and hyperparameters |
| [../docs/FIGURES_AND_PREDICTION.md](../docs/FIGURES_AND_PREDICTION.md) | Figure and prediction scripts |
| [data/README.md](data/README.md) | Local data layout and file inventory |
