# Import/export direction classifier

Stage 2 training pipeline for the PhosLoc-Transport repository.

This subproject trains a binary classifier that predicts **nuclear import versus nuclear export** among annotated transport-positive transcription factor phosphosites (Import as the positive class). It does **not** classify functional transport activity against background phosphosites; that task is handled by the [`functional/`](../functional/) subproject.

## Finalized model

| Field | Value |
|-------|-------|
| Task | Import vs. export direction classification |
| Label convention | Import = 1, Export = 0 |
| Feature set | `import_export_esm_window_only_supcon_ce_import_pos` — ESM-2 local window (21) with PLS-reduced window embeddings |
| Model | `supcon_ce` with supervised contrastive loss and cross-entropy loss |
| Window size | 21 |
| Original run directory | `results/run_20260612_125646_esm_window_only_supcon_ce_import_pos/Import_vs_Export/` |
| Run metadata | [`configs/runs/esm_window_only_supcon_ce_import_pos_run_meta.json`](configs/runs/esm_window_only_supcon_ce_import_pos_run_meta.json) |

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

Large feature files and intermediate inputs are **not** tracked in Git. Prepare or symlink the required files under `import_export/data/` before training; some inputs are shared with or symlinked from `functional/data/`. See [`data/README.md`](data/README.md) for the expected directory layout.

## Outputs

Training writes model checkpoints, fold-level metrics, cross-validation summaries, and run metadata to the configured results directory (default: `results/`). The finalized run is stored at:

```
results/run_20260612_125646_esm_window_only_supcon_ce_import_pos/Import_vs_Export/
```

## Related documentation

| Resource | Description |
|----------|-------------|
| [../docs/TRAINING_RUNS.md](../docs/TRAINING_RUNS.md) | Full reproduction details and hyperparameters |
| [../docs/FIGURES_AND_PREDICTION.md](../docs/FIGURES_AND_PREDICTION.md) | Figure and prediction scripts |
| [data/README.md](data/README.md) | Local data layout and file inventory |
