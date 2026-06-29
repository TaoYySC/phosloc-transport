# Functional transport classifier

Stage 1 training pipeline for the PhosLoc-Transport repository.

This subproject trains a binary classifier that predicts whether a transcription factor phosphosite is likely to have **functional nuclear transport regulatory activity**, compared with background phosphosites. It does **not** predict import versus export direction; direction classification is handled by the [`import_export/`](../import_export/) subproject.

## Finalized model

| Field | Value |
|-------|-------|
| Task | Functional transport phosphosite classification |
| Feature set | `esm_window_site_pdb` — ESM-2 local window (31), center-site ESM embedding, and AlphaFold local graph features |
| Classifier | `esm_cnn2d_site_gnn` |
| Window size | 31 |
| Original run directory | `results/run_20260610_204935_ESM Window+Site+PDB/Functional_Transport/` |
| Run metadata | [`configs/runs/esm_window_site_pdb_run_meta.json`](configs/runs/esm_window_site_pdb_run_meta.json) |

## Train

```bash
cd functional
export PYTHONPATH="${PWD}:${PYTHONPATH}"

python scripts/1_1_run_experiment.py \
  --experiment_cfg configs/experiments/esm_window_site_pdb.yaml \
  --output_tag "ESM Window+Site+PDB"
```

## Config files

| File | Description |
|------|-------------|
| `configs/experiments/esm_window_site_pdb.yaml` | Experiment entry point: data paths, linked config files, and runtime settings (`device`, `output_dir`) |
| `configs/split.yaml` | Predefined held-out test split and 5-fold stratified group cross-validation on the development set |
| `configs/train.yaml` | `esm_cnn2d_site_gnn` architecture, optimization, and early-stopping settings |
| `configs/feature_sets.yaml` | Feature block definitions used by `esm_window_site_pdb`: ESM-2 window embeddings, center-site embedding, and AlphaFold graph inputs |
| `configs/runs/esm_window_site_pdb_run_meta.json` | Snapshot of metrics, fold selection, and paths from the finalized run |

## Data

Large feature files and intermediate inputs are **not** tracked in Git. Prepare or symlink the required files under `functional/data/` before training. See [`data/README.md`](data/README.md) for the expected directory layout.

## Outputs

Training writes model checkpoints, validation metrics, test metrics, and run metadata to the configured results directory (default: `results/`). The finalized run is stored at:

```
results/run_20260610_204935_ESM Window+Site+PDB/Functional_Transport/
```

## Related documentation

| Resource | Description |
|----------|-------------|
| [../docs/TRAINING_RUNS.md](../docs/TRAINING_RUNS.md) | Full reproduction details and hyperparameters |
| [../docs/FIGURES_AND_PREDICTION.md](../docs/FIGURES_AND_PREDICTION.md) | Figure and prediction scripts |
| [data/README.md](data/README.md) | Local data layout and file inventory |
