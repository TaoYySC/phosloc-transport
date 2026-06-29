# PhosLoc-Transport

Machine-learning pipelines for transcription-factor phosphosite nuclear transport prediction.

This monorepo contains **three analysis stages**:

| Subproject | Task |
|------------|------|
| [`functional/`](functional/) | Functional transport phosphosite classification |
| [`import_export/`](import_export/) | Import vs. export direction classification |
| [`cptac_analysis/`](cptac_analysis/) | CPTAC multi-omics target-regulation analysis |

## Requirements

- **Python** 3.10.13
- **Core dependencies** ([`requirements.txt`](requirements.txt)): `numpy==2.0.2`, `pandas==2.2.2`, `scipy==1.13.1`, `scikit-learn==1.6.1`, `matplotlib==3.9.4`, `seaborn==0.13.2`, `PyYAML==6.0.2`, `joblib==1.4.2`, `tqdm==4.67.1`, `xgboost==2.1.4`, `torch==2.6.0`, `torch-geometric==2.6.1` (GPU training validated with `torch==2.6.0+cu124`, CUDA 12.4)
- **GPU** acceleration is recommended for ESM embedding extraction and AlphaFold graph–based model training

**Optional** ([`requirements-optional.txt`](requirements-optional.txt)): `fair-esm==2.0.1`, `umap-learn==0.5.7`, `pyensembl` (Stage 3 CPTAC analysis) — not needed to reproduce the finalized runs if precomputed features are provided.

## Quick start

```bash
cd phosloc-transport
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional: pip install -r requirements-optional.txt
```

Place or symlink local data under `functional/data/` and `import_export/data/` (see [Data availability](#data-availability)).  
New training outputs are written to each subproject's `results/` directory.

### Train functional transport

```bash
cd functional
export PYTHONPATH="${PWD}:${PYTHONPATH}"

python scripts/1_1_run_experiment.py \
  --experiment_cfg configs/experiments/esm_window_site_pdb.yaml \
  --output_tag "ESM Window+Site+PDB"
```

### Train import vs. export

```bash
cd import_export
export PYTHONPATH="${PWD}:${PYTHONPATH}"

python scripts/run_import_export_experiment.py \
  --experiment_cfg configs/experiments/import_export_esm_window_only_supcon_ce_import_pos.yaml \
  --output_tag esm_window_only_supcon_ce_import_pos
```

### Run CPTAC target-regulation analysis (Stage 3)

See **[cptac_analysis/README.md](cptac_analysis/README.md)** for data setup (`cptac_analysis/data/source/`), `pyensembl`, and analysis commands.

## Data availability

Large feature files and intermediate data are **not** tracked in Git. Prepare or symlink local data directories according to the data README files in each subproject:

- [`functional/data/README.md`](functional/data/README.md)
- [`import_export/data/README.md`](import_export/data/README.md)
- [`cptac_analysis/data/README.md`](cptac_analysis/data/README.md)

Full inventory and upload notes: **[DATA.md](DATA.md)**

Reproducing the finalized runs requires processed feature files, training splits, model configs, and run metadata snapshots bundled under each subproject's `data/` and `configs/` trees.

## Repository layout

```
phosloc-transport/
├── DATA.md                      # complete data inventory
├── requirements.txt
├── requirements-optional.txt    # fair-esm, umap-learn (optional)
├── docs/
├── scripts/                     # populate_data.sh (data migration helper)
├── functional/
│   ├── data/                    # functional inputs (+ model_artifacts, precomputed)
│   ├── scripts/
│   ├── configs/
│   ├── src/
│   └── results/                 # runtime outputs (optional)
├── import_export/
│   ├── data/
│   ├── scripts/
│   ├── configs/
│   ├── src/
│   └── results/
└── cptac_analysis/
    ├── data/                    # CPTAC / ChIP / regulon inputs (symlink)
    ├── scripts/
    └── results/                 # integrated pipeline + boxplot outputs
```

## Reproducibility

The finalized training runs are recorded below.

| Pipeline | Original run |
|----------|--------------|
| Functional transport | `run_20260610_204935_ESM Window+Site+PDB` |
| Import/export direction | `run_20260612_125646_esm_window_only_supcon_ce_import_pos` |
| CPTAC target-regulation analysis | `results/import_target_regulation/` (see [cptac_analysis/README.md](cptac_analysis/README.md)) |

Run metadata snapshots for Stages 1–2 are stored under each subproject's `configs/runs/` directory.

## Documentation

| Resource | Description |
|----------|-------------|
| [docs/TRAINING_RUNS.md](docs/TRAINING_RUNS.md) | Full reproduction details and hyperparameters |
| [docs/FIGURES_AND_PREDICTION.md](docs/FIGURES_AND_PREDICTION.md) | Figure and prediction scripts |
| [DATA.md](DATA.md) | Train / plot / predict data layout |
| [functional/README.md](functional/README.md) | Functional pipeline overview |
| [import_export/README.md](import_export/README.md) | Import/export pipeline overview |
| [cptac_analysis/README.md](cptac_analysis/README.md) | CPTAC cancer analysis overview |

## Citation

If you use PhosLoc-Transport, please cite the associated manuscript once available.

## License

TBD
