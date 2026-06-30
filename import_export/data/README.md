# Localization Direction Classifier subproject - data

Full tree and upload notes: **[../../DATA.md](../../DATA.md)**.

The processed data bundles are available from the shared Zenodo record:
[`10.5281/zenodo.21064685`](https://doi.org/10.5281/zenodo.21064685).

| Bundle | Target path | Download / DOI |
|--------|-------------|----------------|
| Localization Direction Classifier data bundle | `import_export/data/` | [Zenodo DOI: 10.5281/zenodo.21064685](https://doi.org/10.5281/zenodo.21064685) |
| Shared functional data bundle | `functional/data/` | [Zenodo DOI: 10.5281/zenodo.21064685](https://doi.org/10.5281/zenodo.21064685) |

## Directory summary

| Path | Purpose |
|------|---------|
| `cluster/` | Training CD-HIT clusters |
| `dataset_phos_site/` | Direction-labeled transport-positive sites |
| `fasta/` | TF FASTA sequences (copy of `functional/data/fasta`) |
| `TF_esm_embedding/` | ESM-2 embeddings (copy of `functional/data/TF_esm_embedding`) |
| `model_artifacts/` | Finalized IE fold artifacts + Platt calibrator (inference) |
| `precomputed/` | Bundled CSV inputs for figure / joint-score scripts |

Negatives, Localization-Regulatory Classifier ensemble predictions, and manual features are loaded from `../functional/data/`.

When uploading **only** the Localization Direction Classifier data bundle, include local `fasta/` and `TF_esm_embedding/` (populated by `scripts/populate_data.sh` from `functional/data/`).

## Prediction input

The default prediction script reads `../functional/data/dataset_phos_site/tf_all_phos_site_for_prediction.csv`.
For custom prediction tables, provide `ACC_ID` and `POSITION`; `INDEX` is recommended.
Rows are dropped if sequence, valid STY position, or ESM embedding resources are missing.
