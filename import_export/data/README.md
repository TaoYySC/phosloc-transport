# Direction-classifier subproject - data

Full tree and upload notes: **[../../DATA.md](../../DATA.md)**.

Public data-download links are intentionally left blank while the data upload is in progress.

| Bundle | Target path | Download / DOI |
|--------|-------------|----------------|
| Direction-classifier data bundle | `import_export/data/` | TBD |
| Shared functional data bundle | `functional/data/` | TBD |

## Directory summary

| Path | Purpose |
|------|---------|
| `cluster/` | Training CD-HIT clusters |
| `dataset_phos_site/` | Direction-labeled transport-positive sites |
| `fasta/` | TF FASTA sequences (copy of `functional/data/fasta`) |
| `TF_esm_embedding/` | ESM-2 embeddings (copy of `functional/data/TF_esm_embedding`) |
| `model_artifacts/` | Finalized IE fold artifacts + Platt calibrator (inference) |
| `precomputed/` | Bundled CSV inputs for figure / joint-score scripts |

Negatives, functional ensemble predictions, and manual features are loaded from `../functional/data/`.

When uploading **only** the direction-classifier data bundle, include local `fasta/` and `TF_esm_embedding/` (populated by `scripts/populate_data.sh` from `functional/data/`).

## Prediction input

The default prediction script reads `../functional/data/dataset_phos_site/tf_all_phos_site_for_prediction.csv`.
For custom prediction tables, provide `ACC_ID` and `POSITION`; `INDEX` is recommended.
Rows are dropped if sequence, valid STY position, or ESM embedding resources are missing.
