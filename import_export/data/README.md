# Import/Export subproject — data

Full tree and upload notes: **[../../DATA.md](../../DATA.md)**

## Directory summary

| Path | Purpose |
|------|---------|
| `cluster/` | Training CD-HIT clusters |
| `dataset_phos_site/` | Import/export positive sites |
| `fasta/` | TF FASTA sequences (copy of `functional/data/fasta`) |
| `TF_esm_embedding/` | ESM-2 embeddings (copy of `functional/data/TF_esm_embedding`) |
| `model_artifacts/` | Finalized IE fold artifacts + Platt calibrator (inference) |
| `precomputed/` | Bundled CSV inputs for figure / joint-score scripts |

Negatives, functional ensemble predictions, and manual features are loaded from `../functional/data/`.

When uploading **only** the import/export data bundle, include local `fasta/` and `TF_esm_embedding/` (populated by `scripts/populate_data.sh` from `functional/data/`).
