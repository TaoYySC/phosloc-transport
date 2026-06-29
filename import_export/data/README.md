# Import/Export subproject — data

Full tree and upload notes: **[../../DATA.md](../../DATA.md)**

## Directory summary

| Path | Purpose |
|------|---------|
| `cluster/` | Training CD-HIT clusters |
| `dataset_phos_site/` | Import/export positive sites |
| `fasta/` | Symlink → `../../functional/data/fasta` |
| `TF_esm_embedding/` | Symlink → `../../functional/data/TF_esm_embedding` |
| `model_artifacts/` | Finalized IE fold artifacts + Platt calibrator (inference) |
| `precomputed/` | Bundled CSV inputs for figure / joint-score scripts |

Negatives, functional ensemble predictions, and manual features are loaded from `../functional/data/`.

When uploading **only** the import/export data bundle, include the linked `functional/data/fasta` and `functional/data/TF_esm_embedding` directories (or replace symlinks with copies).
