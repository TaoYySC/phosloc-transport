# Functional subproject - data

Full tree and upload notes: **[../../DATA.md](../../DATA.md)**.

The processed data bundle is available from Zenodo:
[`10.5281/zenodo.21021066`](https://doi.org/10.5281/zenodo.21021066).

| Bundle | Target path | Download / DOI |
|--------|-------------|----------------|
| Functional data bundle | `functional/data/` | [Zenodo DOI: 10.5281/zenodo.21021066](https://doi.org/10.5281/zenodo.21021066) |

## Directory summary

| Path | Purpose |
|------|---------|
| `cluster/` | Training CD-HIT clusters |
| `dataset_phos_site/` | Site tables (train / plot / predict) |
| `fasta/` | Transcription-factor sequences |
| `features/` | Precomputed manual feature CSVs |
| `TF_esm_embedding/` | ESM-2 window embeddings |
| `alphafold_tf_pdb/` | AlphaFold structure PDBs |
| `TF_family/` | TF family metadata for dataset plots |
| `hpa/` | Reserved for optional Human Protein Atlas inputs (empty by default) |
| `model_artifacts/` | Finalized functional model checkpoints (inference) |
| `precomputed/` | Bundled CSV inputs for figure scripts |

Training reads paths from `configs/experiments/esm_window_site_pdb.yaml`.
Plot scripts read from `data/precomputed/` and `data/features/`.
`predict_functional_transport.py` reads from `data/model_artifacts/`.

## Prediction input

The default prediction table is `dataset_phos_site/tf_all_phos_site_for_prediction.csv`.
For custom prediction tables, provide `ACC_ID` and `POSITION`; `INDEX` is recommended.
If `FULL_SEQUENCE` is absent, it is attached from `fasta/transcription_fasta.fasta`.
