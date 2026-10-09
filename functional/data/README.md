# Functional subproject - data

Full tree and upload notes: **[../../DATA.md](../../DATA.md)**.

Default commands use the manuscript v11 Stage 1 and D3 Stage 2 runs. Earlier runs are retained as legacy versions.

The processed data bundle is available from Zenodo:
[`10.5281/zenodo.21064685`](https://doi.org/10.5281/zenodo.21064685)
(plus the v11 hotspot inputs add-on for manuscript predictions).

| Bundle | Target path | Download / DOI |
|--------|-------------|----------------|
| Functional data bundle | `functional/data/` | [Zenodo DOI: 10.5281/zenodo.21064685](https://doi.org/10.5281/zenodo.21064685) |

## Directory summary

| Path | Purpose |
|------|---------|
| `cluster/` | Training CD-HIT clusters |
| `dataset_phos_site/` | Site tables (147 positives manuscript; train / plot / predict) |
| `fasta/` | Transcription-factor sequences |
| `features/` | Precomputed manual feature CSVs |
| `TF_esm_embedding/` | ESM-2 window embeddings |
| `alphafold_tf_pdb/` | AlphaFold structure PDBs |
| `TF_family/` | TF family metadata for dataset plots |
| `hpa/` | Reserved for optional Human Protein Atlas inputs (empty by default) |
| `model_artifacts/` | Preferred: v11 Stage 1 checkpoints under `run_20260831_183836_..._v11_xlarge/` |
| `precomputed/` | Includes `v11_147pos_5_folds_ensemble_predictions.csv` (manuscript) |

Training reads paths from `configs/experiments/esm_window_site_pdb.yaml`.
Plot scripts read from `data/precomputed/` and `data/features/`.
`predict_functional_transport.py` reads from `data/model_artifacts/`.

## Prediction input

The default prediction table is `dataset_phos_site/tf_all_phos_site_for_prediction.csv`.
For custom prediction tables, provide `ACC_ID` and `POSITION`; `INDEX` is recommended.
If `FULL_SEQUENCE` is absent, it is attached from `fasta/transcription_fasta.fasta`.
