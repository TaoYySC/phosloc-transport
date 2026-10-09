# Data layout

All inputs required to **train**, **plot**, and **predict** live under each subproject's `data/` directory.
Generated figures and new run outputs still go to `results/` (not shipped with the data bundle by default).

Default commands use the manuscript v11 Stage 1 and D3 Stage 2 runs. Earlier runs are retained as legacy versions.

| Item | Manuscript version |
|------|--------------------|
| Stage 1 run | `run_20260831_183836_ESM Window+Site+PDB_147pos_roc08_v11_xlarge` |
| Stage 2 run | `run_20260904_134712_ie147_R3D_D3_kpls_gauto` |
| Stage 1 predictions | `v11_147pos_5_folds_ensemble_predictions.csv` |
| Joint direction scores | `joint_score_v11_147pos_d3_platt/` |

Upload the **code repo** and **data directories** separately if needed.
All processed data bundles are available from one Zenodo record:
[`10.5281/zenodo.21064685`](https://doi.org/10.5281/zenodo.21064685).
The root-level `PhosLoc-Transport_DATA_README.txt` is intended as the README
shipped beside the Zenodo archives. This file is the canonical in-repository
data inventory.

| Upload unit | Path | Approx. size | Download / DOI |
|-------------|------|--------------|----------------|
| Code | repo root (exclude `**/data/` large dirs) | ~55 MB | GitHub repository |
| Base data bundles | `functional/data/`, `import_export/data/`, `cptac_analysis/data/source/` | ~12 GB | [Zenodo DOI: 10.5281/zenodo.21064685](https://doi.org/10.5281/zenodo.21064685) |
| v11 hotspot inputs (add-on) | see archive layout below | ~5 MB | Same Zenodo record (new version; upload `PhosLoc-Transport_v11_hotspot_inputs.tar.gz`) |
| Fig. 5 results (optional) | `cptac_analysis/results/hotspot_mixed_pred_filter_pure_regulon_only_twosided_20260929/` | ~122 MB packed | Same Zenodo record (`PhosLoc-Transport_fig5_twosided_results.tar.gz`) |

**New Zenodo files prepared locally** under `zenodo_upload/` (draft upload in progress; replace this note with the published version DOI after Publish):

| Archive | Extracts to | Contents |
|---------|-------------|----------|
| `PhosLoc-Transport_v11_hotspot_inputs.tar.gz` | repo root prefixes | 147-pos set, cluster-sheet co-regulatory sites, v11 ensemble predictions, v11 joint scores + summary, hotspot catalogs |
| `PhosLoc-Transport_fig5_twosided_results.tar.gz` | `cptac_analysis/results/...twosided_20260929/` | Optional four-arm two-sided result snapshot |
| `PhosLoc-Transport_DATA_README.txt` | beside archives | Updated install notes (repo URL: `phosloc-transport`) |

Until the new version is published, cite the existing concept DOI and note that v11/hotspot add-ons are required for Figure 5 / Tables S3–S4.

## `functional/data/`

```text
functional/data/
|-- cluster/
|   `-- func_train_window31_c70_cluster.csv          # training clusters
|-- dataset_phos_site/
|   |-- TF_positive_phos_site_0608.csv               # 147 positives (manuscript)
|   |-- TF_deepmvp_negative_phos_site_tf_only.csv    # training / plots
|   |-- co_working_multi_site_with_PMID.csv          # validation plots (legacy)
|   |-- co_working_multi_site_from_cluster_sheets.csv # Table S3 reported co-regulatory sites (v11 add-on)
|   |-- tf_all_phos_site_for_prediction.csv          # default predict input
|   `-- Regulatory_sites                             # FuncPhos benchmark negatives
|-- fasta/
|   `-- transcription_fasta.fasta                    # training / predict
|-- features/                                        # manual feature tables (feature panels)
|-- TF_esm_embedding/                                # ESM-2 window embeddings (training / predict)
|-- alphafold_tf_pdb/                                # AlphaFold PDBs (training / predict)
|-- TF_family/
|   `-- TF_Information.txt                           # dataset description plots
|-- hpa/                                             # optional HPA inputs (reserved)
|-- model_artifacts/                                 # inference checkpoints
|   `-- run_20260831_183836_ESM Window+Site+PDB_147pos_roc08_v11_xlarge/
|       `-- Functional_Transport/artifacts/           # manuscript Stage 1 (preferred)
|   # legacy: run_20260610_204935_ESM Window+Site+PDB/ (base Zenodo bundle may still ship this)
`-- precomputed/                                     # read-only inputs for plotting / IE pipeline
    |-- 2_1_functional_classifier_results/predictions/
    |   |-- v11_147pos_5_folds_ensemble_predictions.csv   # manuscript (v11 add-on)
    |   `-- esm_window_site_pdb_5_folds_ensemble_predictions.csv  # legacy
    |-- 3_figure3/
    |   |-- funcphos_str_scores.csv
    |   `-- funcphos_seq_scores.csv
    `-- run_*/Functional_Transport/
        `-- metrics_all_folds.csv (+ fixed test tables)
```

## `import_export/data/`

```text
import_export/data/
|-- cluster/
|   `-- ie_train_window21_c70_cluster.csv            # training clusters
|-- dataset_phos_site/
|   `-- TF_positive_phos_site_0608.csv               # 147 Import/Export positives
|-- fasta                                           # copy/symlink of functional/data/fasta
|-- TF_esm_embedding                                # copy/symlink of functional/data/TF_esm_embedding
|-- model_artifacts/                                # fold artifacts + Platt calibrator
|   `-- run_20260904_134712_ie147_R3D_D3_kpls_gauto/
|       `-- Import_vs_Export/                       # manuscript Stage 2 (preferred)
|   # legacy: run_20260612_125646_esm_window_only_supcon_ce_import_pos/
`-- precomputed/                                    # plotting / joint-score inputs
    |-- 1_transport_classifier_results/
    |   |-- d3_kpls_gauto_predictions_platt/        # preferred IE per-fold inputs for joint score
    |   |   `-- tf_all_phos_site_predictions_per_fold.csv
    |   |-- joint_score_v11_147pos_d3_platt/         # manuscript joint scores (v11 add-on)
    |   |   `-- tf_all_phos_site_joint_direction_score.csv (+ stable predicted tables)
    |   |-- tf_phos_site_FuncTransport_Direction_summary_with_PMID_annotation_v11_147pos_d3_platt.csv
    |   |-- esm_window_only_import_pos_predictions/  # legacy
    |   `-- joint_score/                             # legacy joint scores
    `-- run_*/Import_vs_Export/
        |-- metrics_all_runs.csv
        `-- all_fold_test_predictions_platt.csv
```

Shared negatives, Localization-Regulatory Classifier ensemble predictions, and feature tables are read from `../functional/data/`.

## `cptac_analysis/data/`

```text
cptac_analysis/data/
|-- source/                                        # CPTAC / ChIP / regulon bundle (base Zenodo)
|   |-- 1.cpatac/LinkedOmicsKB/
|   |-- 3.idmapping/HUMAN_9606_idmapping.dat
|   |-- 4.chipaltas/1.target_genes/targets_5kb/
|   `-- 5.regulons/CollecTRI_regulons.csv
`-- hotspot_catalogs/                              # v11 add-on
    |-- hotspots_mixed_pred_filter/
    `-- hotspots_mixed_pred_filter_pure_direction/
```

CPTAC hotspot validation reads catalogs from `data/hotspot_catalogs/` (or the matching
`results/.../hotspots_*` trees), v11 joint-score / summary tables from
`import_export/data/precomputed/`, and omics from `data/source/` (see
[cptac_analysis/data/README.md](cptac_analysis/data/README.md)).

## `results/` (runtime outputs)

| Path | Produced by | Zenodo |
|------|-------------|--------|
| `functional/results/run_20260831_183836_ESM Window+Site+PDB_147pos_roc08_v11_xlarge/` | Stage 1 manuscript run | not shipped (artifacts may be mirrored under `data/model_artifacts/`) |
| `import_export/results/run_20260904_134712_ie147_R3D_D3_kpls_gauto/` | Stage 2 manuscript run | not shipped |
| `import_export/results/1_transport_classifier_results/joint_score_v11_147pos_d3_platt/` | Joint direction score | also in v11 add-on under `data/precomputed/` |
| `cptac_analysis/results/hotspot_mixed_pred_filter_pure_regulon_only_twosided_20260929/` | Figure 5 four-arm two-sided | optional `fig5_twosided_results` archive |
| `supplement/` | Supplemental Tables S3–S4 | tracked in Git |

Base Zenodo bundles ship Stage 1–2 `data/` + CPTAC `source/`. Manuscript Figure 5 / Tables S3–S4 additionally need the **v11 hotspot inputs** archive (and optionally the Fig. 5 results archive).

After running `calculate_joint_direction_score.py`, copy refreshed joint-score CSVs into
`import_export/data/precomputed/1_transport_classifier_results/joint_score_v11_147pos_d3_platt/`
if you want downstream plots to use the latest scores without editing script paths.

## Re-populate from original projects

If you have the legacy `phosloc-Func`, `phosloc-ImportExport`, and `phosloc-TF` trees locally:

```bash
bash scripts/populate_data.sh
```

Environment overrides: `REPO`, `FUNC_SRC`, `IE_SRC`, `TF_SRC`, `CPTAC_SRC` (see script header).

This rsyncs large assets, refreshes `precomputed/` and `model_artifacts/` for the two classifier modules, and copies `cptac_analysis/data/source/` when `CPTAC_SRC` is available.
