# Data layout

All inputs required to **train**, **plot**, and **predict** live under each subproject's `data/` directory.
Generated figures and new run outputs still go to `results/` (not shipped with the data bundle by default).

Upload the **code repo** and **data directories** separately if needed.
All processed data bundles are available from one Zenodo record:
[`10.5281/zenodo.21064685`](https://doi.org/10.5281/zenodo.21064685).
The root-level `PhosLoc-Transport_DATA_README.txt` is intended as the README
shipped beside the Zenodo archives. This file is the canonical in-repository
data inventory.

| Upload unit | Path | Approx. size | Download / DOI |
|-------------|------|--------------|----------------|
| Code | repo root (exclude `**/data/` large dirs) | ~55 MB | GitHub repository |
| Processed data bundles | `functional/data/`, `import_export/data/`, `cptac_analysis/data/source/` | See Zenodo record | [Zenodo DOI: 10.5281/zenodo.21064685](https://doi.org/10.5281/zenodo.21064685) |

## `functional/data/`

```text
functional/data/
|-- cluster/
|   `-- func_train_window31_c70_cluster.csv          # training clusters
|-- dataset_phos_site/
|   |-- TF_positive_phos_site_0608.csv               # training / plots
|   |-- TF_deepmvp_negative_phos_site_tf_only.csv    # training / plots
|   |-- co_working_multi_site_with_PMID.csv          # validation plots
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
|-- model_artifacts/                                 # saved checkpoints (predict)
|   `-- run_20260610_204935_ESM Window+Site+PDB/
|       `-- Functional_Transport/artifacts/
`-- precomputed/                                     # read-only inputs for plotting / IE pipeline
    |-- 2_1_functional_classifier_results/predictions/
    |   `-- esm_window_site_pdb_5_folds_ensemble_predictions.csv
    |-- 3_figure3/
    |   |-- funcphos_str_scores.csv
    |   `-- funcphos_seq_scores.csv
    `-- run_20260610_*/Functional_Transport/
        `-- metrics_all_folds.csv (+ fixed test tables for main run)
```

## `import_export/data/`

```text
import_export/data/
|-- cluster/
|   `-- ie_train_window21_c70_cluster.csv            # training clusters
|-- dataset_phos_site/
|   `-- TF_positive_phos_site_0608.csv               # training positives
|-- fasta                                           # copy/symlink of functional/data/fasta
|-- TF_esm_embedding                                # copy/symlink of functional/data/TF_esm_embedding
|-- model_artifacts/                                # fold artifacts + Platt calibrator (predict)
|   `-- run_20260612_125646_esm_window_only_supcon_ce_import_pos/
|       `-- Import_vs_Export/
|           |-- fold_artifacts/
|           |-- platt_calibrator.json
|           |-- platt_calibration_meta.json
|           `-- run_meta.json
`-- precomputed/                                    # plotting / joint-score inputs
    |-- 1_transport_classifier_results/
    |   |-- esm_window_only_import_pos_predictions/
    |   |   `-- tf_all_phos_site_predictions_per_fold.csv
    |   `-- joint_score/
    |       `-- predicted_{import,export}_stable_gt0p6_vote4.csv
    `-- run_20260612_125646_esm_window_only_supcon_ce_import_pos/
        `-- Import_vs_Export/
            |-- metrics_all_runs.csv
            `-- all_fold_test_predictions_platt.csv
```

Shared negatives, Localization-Regulatory Classifier ensemble predictions, and feature tables are read from `../functional/data/`.

## `cptac_analysis/data/`

```text
cptac_analysis/data/
`-- source/                                        # CPTAC / ChIP / regulon bundle (local copy)
    |-- 1.cpatac/LinkedOmicsKB/
    |-- 3.idmapping/HUMAN_9606_idmapping.dat
    |-- 4.chipaltas/1.target_genes/targets_5kb/
    `-- 5.regulons/CollecTRI_regulons.csv
```

CPTAC validation also reads stable nuclear accumulation predictions and known positive labels from `import_export/data/precomputed/` and `functional/data/` (see [cptac_analysis/data/README.md](cptac_analysis/data/README.md)).

## `results/` (runtime outputs, not included in the Zenodo data bundle)

| Path | Produced by |
|------|-------------|
| `functional/results/` | Training, plotting, `predict_functional_transport.py` |
| `import_export/results/` | Training, plotting, `calculate_joint_direction_score.py`, `predict_import_export_direction.py` |
| `cptac_analysis/results/import_target_regulation/` | `run_import_target_regulation_analysis.py` |
| `cptac_analysis/results/phosphosite_across_cancers_boxplots/` | `plot_phosphosite_across_cancers.py` |

After running `calculate_joint_direction_score.py`, copy refreshed joint-score CSVs into `import_export/data/precomputed/.../joint_score/` if you want feature-panel plots to use the latest scores without editing script paths.

## Re-populate from original projects

If you have the legacy `phosloc-Func`, `phosloc-ImportExport`, and `phosloc-TF` trees locally:

```bash
bash scripts/populate_data.sh
```

Environment overrides: `REPO`, `FUNC_SRC`, `IE_SRC`, `TF_SRC`, `CPTAC_SRC` (see script header).

This rsyncs large assets, refreshes `precomputed/` and `model_artifacts/` for the two classifier modules, and copies `cptac_analysis/data/source/` when `CPTAC_SRC` is available.
