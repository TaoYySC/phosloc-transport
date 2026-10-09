# Finalized training and analysis runs

This repository keeps the scripts and configs used to train the two main classifiers (Stages 1–2) and to reproduce the reference CPTAC **phospho-hotspot** analysis (Stage 3).

Manuscript-aligned defaults below use the **147-positive / 868-negative** Functional Transport set and the **D3 kernel-PLS** Import/Export model. Older 124-pos / SupCon+CE-PLS-64 runs are retained only as legacy snapshots.

---

## Run 1 — Functional Transport (ESM Window + Site + PDB, v11)

| Field | Value |
|-------|-------|
| **Original output** | `results/run_20260831_183836_ESM Window+Site+PDB_147pos_roc08_v11_xlarge/Functional_Transport/` |
| **Task** | Functional Transport |
| **Feature set** | `esm_graph` (ESM window-31 + AlphaFold graph) |
| **Model** | `esm_cnn2d_site_gnn` (`train_roc08_v11_xlarge.yaml`) |
| **Samples** | **147** positives, **868** negatives (after `negative_min_distance: 30`) |
| **CV** | Fixed test (20%) + 5-fold StratifiedGroupKFold on development set |
| **Seed** | 42 (fixed test seed 42) |
| **Selection metric** | `val_auroc` |
| **Cluster** | `data/cluster/func_train_window31_c70_cluster.csv` |
| **Best fold** | fold 3, seed 45 (see `configs/runs/esm_window_site_pdb_run_meta.json`) |
| **Run meta** | `configs/runs/esm_window_site_pdb_run_meta.json` (also under the results path above) |

### Config files

```
functional/configs/
├── experiments/esm_window_site_pdb.yaml              # default entry → v11
├── experiments/esm_window_site_pdb_roc08_v11_xlarge.yaml
├── split.yaml
├── train_roc08_v11_xlarge.yaml                       # manuscript HPs
├── train.yaml                                        # legacy narrower HPs
├── feature_sets.yaml
└── runs/
    ├── esm_window_site_pdb_run_meta.json             # 147 / 868 (v11)
    └── esm_window_site_pdb_run_meta_124pos_legacy.json
```

### Train command

```bash
cd functional
export PYTHONPATH="${PWD}:${PYTHONPATH}"

python scripts/1_1_run_experiment.py \
  --experiment_cfg configs/experiments/esm_window_site_pdb.yaml \
  --output_tag "ESM Window+Site+PDB_147pos_roc08_v11_xlarge"
```

Equivalent explicit config: `configs/experiments/esm_window_site_pdb_roc08_v11_xlarge.yaml`.

### Key hyperparameters (`configs/train_roc08_v11_xlarge.yaml`)

| Parameter | Value |
|-----------|-------|
| `proj_input_dim` | 1024 |
| `conv_channels` | [576, 288] |
| `site_hidden_dims` | [1152, 576] |
| `gnn_hidden_dim` | 144 |
| `gnn_heads` | 4 (as in finalized checkpoints) |
| `fusion_hidden_dims` | [224, 112] |
| `dropout` | 0.14 |
| `lr` | 0.00028 |
| `weight_decay` | 0.0012 |
| `batch_size` | 28 |
| `num_epochs` | 160 |
| `early_stopping_metric` | val_auroc |
| `early_stopping_patience` | 25 |

### Data inputs

| Path | Description |
|------|-------------|
| `data/dataset_phos_site/TF_positive_phos_site_0608.csv` | **147** positive sites (manuscript) |
| `data/dataset_phos_site/TF_deepmvp_negative_phos_site_tf_only.csv` | Negative pool (filtered to 868 at train time) |
| `data/fasta/transcription_fasta.fasta` | Sequences |
| `data/cluster/func_train_window31_c70_cluster.csv` | CD-HIT clusters |
| `data/TF_esm_embedding/` | ESM-2 embeddings (symlink / Zenodo) |
| `data/alphafold_tf_pdb/` | AlphaFold PDBs (symlink / Zenodo) |

Large fold checkpoints ship via Zenodo, not Git. Point `data/model_artifacts/` or the results `artifacts/` tree at the downloaded pack when predicting.

---

## Run 2 — Import vs Export (D3 kernel-PLS + SupCon+CE, Import positive)

| Field | Value |
|-------|-------|
| **Original output** | `results/run_20260904_134712_ie147_R3D_D3_kpls_gauto/Import_vs_Export/` |
| **Task** | Import vs Export |
| **Feature set** | ESM window-41 + site, **kernel-PLS** (RBF, auto γ), 96+96 comps |
| **Model** | `supcon_ce` on D3 reduced features (`train_d3_kpls_gauto.yaml`) |
| **Positive class** | Import (LABEL=1); **147** labeled Import/Export sites |
| **CV** | 5-fold StratifiedGroupKFold, seed 42 |
| **Cluster** | `data/cluster/ie_train_window21_c70_cluster.csv` |
| **Run meta** | `configs/runs/ie147_R3D_D3_kpls_gauto_run_meta.json` |
| **Downstream joint scores** | `import_export/data/precomputed/1_transport_classifier_results/joint_score_v11_147pos_d3_platt/` |

### Config files

```
import_export/configs/
├── experiments/import_export_ie147_D3_kpls_gauto.yaml   # default (manuscript)
├── feature_sets_d3_kpls_gauto.yaml
├── train_d3_kpls_gauto.yaml
├── sweeps/ie147_round3D_20260904_134433/               # sweep originals
│   ├── exp_D3_kpls_gauto.yaml
│   ├── feature_D3_kpls_gauto.yaml
│   └── train_D3_kpls_gauto.yaml
├── split.yaml
├── runs/ie147_R3D_D3_kpls_gauto_run_meta.json
└── (legacy) experiments/import_export_esm_window_only_supcon_ce_import_pos.yaml
```

### Train command

```bash
cd import_export
export PYTHONPATH="${PWD}:${PYTHONPATH}"

python scripts/run_import_export_experiment.py \
  --experiment_cfg configs/experiments/import_export_ie147_D3_kpls_gauto.yaml \
  --output_tag ie147_R3D_D3_kpls_gauto
```

### Key hyperparameters (D3)

**Features** (`feature_sets_d3_kpls_gauto.yaml`):

| Parameter | Value |
|-----------|-------|
| `window_size` | 41 |
| `use_window_embedding` / `use_site_embedding` | true / true |
| `reducer` | `kernel_pls` (RBF) |
| `pls_components_window` / `site` | 96 / 96 |

**Model** (`train_d3_kpls_gauto.yaml`):

| Parameter | Value |
|-----------|-------|
| `C` | 0.1 |
| `alpha` | 10.0 |
| `temperature` | 0.2 |
| `embed_dim` | 32 |
| `lr` | 0.05 |
| `max_iter` | 5000 |
| `class_weight` | balanced |

### Legacy Stage 2 (not manuscript primary)

Earlier public default: ESM window-21 + linear PLS-64 + SupCon+CE  
(`import_export_esm_window_only_supcon_ce_import_pos.yaml`, run `20260612_125646`). Keep for sensitivity only; do not use for Figure 5 / joint_score_v11 tables.

---

## Output layout (both runs)

```
results/run_<timestamp>_<output_tag>/
├── Functional_Transport/          # functional run
│   ├── metrics_all_folds.csv
│   ├── run_meta.json
│   └── artifacts/fold_<N>/esm_graph/
└── Import_vs_Export/              # import_export run
    ├── metrics_all_runs.csv
    ├── run_meta.json
    └── fold_artifacts/...
```

---

## Prerequisites

1. Install dependencies: `pip install -r requirements.txt`
2. Optional: uncomment and install `pyensembl` in [`requirements.txt`](../requirements.txt) (Stage 3)
3. Symlink large data dirs (see `functional/data/README.md`, `cptac_analysis/data/README.md`, [`DATA.md`](../DATA.md))
4. Cluster CSVs included under each subproject's `data/cluster/`; ESM embeddings and PDB must be prepared locally or from Zenodo

---

## Run 3 — CPTAC phospho-hotspot target-regulation analysis (reference)

Manuscript-aligned Stage 3 analysis is **hotspot-level** via **`build_hotspots_mixed_pred_filter.py`** (not `build_phospho_hotspots.py`).

| Field | Value |
|-------|-------|
| **Fig. 5 reference output** | `cptac_analysis/results/hotspot_mixed_pred_filter_pure_regulon_only_twosided_20260929/` |
| **Catalog script** | `cptac_analysis/scripts/build_hotspots_mixed_pred_filter.py` |
| **Direction gate** | `reannotate_hotspot_pure_direction.py` (pure nuclear / pure cytoplasmic) |
| **Scan script** | `cptac_analysis/scripts/run_hotspot_target_regulation_analysis.py` |
| **Catalog** | Known∪Predicted; adj≤15 aa; **span≤40 aa**; Mixed / Predicted-candidate; **99** hotspots (85 TFs) |
| **Activity** | Per-cancer **mean z-score** (`mean_z`) |
| **Phospho split** | `median_nonmissing` |
| **Fig. 5 stats** | `signed_target_mode=regulon_only`; `test_alternative=two-sided`; BH within cancer among evaluable (`n ≥ 10`) |
| **Four-arm runner** | `cptac_analysis/scripts/run_four_arm_twosided_pipeline.sh` |
| **Fig. 5b concordance** | Import×act 17/20; Import×rep 0/1; Export×act 16/45; Export×rep 8/13 |

Spatial / direction definition panels (Fig. 4A–C) and earlier Import-focused association drafts live under separate result trees; see [cptac_analysis/README.md](../cptac_analysis/README.md).

Legacy unit-site reference (not manuscript primary): `results/import_target_regulation/` via `run_import_target_regulation_analysis.py`.
