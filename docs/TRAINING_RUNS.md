# Finalized training and analysis runs

This repository keeps the scripts and configs used to train the two main classifiers (Stages 1–2) and to reproduce the reference CPTAC **phospho-hotspot** analysis (Stage 3).

---

## Run 1 — Functional Transport (ESM Window + Site + PDB)

| Field | Value |
|-------|-------|
| **Original output** | `results/run_20260610_204935_ESM Window+Site+PDB/Functional_Transport/` |
| **Task** | Functional Transport |
| **Feature set** | `esm_graph` (ESM window-31 + AlphaFold graph) |
| **Model** | `esm_cnn2d_site_gnn` |
| **CV** | Fixed test (20%) + 5-fold StratifiedGroupKFold on development set |
| **Seed** | 42 (fixed test seed 42) |
| **Selection metric** | `val_auroc` |
| **Cluster** | `data/cluster/func_train_window31_c70_cluster.csv` |
| **Best fold** | fold 3, seed 45 (see `configs/runs/esm_window_site_pdb_run_meta.json`) |

### Config files

```
functional/configs/
├── experiments/esm_window_site_pdb.yaml   # experiment entry
├── split.yaml                             # CV / fixed-test split
├── train.yaml                             # esm_cnn2d_site_gnn hyperparameters
├── feature_sets.yaml                      # esm_graph feature blocks
└── runs/esm_window_site_pdb_run_meta.json # snapshot from original run
```

### Train command

```bash
cd functional
export PYTHONPATH="${PWD}:${PYTHONPATH}"

python scripts/1_1_run_experiment.py \
  --experiment_cfg configs/experiments/esm_window_site_pdb.yaml \
  --output_tag "ESM Window+Site+PDB"
```

### Key hyperparameters (`configs/train.yaml`)

| Parameter | Value |
|-----------|-------|
| `proj_input_dim` | 512 |
| `conv_channels` | [256, 128] |
| `site_hidden_dims` | [512, 256] |
| `gnn_hidden_dim` | 64 |
| `dropout` | 0.05 |
| `lr` | 0.0002 |
| `weight_decay` | 0.001 |
| `batch_size` | 32 |
| `num_epochs` | 80 |
| `early_stopping_metric` | val_auprc |
| `early_stopping_patience` | 10 |

### Data inputs (`configs/experiments/esm_window_site_pdb.yaml`)

| Path | Description |
|------|-------------|
| `data/dataset_phos_site/TF_positive_phos_site_0608.csv` | Positive sites |
| `data/dataset_phos_site/TF_deepmvp_negative_phos_site_tf_only.csv` | Negatives |
| `data/fasta/transcription_fasta.fasta` | Sequences |
| `data/cluster/func_train_window31_c70_cluster.csv` | CD-HIT clusters |
| `data/TF_esm_embedding/` | ESM-2 embeddings (symlink) |
| `data/alphafold_tf_pdb/` | AlphaFold PDBs (symlink) |

---

## Run 2 — Import vs Export (ESM Window + SupCon+CE, Import positive)

| Field | Value |
|-------|-------|
| **Original output** | `results/run_20260612_125646_esm_window_only_supcon_ce_import_pos/Import_vs_Export/` |
| **Task** | Import vs Export |
| **Feature set** | `esm_window_only_supcon_ce` |
| **Model** | `supcon_ce` (PLS-64 + SupCon+CE) |
| **Positive class** | Import (LABEL=1) |
| **CV** | 5-fold StratifiedGroupKFold, seed 42 |
| **Window size** | 21 |
| **Cluster** | `data/cluster/ie_train_window21_c70_cluster.csv` |
| **Test AUROC** | 0.765 ± 0.135 (see run meta) |

### Config files

```
import_export/configs/
├── experiments/import_export_esm_window_only_supcon_ce_import_pos.yaml
├── split.yaml
├── train_supcon_ce_window_only.yaml
├── feature_sets_esm_window_only_supcon_ce.yaml
└── runs/esm_window_only_supcon_ce_import_pos_run_meta.json
```

### Train command

```bash
cd import_export
export PYTHONPATH="${PWD}:${PYTHONPATH}"

python scripts/run_import_export_experiment.py \
  --experiment_cfg configs/experiments/import_export_esm_window_only_supcon_ce_import_pos.yaml \
  --output_tag esm_window_only_supcon_ce_import_pos
```

### Key hyperparameters

**Split** (`configs/split.yaml`):

| Parameter | Value |
|-----------|-------|
| `seeds` | [42] |
| `n_splits` | 5 |
| `stratify_group_kfold` | true |
| `group_col` | Cluster_ID |

**Model** (`configs/train_supcon_ce_window_only.yaml`):

| Parameter | Value |
|-----------|-------|
| `C` | 0.03 |
| `alpha` | 1.0 |
| `temperature` | 0.05 |
| `embed_dim` | 64 |
| `lr` | 0.05 |
| `max_iter` | 5000 |
| `class_weight` | balanced |

**Features** (`configs/feature_sets_esm_window_only_supcon_ce.yaml`):

| Parameter | Value |
|-----------|-------|
| `window_size` | 21 |
| `reducer` | pls |
| `pls_components_window` | 64 |
| `use_window_embedding` | true |
| `use_site_embedding` | false |

### Data inputs

| Path | Description |
|------|-------------|
| `data/dataset_phos_site/TF_positive_phos_site_0608.csv` | Import + Export positives |
| `data/fasta/transcription_fasta.fasta` | Sequences |
| `data/cluster/ie_train_window21_c70_cluster.csv` | IE training clusters |
| `data/TF_esm_embedding/` | ESM-2 embeddings (symlink) |

---

## Output layout (both runs)

```
results/run_<timestamp>_<output_tag>/
├── Functional_Transport/          # functional run
│   ├── metrics_all_folds.csv
│   ├── run_meta.json
│   └── artifacts/fold_<N>/esm_graph/
│       ├── model.pt
│       └── ...
└── Import_vs_Export/              # import_export run
    ├── metrics_all_runs.csv
    ├── run_meta.json
    └── fold_artifacts/feature_set=esm_window_only_supcon_ce/seed=42_fold=<N>/
        ├── model.joblib
        └── pipeline.joblib
```

---

## Prerequisites

1. Install dependencies: `pip install -r requirements.txt`
2. Optional: uncomment and install `pyensembl` in [`requirements.txt`](../requirements.txt) (Stage 3)
3. Symlink large data dirs (see `functional/data/README.md`, `cptac_analysis/data/README.md`)
4. Cluster CSVs included under each subproject's `data/cluster/` (one file per run); ESM embeddings and PDB must be prepared locally

---

## Run 3 — CPTAC phospho-hotspot target-regulation analysis (reference)

Manuscript-aligned Stage 3 analysis is **hotspot-level** (not per-phosphosite).

| Field | Value |
|-------|-------|
| **Fig. 5 reference output** | `cptac_analysis/results/hotspot_mixed_pred_filter_pure_regulon_only_twosided_20260929/` |
| **Fig. 4 reference output** | `cptac_analysis/results/hotspot_mixed_pred_filter_pure_mean_z_median_20260914/` |
| **Task** | Phospho-hotspot activity vs signed TF target-gene expression (Import/Export × activate/repress) |
| **Catalog script** | `cptac_analysis/scripts/build_hotspots_mixed_pred_filter.py` |
| **Direction gate** | `reannotate_hotspot_pure_direction.py` (pure nuclear / pure cytoplasmic) |
| **Scan script** | `cptac_analysis/scripts/run_hotspot_target_regulation_analysis.py` |
| **Catalog** | Known∪Predicted; adj≤15 aa; span≤40 aa; 99 Mixed/Predicted hotspots (85 TFs) |
| **Activity** | Per-cancer **mean z-score** of measured hotspot members (`mean_z`) |
| **Phospho split** | `median_nonmissing` on hotspot activity |
| **Fig. 5 stats** | Two-sided tests; BH within cancer among evaluable (`n ≥ 10`); regulon_only |
| **Figure panels** | Fig. 5 heatmap (`plot_two_sided_four_arm_significance_heatmap.py`); Fig. 4b–4f / 4A–C as above |
| **Fig. 5b concordance** | Import×act 17/20; Import×rep 0/1; Export×act 16/45; Export×rep 8/13 |

Legacy unit-site reference (not manuscript primary): `results/import_target_regulation/` via `run_import_target_regulation_analysis.py`.

See **[cptac_analysis/README.md](../cptac_analysis/README.md)** for full commands and output paths.
