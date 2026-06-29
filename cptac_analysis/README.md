# CPTAC cancer analysis

Stage 3 analysis pipeline for the PhosLoc-Transport repository.

This subproject links stable PhosLoc-Transport **import** predictions to CPTAC matched tumor multi-omics data and evaluates whether high phosphorylation of predicted import-associated phosphosites is associated with directionally consistent **signed TF target-gene expression** across cancer types.

## Analysis overview

The integrated pipeline:

- uses matched CPTAC phosphoproteomics, proteomics, and RNA-seq tumor samples;
- stratifies tumors by phosphosite phosphorylation level;
- evaluates curated signed TF target-gene expression for activate and repress regulon groups;
- compares observed effects to random matched controls; and
- optionally adjusts for TF abundance when protein (and, if configured, mRNA) measurements are available.

The reference run stratifies samples by site phosphoproteomics abundance and evaluates target-gene expression after residualizing against TF protein abundance when available (`abundance_primary_mode: residual`).

| Script | Role |
|--------|------|
| [`scripts/run_import_target_regulation_analysis.py`](scripts/run_import_target_regulation_analysis.py) | Integrated target-regulation pipeline (omics matching, high/low phospho splits, random controls, abundance adjustment, summary tables) |
| [`scripts/plot_phosphosite_across_cancers.py`](scripts/plot_phosphosite_across_cancers.py) | Per-phosphosite low/high phospho target-expression boxplots across cancers |
| [`scripts/plot_significant_sites_combined.py`](scripts/plot_significant_sites_combined.py) | Combined boxplot for all BH-significant import activate-target sites |

## Requirements

- Root dependencies: [`requirements.txt`](../requirements.txt)
- Additional packages: `pyensembl` (Ensembl gene annotation; listed in [`requirements-optional.txt`](../requirements-optional.txt))
- Ensure the required Ensembl release cache or annotation files are available locally before running the integrated pipeline

## Data

Large CPTAC and reference files are **not** tracked in Git. Prepare or symlink inputs under `cptac_analysis/data/` before running. Required input categories:

- CPTAC phosphoproteomics, proteomics, and RNA-seq matrices
- stable PhosLoc-Transport import predictions
- curated signed TF target sets
- known positive phosphosite labels
- gene annotation / id-mapping files

See [`data/README.md`](data/README.md) for the expected directory layout.

## Run integrated pipeline

```bash
cd cptac_analysis
export PYTHONPATH="${PWD}/scripts:${PYTHONPATH}"

python scripts/run_import_target_regulation_analysis.py
```

Default outputs: `results/import_target_regulation/`

## Plot per-site boxplots across cancers

```bash
cd cptac_analysis
export PYTHONPATH="${PWD}/scripts:${PYTHONPATH}"

python scripts/plot_phosphosite_across_cancers.py \
  --site-labels STAT3_Y705 STAT3_S701 E2F4_S244 NFATC2_S53 HSF1_S326
```

Default outputs: `results/phosphosite_across_cancers_boxplots/`

## Plot combined significant sites

```bash
cd cptac_analysis
export PYTHONPATH="${PWD}/scripts:${PYTHONPATH}"

python scripts/plot_significant_sites_combined.py
```

Default outputs: `results/import_target_regulation/high_low_phospho_boxplots/all_significant_sites_combined/`

## Reference output directories

| Output | Path |
|--------|------|
| Integrated pipeline run | `results/import_target_regulation/` |
| Per-site across-cancer boxplots | `results/phosphosite_across_cancers_boxplots/` |

## Related documentation

| Resource | Description |
|----------|-------------|
| [data/README.md](data/README.md) | CPTAC and reference data layout |
| [../docs/FIGURES_AND_PREDICTION.md](../docs/FIGURES_AND_PREDICTION.md) | Figure and prediction scripts for Stages 1–2 |
| [../docs/TRAINING_RUNS.md](../docs/TRAINING_RUNS.md) | Finalized model training details for Stages 1–2 |
| [../import_export/data/precomputed/1_transport_classifier_results/joint_score/](../import_export/data/precomputed/1_transport_classifier_results/joint_score/) | Stable import/export predictions used as pipeline input |
| [../functional/data/dataset_phos_site/TF_positive_phos_site_0608.csv](../functional/data/dataset_phos_site/TF_positive_phos_site_0608.csv) | Known positive phosphosite labels |
