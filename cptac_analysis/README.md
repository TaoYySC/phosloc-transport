# CPTAC validation

CPTAC validation module for the PhosLoc-Transport repository.

This subproject links stable PhosLoc-Transport nuclear accumulation predictions to CPTAC matched tumor multi-omics data and evaluates whether high phosphorylation of predicted nuclear accumulation-associated phosphosites is associated with directionally consistent **signed TF target-gene expression** across cancer types.

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

## Statistical design

The default analysis uses:

| Component | Default behavior |
|-----------|------------------|
| Site set | Stable predicted nuclear accumulation sites from direction-classifier joint-score outputs |
| Phospho split | `median_nonmissing` split on site phosphoproteomics abundance |
| Target sets | Signed TF target genes from curated regulons, optionally intersected with ChIP-derived target support |
| Directional test | High-phospho versus low-phospho target expression in the expected signed direction |
| Random control | Matched random controls with `--random-iterations 100` and `--random-seed 42` |
| TF abundance adjustment | Residualized target expression/activity when TF protein abundance is available |
| Multiple testing | BH correction is enabled by default (`--use-bh-pvalue-correction`) |

## Requirements

- Root dependencies: [`requirements.txt`](../requirements.txt)
- Additional packages: `pyensembl` (Ensembl gene annotation; listed in [`requirements-optional.txt`](../requirements-optional.txt))
- Ensure the required Ensembl release cache or annotation files are available locally before running the integrated pipeline

Example cache preparation:

```bash
python -c "from pyensembl import EnsemblRelease; EnsemblRelease(100).download(); EnsemblRelease(100).index()"
```

Use the release configured by `--ensembl-release` if you change the default.

## Data

Large CPTAC and reference files are **not** tracked in Git. Prepare or symlink inputs under `cptac_analysis/data/` before running. Required input categories:

- CPTAC phosphoproteomics, proteomics, and RNA-seq matrices
- stable PhosLoc-Transport import predictions
- curated signed TF target sets
- known positive phosphosite labels
- gene annotation / id-mapping files

See [`data/README.md`](data/README.md) for the expected directory layout.

The processed data bundles are available from the shared Zenodo record:
[`10.5281/zenodo.21021066`](https://doi.org/10.5281/zenodo.21021066).

| Input bundle | Target path | Download / DOI |
|--------------|-------------|----------------|
| Functional data bundle | `../functional/data/` | [Zenodo DOI: 10.5281/zenodo.21021066](https://doi.org/10.5281/zenodo.21021066) |
| Direction-classifier data bundle | `../import_export/data/` | [Zenodo DOI: 10.5281/zenodo.21021066](https://doi.org/10.5281/zenodo.21021066) |
| CPTAC / ChIP / regulon source bundle | `data/source/` | [Zenodo DOI: 10.5281/zenodo.21021066](https://doi.org/10.5281/zenodo.21021066) |

## Run integrated pipeline

```bash
cd cptac_analysis
export PYTHONPATH="${PWD}/scripts:${PYTHONPATH}"

python scripts/run_import_target_regulation_analysis.py
```

Default outputs: `results/import_target_regulation/`

Run a subset of cancer types:

```bash
python scripts/run_import_target_regulation_analysis.py \
  --cancer-types BRCA LUAD OV \
  --output-dir results/import_target_regulation_subset
```

Use explicit local input paths when `data/source/` is not populated:

```bash
python scripts/run_import_target_regulation_analysis.py \
  --linkedomics-base /path/to/LinkedOmicsKB \
  --chip-dir /path/to/targets_5kb \
  --signed-regulon-path /path/to/CollecTRI_regulons.csv \
  --idmapping-path /path/to/HUMAN_9606_idmapping.dat
```

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

## Main output tables

| File or directory | Description |
|-------------------|-------------|
| `run_config.json` | Full resolved pipeline configuration |
| `all_target_gene_mean_expression_points.csv` | Primary per-site/per-target expression table used for downstream summaries and plots |
| `all_target_gene_mean_expression_points_all_modes.csv` | All expression modes when abundance adjustment modes are enabled |
| `site_level_processing_summary.csv` | Site-level processing status, sample counts, target counts, and abundance/robustness columns |
| `high_low_phospho_boxplots/` | High/low phospho expression points, summary tables, statistics, and plots |
| `merged_all_sites_boxplots/` | Merged across-site high/low comparisons |
| `target_logfc_activity_random_analysis/` | Observed target logFC, signed TF activity scores, random-control summaries, and abundance sensitivity outputs |
| `baseline_unadjusted/` | Baseline outputs when `--abundance-primary-mode dual` is used |

Common statistics columns include raw p-values, BH-adjusted q-values (`*_q_bh`), `significance_bh`, and `is_significant` indicators.

## Upstream data sources

The analysis expects local copies of third-party resources. Record exact versions, download dates, and provider URLs before final release.

| Resource | Use |
|----------|-----|
| CPTAC / LinkedOmicsKB matrices | Tumor phosphoproteomics, proteomics, and RNA-seq |
| ChIP-Atlas target tables | TF target support / filtering |
| CollecTRI regulons | Signed activate/repress TF-target relationships |
| UniProt idmapping | Gene/protein identifier mapping |
| Ensembl / pyensembl | Gene annotation |
| PhosLoc-Transport classifier outputs | Stable predicted import sites and known positive labels |

## Related documentation

| Resource | Description |
|----------|-------------|
| [data/README.md](data/README.md) | CPTAC and reference data layout |
| [../docs/FIGURES_AND_PREDICTION.md](../docs/FIGURES_AND_PREDICTION.md) | Figure, prediction, and CPTAC validation scripts |
| [../docs/TRAINING_RUNS.md](../docs/TRAINING_RUNS.md) | Finalized classifier training and CPTAC validation details |
| [../import_export/data/precomputed/1_transport_classifier_results/joint_score/](../import_export/data/precomputed/1_transport_classifier_results/joint_score/) | Stable direction predictions used as pipeline input |
| [../functional/data/dataset_phos_site/TF_positive_phos_site_0608.csv](../functional/data/dataset_phos_site/TF_positive_phos_site_0608.csv) | Known positive phosphosite labels |
