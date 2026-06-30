# CPTAC analysis - data

Large CPTAC and reference files are **not** tracked in Git. Copy them under `cptac_analysis/data/` before running CPTAC validation.

The processed data bundles are available from the shared Zenodo record:
[`10.5281/zenodo.21021066`](https://doi.org/10.5281/zenodo.21021066).

| Bundle | Target path | Download / DOI |
|--------|-------------|----------------|
| CPTAC / ChIP / regulon source bundle | `cptac_analysis/data/source/` | [Zenodo DOI: 10.5281/zenodo.21021066](https://doi.org/10.5281/zenodo.21021066) |
| Functional data bundle | `functional/data/` | [Zenodo DOI: 10.5281/zenodo.21021066](https://doi.org/10.5281/zenodo.21021066) |
| Localization Direction Classifier data bundle | `import_export/data/` | [Zenodo DOI: 10.5281/zenodo.21021066](https://doi.org/10.5281/zenodo.21021066) |

## Required setup

1. Populate `cptac_analysis/data/source/` with the CPTAC / ChIP / regulon bundle (~3.9 GB).
   If you have the legacy `phosloc-TF` tree locally, run from the repo root:

   ```bash
   bash scripts/populate_data.sh
   ```

   Or copy manually:

   ```bash
   rsync -a /path/to/cptac_source/ cptac_analysis/data/source/
   ```

2. Ensure classifier inputs are present under `functional/data/` and `import_export/data/precomputed/` (stable nuclear accumulation predictions and known positive site labels).

3. Install `pyensembl` and download the Ensembl release cache before running the integrated pipeline (see [cptac_analysis/README.md](../README.md)).

Without `data/source/` in place, the integrated pipeline cannot access CPTAC phosphoproteomics, proteomics, or RNA-seq matrices.

## Directory summary

| Path | Purpose |
|------|---------|
| `source/` | CPTAC omics, ChIP-Atlas targets, CollecTRI regulons, and UniProt idmapping |

## Expected `source/` layout

```text
data/source/
|-- 1.cpatac/LinkedOmicsKB/          # CPTAC phospho / RNA / protein matrices
|-- 2.dataset/                       # auxiliary tables (if used)
|-- 3.idmapping/HUMAN_9606_idmapping.dat
|-- 4.chipaltas/1.target_genes/targets_5kb/
`-- 5.regulons/CollecTRI_regulons.csv
```

## Upstream model outputs (repo-relative)

The integrated pipeline also reads PhosLoc-Transport classifier outputs from the monorepo:

| Path | Purpose |
|------|---------|
| `../../import_export/data/precomputed/1_transport_classifier_results/joint_score/` | Stable Localization Direction Classifier site predictions |
| `../../functional/data/dataset_phos_site/TF_positive_phos_site_0608.csv` | Known positive phosphosite labels |

If `source/` is missing, pass `--linkedomics-base`, `--chip-dir`, and related CLI flags in `run_import_target_regulation_analysis.py` to your local copies.

Full inventory: **[../../DATA.md](../../DATA.md)**.
