# CPTAC analysis — data

Large CPTAC and reference files are **not** tracked in Git. Copy them under `cptac_analysis/data/` before running Stage 3.

Default commands use the manuscript v11 Stage 1 and D3 Stage 2 runs. Earlier runs are retained as legacy versions.

## Required setup

1. Populate `cptac_analysis/data/source/` with the CPTAC / ChIP / regulon bundle (base Zenodo archive).  
   If you have the legacy `phosloc-TF` tree locally, run from the repo root:

   ```bash
   bash scripts/populate_data.sh
   ```

   Or copy manually:

   ```bash
   rsync -a /path/to/cptac_source/ cptac_analysis/data/source/
   ```

2. Extract the **v11 hotspot inputs** add-on into the repository root so that
   `cptac_analysis/data/hotspot_catalogs/`,
   `import_export/data/precomputed/.../joint_score_v11_147pos_d3_platt/`, and
   `functional/data/precomputed/.../v11_147pos_5_folds_ensemble_predictions.csv`
   are present (see [`../../DATA.md`](../../DATA.md)).

3. Install `pyensembl` and download the Ensembl release cache before running the Stage 3 hotspot CPTAC scan (see [cptac_analysis/README.md](../README.md)).

Without `data/source/` in place, the hotspot (or legacy unit-site) pipeline cannot access CPTAC phosphoproteomics, proteomics, or RNA-seq matrices.

## Directory summary

| Path | Purpose |
|------|---------|
| `source/` | CPTAC omics, ChIP-Atlas targets, CollecTRI regulons, and UniProt idmapping |
| `hotspot_catalogs/` | Manuscript Mixed/Predicted catalogs (v11 add-on), including pure-direction gates |

## Expected `source/` layout

```
data/source/
├── 1.cpatac/LinkedOmicsKB/          # CPTAC phospho / RNA / protein matrices
├── 2.dataset/                       # auxiliary tables (if used)
├── 3.idmapping/HUMAN_9606_idmapping.dat
├── 4.chipaltas/1.target_genes/targets_5kb/
└── 5.regulons/CollecTRI_regulons.csv
```

## Upstream model outputs (repo-relative)

The Stage 3 hotspot pipeline also reads PhosLoc-Transport classifier outputs from the monorepo:

| Path | Purpose |
|------|---------|
| `../../import_export/data/precomputed/1_transport_classifier_results/tf_phos_site_FuncTransport_Direction_summary_with_PMID_annotation_v11_147pos_d3_platt.csv` | Summary table for Mixed/Predicted hotspot clustering |
| `../../import_export/data/precomputed/1_transport_classifier_results/joint_score_v11_147pos_d3_platt/` | Stable import/export site predictions (D3+Platt) |
| `../../functional/data/dataset_phos_site/TF_positive_phos_site_0608.csv` | Known positive phosphosite labels (147 sites) |
| `data/hotspot_catalogs/hotspots_mixed_pred_filter(_pure_direction)/` | Prebuilt 99-hotspot catalogs |

If `source/` is missing, pass `--linkedomics-base`, `--chip-dir`, and related CLI flags in `run_hotspot_target_regulation_analysis.py` (or the legacy `run_import_target_regulation_analysis.py`) to your local copies.

Full inventory: **[../../DATA.md](../../DATA.md)**
