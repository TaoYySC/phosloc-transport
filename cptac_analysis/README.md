# CPTAC cancer analysis

Stage 3 analysis pipeline for the PhosLoc-Transport repository.

This subproject links PhosLoc-Transport direction predictions to CPTAC matched tumor
multi-omics and tests whether elevated **phospho-hotspot activity** associates with
directionally consistent **signed TF target-gene expression** across cancer types.

**Manuscript-aligned primary analysis is hotspot-level** (`mixed_pred_filter` catalog;
pure nuclear / pure cytoplasmic gates; `mean_z` activity; `median_nonmissing` High/Low
split). Figure 5 reports a **four-arm, two-sided** scan (Import/Export × activate/repress)
with BH correction within cancer among evaluable associations. Per-phosphosite
(unit-site) runs are retained only as legacy / sensitivity outputs.

## Analysis overview

The primary pipeline:

- builds Mixed / Predicted-candidate phospho-hotspots on each TF protein
  (Known ∪ Predicted sites; adjacent gap ≤15 aa; span ≤40 aa);
- optionally re-annotates Import/Export gates as **pure nuclear / pure cytoplasmic**
  (has nuclear members and no cytoplasmic members, or the reverse);
- uses matched CPTAC phosphoproteomics, proteomics, and RNA-seq tumor samples;
- defines hotspot activity as the per-cancer **mean z-score** of measured member sites;
- stratifies tumors by median hotspot activity among detected samples
  (`median_nonmissing`);
- evaluates curated signed TF target-gene expression (activate / repress);
- compares observed effects to random matched controls; and
- optionally runs hotspot-level Cox survival on BH-significant cancer×hotspot pairs.

| Script | Role |
|--------|------|
| [`scripts/build_hotspots_mixed_pred_filter.py`](scripts/build_hotspots_mixed_pred_filter.py) | **Primary** hotspot catalog (adj≤15, span≤40; Mixed / Predicted candidate) |
| [`scripts/reannotate_hotspot_pure_direction.py`](scripts/reannotate_hotspot_pure_direction.py) | Pure nuclear/cytoplasmic Import/Export gates on an existing catalog |
| [`scripts/run_hotspot_target_regulation_analysis.py`](scripts/run_hotspot_target_regulation_analysis.py) | Hotspot CPTAC scan (mean-z activity → High/Low target association) |
| [`scripts/analyze_two_sided_concordance.py`](scripts/analyze_two_sided_concordance.py) | Figure **5** concordance vs expected direction; Table S4 flags |
| [`scripts/plot_two_sided_four_arm_significance_heatmap.py`](scripts/plot_two_sided_four_arm_significance_heatmap.py) | Figure **5** four-arm hotspot × cancer heatmap |
| [`scripts/plot_hotspot_figure4.py`](scripts/plot_hotspot_figure4.py) | Figure **4b–4f** from hotspot CPTAC results |
| [`scripts/run_figure4_spatial_direction_panels.py`](scripts/run_figure4_spatial_direction_panels.py) | Figure **4A–C** / S5 spatial & direction panels (not CPTAC stratification) |
| [`scripts/run_hotspot_survival_cox.py`](scripts/run_hotspot_survival_cox.py) | Hotspot activity Cox / KM on BH-significant pairs |
| [`scripts/build_phospho_hotspots.py`](scripts/build_phospho_hotspots.py) | Earlier distance catalogs (d=10/15/30); not the manuscript primary catalog |
| [`scripts/run_import_target_regulation_analysis.py`](scripts/run_import_target_regulation_analysis.py) | **Legacy** unit-site CPTAC pipeline |
| [`scripts/plot_phosphosite_across_cancers.py`](scripts/plot_phosphosite_across_cancers.py) | Legacy per-site across-cancer boxplots |
| [`scripts/plot_significant_sites_combined.py`](scripts/plot_significant_sites_combined.py) | Legacy combined significant-site boxplots |

## Requirements

- Root dependencies: [`requirements.txt`](../requirements.txt)
- Additional packages: `pyensembl` (Ensembl gene annotation; commented in [`requirements.txt`](../requirements.txt))
- Ensure the required Ensembl release cache or annotation files are available locally before running the CPTAC scan

## Data

Large CPTAC and reference files are **not** tracked in Git. Prepare or symlink inputs under `cptac_analysis/data/` before running. Required input categories:

- CPTAC phosphoproteomics, proteomics, and RNA-seq matrices
- FuncTransport + Direction summary / stable PhosLoc-Transport predictions
- curated signed TF target sets
- known positive phosphosite labels
- gene annotation / id-mapping files

See [`data/README.md`](data/README.md) for the expected directory layout.

## Primary run (hotspot; manuscript-aligned)

CLI entry points live under `scripts/` (thin wrappers); implementations are in `src/`.

```bash
cd cptac_analysis

# 1) Mixed / Predicted-candidate catalog (99 hotspots / 85 TFs for v11)
python scripts/build_hotspots_mixed_pred_filter.py

# 2) Pure-direction Import/Export gates
python scripts/reannotate_hotspot_pure_direction.py

# 3) Four-arm CPTAC scan (regulon_only; two-sided; mean_z + median_nonmissing)
#    Use the project runner / arm configs that write to the twosided_20260929 tree,
#    or call run_hotspot_target_regulation_analysis.py per arm with --test-alternative two-sided.
# 4) Concordance + Figure 5 heatmap + Table S4 flags
python scripts/analyze_two_sided_concordance.py
python scripts/plot_two_sided_four_arm_significance_heatmap.py

# 5) Optional: Figure 4 spatial / association panels (earlier Import-focused tree retained)
python scripts/run_figure4_spatial_direction_panels.py
python scripts/plot_hotspot_figure4.py \
  --results-dir results/hotspot_mixed_pred_filter_pure_mean_z_median_20260914/cptac \
  --output-dir results/hotspot_mixed_pred_filter_pure_mean_z_median_20260914/figure4
```

Catalog definition (v11_147pos_d3_platt):

- Sites clustered: **Known** (PMID present) ∪ **Predicted** (ordinary Observed excluded)
- Break cluster if adjacent gap **>15 aa** or span **>40 aa**
- Keep **Mixed evidence** (≥1 Known + ≥1 Predicted) or **Predicted candidate**
  (no Known, ≥3 Predicted, mean FuncTransport ≥0.6)
- Fixed size: **99** hotspots (48 Mixed / 51 Predicted) on **85** TFs

Figure 5b concordance among BH-significant evaluable associations (two-sided; within-cancer BH):
Import×activate **17/20**, Import×repress **0/1**, Export×activate **16/45**, Export×repress **8/13**.
Supplemental Table S4 lists all evaluable associations (significant and non-significant).

## Reference output directories

| Output | Path |
|--------|------|
| Primary hotspot catalog | `results/import_target_regulation_hotspot_v11_147pos_d3_platt/hotspots_mixed_pred_filter/` |
| Pure-direction catalog | `.../hotspots_mixed_pred_filter_pure_direction/` |
| **Fig. 5 four-arm two-sided (primary)** | `results/hotspot_mixed_pred_filter_pure_regulon_only_twosided_20260929/` |
| Earlier Import-focused CPTAC + Fig 4b–4f | `results/hotspot_mixed_pred_filter_pure_mean_z_median_20260914/` |
| Non-pure sensitivity (full) | `results/hotspot_mixed_pred_filter_mean_z_median_full_20260914/` |
| Spatial Fig 4A–C / S5 | `results/figure4_spatial_direction_panels_v11_147pos_d3_platt/` |
| Hotspot Cox | `results/survival_analysis/hotspot_cox_*` |
| Legacy unit-site pipeline | `results/import_target_regulation/` / `results/import_target_regulation_v11_147pos_d3_platt/` |
| Tables S3–S4 (repo) | [`../supplement/`](../supplement/) |

Summaries: [`results/hotspot_mixed_pred_filter_mean_z_median_full_20260914/RESULTS_SUMMARY.md`](results/hotspot_mixed_pred_filter_mean_z_median_full_20260914/RESULTS_SUMMARY.md),
[`results/import_target_regulation_hotspot_v11_147pos_d3_platt_mixed_pred_filter/PIPELINE_SUMMARY.md`](results/import_target_regulation_hotspot_v11_147pos_d3_platt_mixed_pred_filter/PIPELINE_SUMMARY.md).

## Legacy unit-site analysis

Unit-site stratification (per predicted import phosphosite abundance) is **not** the
manuscript primary analysis. To re-run:

```bash
cd cptac_analysis

python scripts/run_import_target_regulation_analysis.py
python scripts/plot_phosphosite_across_cancers.py \
  --site-labels STAT3_Y705 STAT3_S701 E2F4_S244 NFATC2_S53 HSF1_S326
python scripts/plot_significant_sites_combined.py
```

Default legacy outputs: `results/import_target_regulation/`.

## Statistical notes (Figure 5 / Table S4)

- Test: two-sided Wilcoxon / Mann–Whitney on High vs Low **hotspot activity**
- Evaluable: ≥10 paired target genes; BH **within cancer** among evaluable tests in that arm
- Table S4 columns include Hotspot direction, Target regulation, Test alternative,
  BH significant, and Matches expected direction
- Table S3 (reported co-regulatory site enrichment) is produced by
  `functional/scripts/plot_functional_validation_scores.py`; figure stars use
  **unadjusted permutation P** (`* P < 0.05`), with BH q reported in the table for transparency

## Related documentation

| Resource | Description |
|----------|-------------|
| [data/README.md](data/README.md) | CPTAC and reference data layout |
| [../docs/FIGURES_AND_PREDICTION.md](../docs/FIGURES_AND_PREDICTION.md) | Figure and prediction scripts (incl. Stage 3 hotspot panels) |
| [../docs/TRAINING_RUNS.md](../docs/TRAINING_RUNS.md) | Finalized model training + Stage 3 reference run |
| [../supplement/](../supplement/) | Supplemental Tables S3–S4 |
| [../import_export/results/1_transport_classifier_results/joint_score_v11_147pos_d3_platt/](../import_export/results/1_transport_classifier_results/joint_score_v11_147pos_d3_platt/) | Stable import/export predictions (D3+Platt) |
| [../functional/data/dataset_phos_site/TF_positive_phos_site_0608.csv](../functional/data/dataset_phos_site/TF_positive_phos_site_0608.csv) | Known positive phosphosite labels |
