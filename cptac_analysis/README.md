# CPTAC cancer analysis

Stage 3 analysis pipeline for the PhosLoc-Transport repository.

This subproject links PhosLoc-Transport direction predictions to CPTAC matched tumor
multi-omics and tests whether elevated **phospho-hotspot activity** associates with
directionally consistent **signed TF target-gene expression** across cancer types.

**Manuscript Methods / primary analysis is hotspot-level** via
[`scripts/build_hotspots_mixed_pred_filter.py`](scripts/build_hotspots_mixed_pred_filter.py)
(not `build_phospho_hotspots.py`). Catalog: Known ∪ Predicted; adjacent gap ≤15 aa;
**span ≤40 aa**; keep **Mixed evidence** or **Predicted candidate**; then pure nuclear /
pure cytoplasmic direction gates; `mean_z` activity; `median_nonmissing` High/Low split.
Figure 5 reports hotspot-associated target-gene expression across four categories
defined by localization direction and target regulation (Nuclear accumulation ×
activate/repress and cytoplasmic redistribution × activate/repress). The primary
analysis uses two-sided Wilcoxon signed-rank tests; BH correction is applied within
each cancer type and localization direction × target regulation category.
Per-phosphosite (unit-site) runs are retained only as legacy / sensitivity outputs.

## Analysis overview

The primary pipeline:

- builds Mixed / Predicted-candidate phospho-hotspots on each TF protein
  (Known ∪ Predicted sites; adjacent gap ≤15 aa; span ≤40 aa);
- re-annotates Import/Export gates as **pure nuclear / pure cytoplasmic**;
- uses matched CPTAC phosphoproteomics, proteomics, and RNA-seq tumor samples;
- defines hotspot activity as the per-cancer **mean z-score** of measured member sites;
- stratifies tumors by median hotspot activity among detected samples
  (`median_nonmissing`);
- evaluates curated signed TF target-gene expression (activate / repress) with
  `signed_target_mode=regulon_only` and `test_alternative=two-sided`;
- compares observed effects to random matched controls; and
- optionally runs hotspot-level Cox survival on BH-significant cancer×hotspot pairs.

| Script | Role |
|--------|------|
| [`scripts/build_hotspots_mixed_pred_filter.py`](scripts/build_hotspots_mixed_pred_filter.py) | **Primary** hotspot catalog (adj≤15, span≤40; Mixed / Predicted candidate) |
| [`scripts/reannotate_hotspot_pure_direction.py`](scripts/reannotate_hotspot_pure_direction.py) | Pure nuclear/cytoplasmic Import/Export gates |
| [`scripts/run_hotspot_target_regulation_analysis.py`](scripts/run_hotspot_target_regulation_analysis.py) | Hotspot CPTAC scan (defaults: distance=15, regulon_only, two-sided) |
| [`scripts/run_four_arm_twosided_pipeline.sh`](scripts/run_four_arm_twosided_pipeline.sh) | **Full four-arm Figure 5 runner** |
| [`scripts/analyze_two_sided_concordance.py`](scripts/analyze_two_sided_concordance.py) | Figure **5b** concordance; Table S4 association table |
| [`scripts/plot_two_sided_four_arm_significance_heatmap.py`](scripts/plot_two_sided_four_arm_significance_heatmap.py) | Figure **5** four-arm hotspot × cancer heatmap |
| [`scripts/plot_hotspot_figure4.py`](scripts/plot_hotspot_figure4.py) | Per-arm association example panels (script name historical) |
| [`scripts/run_figure4_spatial_direction_panels.py`](scripts/run_figure4_spatial_direction_panels.py) | Figure **4A–C** / S5 spatial & direction panels (not CPTAC stratification) |
| [`scripts/run_hotspot_survival_cox.py`](scripts/run_hotspot_survival_cox.py) | Hotspot activity Cox / KM on BH-significant pairs |
| [`scripts/build_phospho_hotspots.py`](scripts/build_phospho_hotspots.py) | Earlier distance catalogs (singletons / known_* classes); **not** manuscript primary |
| [`scripts/run_import_target_regulation_analysis.py`](scripts/run_import_target_regulation_analysis.py) | **Legacy** unit-site CPTAC pipeline |
| [`scripts/plot_phosphosite_across_cancers.py`](scripts/plot_phosphosite_across_cancers.py) | Legacy / example per-site across-cancer boxplots |

## Requirements

- Root dependencies: [`requirements.txt`](../requirements.txt)
- Additional packages: `pyensembl` (Ensembl gene annotation; commented in [`requirements.txt`](../requirements.txt))
- Ensure the required Ensembl release cache or annotation files are available locally before running the CPTAC scan

## Data

Large CPTAC and reference files are **not** tracked in Git. Prepare or symlink inputs under `cptac_analysis/data/` before running. Required input categories:

- CPTAC phosphoproteomics, proteomics, and RNA-seq matrices
- FuncTransport + Direction summary / stable PhosLoc-Transport predictions (v11 + D3 + Platt)
- curated signed TF target sets
- known positive phosphosite labels
- gene annotation / id-mapping files

See [`data/README.md`](data/README.md) for the expected directory layout.

## Primary run (Figure 5 four-arm; manuscript-aligned)

CLI entry points live under `scripts/` (thin wrappers); implementations are in `src/`.

### One-shot runner

```bash
cd cptac_analysis
bash scripts/run_four_arm_twosided_pipeline.sh
```

### Explicit four-arm commands (same parameters as the runner)

```bash
cd cptac_analysis
export PYTHONPATH="${PWD}/scripts:${PYTHONPATH:-}"

# 1) Mixed / Predicted-candidate catalog (99 hotspots / 85 TFs for v11)
python scripts/build_hotspots_mixed_pred_filter.py \
  --output-dir results/import_target_regulation_hotspot_v11_147pos_d3_platt/hotspots_mixed_pred_filter

# 2) Pure-direction Import/Export gates
python scripts/reannotate_hotspot_pure_direction.py \
  --src-dir results/import_target_regulation_hotspot_v11_147pos_d3_platt/hotspots_mixed_pred_filter \
  --output-dir results/import_target_regulation_hotspot_v11_147pos_d3_platt/hotspots_mixed_pred_filter_pure_direction \
  --distance 15

HOTSPOT_DIR=results/import_target_regulation_hotspot_v11_147pos_d3_platt/hotspots_mixed_pred_filter_pure_direction
OUT=results/hotspot_mixed_pred_filter_pure_regulon_only_twosided_20260929

# 3a) Import arm CPTAC scan (covers Import×activate and Import×repress tables)
python scripts/run_hotspot_target_regulation_analysis.py \
  --hotspot-dir "$HOTSPOT_DIR" \
  --distance 15 \
  --activity mean_z \
  --phospho-split-mode median_nonmissing \
  --directions "Nuclear Import" \
  --signed-target-mode regulon_only \
  --test-alternative two-sided \
  --random-iterations 100 \
  --confounder-analysis none \
  --output-dir "$OUT/Import_activate/cptac"

# 3b) Export arm CPTAC scan
python scripts/run_hotspot_target_regulation_analysis.py \
  --hotspot-dir "$HOTSPOT_DIR" \
  --distance 15 \
  --activity mean_z \
  --phospho-split-mode median_nonmissing \
  --directions "Nuclear Export" \
  --signed-target-mode regulon_only \
  --test-alternative two-sided \
  --random-iterations 100 \
  --confounder-analysis none \
  --output-dir "$OUT/Export_activate/cptac"

# Optional: per-arm association example panels (script name is historical)
python scripts/plot_hotspot_figure4.py \
  --results-dir "$OUT/Import_activate/cptac" \
  --output-dir "$OUT/Import_activate/figure4" \
  --direction-short Import --target-regulation activate
python scripts/plot_hotspot_figure4.py \
  --results-dir "$OUT/Import_activate/cptac" \
  --output-dir "$OUT/Import_repress/figure4" \
  --direction-short Import --target-regulation repress
python scripts/plot_hotspot_figure4.py \
  --results-dir "$OUT/Export_activate/cptac" \
  --output-dir "$OUT/Export_activate/figure4" \
  --direction-short Export --target-regulation activate
python scripts/plot_hotspot_figure4.py \
  --results-dir "$OUT/Export_activate/cptac" \
  --output-dir "$OUT/Export_repress/figure4" \
  --direction-short Export --target-regulation repress

# 4) Figure 5 concordance + four-arm heatmap + Table S4 flags
python scripts/analyze_two_sided_concordance.py
python scripts/plot_two_sided_four_arm_significance_heatmap.py

# 5) Optional: Figure 4A–C spatial / direction definition panels
python scripts/run_figure4_spatial_direction_panels.py
```

### Catalog definition (Methods; v11_147pos_d3_platt)

- Sites clustered: **Known** (PMID present) ∪ **Predicted** (ordinary Observed excluded)
- Break cluster if adjacent gap **>15 aa** or span **>40 aa**
- Keep **Mixed evidence** (≥1 Known + ≥1 Predicted) or **Predicted candidate**
  (no Known, ≥3 Predicted, mean FuncTransport ≥0.6)
- Fixed size: **99** hotspots (48 Mixed / 51 Predicted) on **85** TFs
- Do **not** treat equal hotspot counts from `build_phospho_hotspots.py` as the same definition

**Figure 4c:** representative phospho-hotspots in STAT1 (T699 to S708), AR (S647 to T653)
and CUX1 (Y1209 to S1218).

**Figure 5:** hotspot-associated target-gene expression across four categories defined by
localization direction and target regulation (Nuclear accumulation × activate/repress and
cytoplasmic redistribution × activate/repress).

Figure 5b concordance among BH-significant evaluable associations:
Import×activate **17/20**, Import×repress **0/1**, Export×activate **16/45**, Export×repress **8/13**.

## Reference output directories

| Output | Path |
|--------|------|
| Primary hotspot catalog | `results/import_target_regulation_hotspot_v11_147pos_d3_platt/hotspots_mixed_pred_filter/` |
| Pure-direction catalog | `.../hotspots_mixed_pred_filter_pure_direction/` |
| **Fig. 5 four-arm two-sided (primary)** | `results/hotspot_mixed_pred_filter_pure_regulon_only_twosided_20260929/` |
| Spatial Fig 4A–C / S5 | `results/figure4_spatial_direction_panels_v11_147pos_d3_platt/` |
| Hotspot Cox | `results/survival_analysis/hotspot_cox_*` |
| Tables S3–S4 (repo) | [`../supplement/`](../supplement/) |
| Legacy unit-site pipeline | `results/import_target_regulation/` / `results/import_target_regulation_v11_147pos_d3_platt/` |

## Statistical notes (Figure 5 / Table S4)

The primary analysis uses two-sided Wilcoxon signed-rank tests on paired target-gene
expression summaries. Associations require at least 10 paired target genes. BH correction
is applied within each cancer type and localization direction × target regulation category.

- Table S3 reports unadjusted empirical one-sided P values from 1,000 within-protein
  permutations with a plus-one correction.
- Table S4 lists all evaluable cancer–hotspot associations, including raw P values,
  BH-adjusted q values and BH significance flags.

## Representative hotspot examples (Figure 4c)

```bash
cd cptac_analysis
python scripts/run_figure4_spatial_direction_panels.py --only 4c
# exemplars: STAT1_T699-S708, AR_S647-T653, CUX1_Y1209-S1218
```

## Legacy unit-site analysis

Unit-site stratification is **not** the manuscript primary analysis:

```bash
cd cptac_analysis
python scripts/run_import_target_regulation_analysis.py
python scripts/plot_phosphosite_across_cancers.py
python scripts/plot_significant_sites_combined.py
```

Default legacy outputs: `results/import_target_regulation/`.

## Related documentation

| Resource | Description |
|----------|-------------|
| [data/README.md](data/README.md) | CPTAC and reference data layout |
| [../docs/FIGURES_AND_PREDICTION.md](../docs/FIGURES_AND_PREDICTION.md) | Figure and prediction scripts |
| [../docs/TRAINING_RUNS.md](../docs/TRAINING_RUNS.md) | Finalized model training + Stage 3 reference run |
| [../supplement/`](../supplement/) | Supplemental Tables S3–S4 |
| [../import_export/results/1_transport_classifier_results/joint_score_v11_147pos_d3_platt/](../import_export/results/1_transport_classifier_results/joint_score_v11_147pos_d3_platt/) | Stable import/export predictions (D3+Platt; from Zenodo precomputed) |
| [../functional/data/dataset_phos_site/TF_positive_phos_site_0608.csv](../functional/data/dataset_phos_site/TF_positive_phos_site_0608.csv) | Known positive phosphosite labels (**147** sites) |
