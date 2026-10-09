PhosLoc-Transport — Data bundle README
========================================

This Zenodo record contains processed inputs and model artifacts for the
PhosLoc-Transport pipelines (Localization-Regulatory Classifier, Localization
Direction Classifier, and CPTAC phospho-hotspot validation).

Source code (not included here):
  https://github.com/TaoYySC/phosloc-transport

Clone the code repository first, then download and extract the archives
below into the repository root.


Files in this record
--------------------

Base bundles (Stage 1–2 training/prediction + CPTAC omics source):

  PhosLoc-Transport_functional_data.tar.gz
    ~5.7 GB — Stage 1: features, ESM embeddings, AlphaFold PDBs,
             model checkpoints, precomputed CSVs
    Extracts to: functional/data/

  PhosLoc-Transport_import_export_data.tar.gz
    ~5.1 GB — Stage 2: embeddings symlink targets, model artifacts,
              Platt calibrator, precomputed predictions and joint scores
    Extracts to: import_export/data/

  PhosLoc-Transport_cptac_source.tar.gz
    ~1.1 GB — Stage 3: CPTAC omics, ChIP-Atlas targets, regulons,
              UniProt idmapping
    Extracts to: cptac_analysis/data/source/

Manuscript v11 / hotspot add-ons (required for Figure 5 / Tables S3–S4):

  PhosLoc-Transport_v11_hotspot_inputs.tar.gz
    ~5 MB — v11 147-positive set, reported co-regulatory cluster-sheet
            sites, v11 functional ensemble predictions, v11 joint-score
            tables + FuncTransport×Direction summary, Mixed/Predicted
            hotspot catalogs (including pure-direction gates)
    Extracts to:
      functional/data/...
      import_export/data/precomputed/...
      cptac_analysis/data/hotspot_catalogs/

Optional Figure 5 result snapshot (skip if you will re-run CPTAC):

  PhosLoc-Transport_fig5_twosided_results.tar.gz
    ~0.5–1 GB — four-arm two-sided CPTAC outputs used for Figure 5
    Extracts to:
      cptac_analysis/results/hotspot_mixed_pred_filter_pure_regulon_only_twosided_20260929/


Quick start
-----------

1. Clone the code repository:

     git clone https://github.com/TaoYySC/phosloc-transport.git
     cd phosloc-transport

2. Download archives from this Zenodo record.

3. From the repository root, extract (add -z if needed for gzip):

     tar -xzf /path/to/PhosLoc-Transport_functional_data.tar.gz
     tar -xzf /path/to/PhosLoc-Transport_import_export_data.tar.gz
     tar -xzf /path/to/PhosLoc-Transport_cptac_source.tar.gz
     tar -xzf /path/to/PhosLoc-Transport_v11_hotspot_inputs.tar.gz
     # optional:
     tar -xzf /path/to/PhosLoc-Transport_fig5_twosided_results.tar.gz

IMPORTANT: Run extractions from the repository root (the folder that contains
functional/, import_export/, and cptac_analysis/). Each archive already
contains the correct top-level path prefixes.


Expected layout (v11 hotspot inputs)
------------------------------------

  functional/data/dataset_phos_site/TF_positive_phos_site_0608.csv
  functional/data/dataset_phos_site/co_working_multi_site_from_cluster_sheets.csv
  functional/data/precomputed/2_1_functional_classifier_results/predictions/v11_147pos_5_folds_ensemble_predictions.csv
  import_export/data/precomputed/1_transport_classifier_results/joint_score_v11_147pos_d3_platt/
  import_export/data/precomputed/1_transport_classifier_results/tf_phos_site_FuncTransport_Direction_summary_with_PMID_annotation_v11_147pos_d3_platt.csv
  cptac_analysis/data/hotspot_catalogs/hotspots_mixed_pred_filter/
  cptac_analysis/data/hotspot_catalogs/hotspots_mixed_pred_filter_pure_direction/


Notes
-----

- Large runtime trees under **/results/** are not required for Stage 1–2
  prediction. Figure 5 can be regenerated from the v11 hotspot inputs + CPTAC
  source, or restored from the optional fig5 results archive.
- Supplemental Tables S3–S4 are shipped in the GitHub repository under
  supplement/ (small Excel files).
- Third-party resources (CPTAC, LinkedOmicsKB, ChIP-Atlas, CollecTRI,
  UniProt, AlphaFold, Ensembl) remain subject to their original terms.

See also DATA.md in the GitHub repository for the canonical in-repo inventory.
