PhosLoc-Transport - Data bundle README
======================================

This Zenodo record contains processed inputs and model artifacts for the
PhosLoc-Transport three-stage pipelines (functional classification,
import/export direction, and CPTAC target-regulation analysis).

Source code (not included here):
  https://github.com/TaoYySC/phosloc-transport

Clone the code repository first, then download and extract the archives
below into the repository root.


Files in this record
--------------------

  PhosLoc-Transport_functional_data.tar
    Stage 1: features, ESM embeddings, AlphaFold PDBs,
             model checkpoints, precomputed CSVs
    Archive top-level path: data/
    Extracts to: functional/data/

  PhosLoc-Transport_import_export_data.tar
    Stage 2: features, model artifacts, Platt calibrator,
             precomputed predictions and joint scores
    Archive top-level path: data/
    Extracts to: import_export/data/

  PhosLoc-Transport_cptac_source.tar
    Stage 3: CPTAC omics, ChIP-Atlas targets, regulons,
             UniProt idmapping
    Archive top-level path: source/
    Extracts to: cptac_analysis/data/source/


Quick start
-----------

1. Clone the code repository:

     git clone https://github.com/TaoYySC/phosloc-transport.git
     cd phosloc-transport

2. Place all four files from this Zenodo record (three archives + this
   README) in any location on your computer. The archives do not need
   to sit inside the cloned repository before extraction.

3. Open a terminal and change to the repository root (the folder that
   contains functional/, import_export/, and cptac_analysis/). Then
   extract each archive into the matching subproject directory. Use the
   full path to each archive if it is stored outside the repository.

   Linux / macOS (from repository root):

     tar -xf /path/to/PhosLoc-Transport_functional_data.tar -C functional
     tar -xf /path/to/PhosLoc-Transport_import_export_data.tar -C import_export
     tar -xf /path/to/PhosLoc-Transport_cptac_source.tar -C cptac_analysis/data

   If your system reports that a file is gzip-compressed, add -z:

     tar -xzf /path/to/PhosLoc-Transport_functional_data.tar -C functional
     tar -xzf /path/to/PhosLoc-Transport_import_export_data.tar -C import_export
     tar -xzf /path/to/PhosLoc-Transport_cptac_source.tar -C cptac_analysis/data

   Windows PowerShell (from repository root):

     tar -xf D:\path\to\PhosLoc-Transport_functional_data.tar -C functional
     tar -xf D:\path\to\PhosLoc-Transport_import_export_data.tar -C import_export
     tar -xf D:\path\to\PhosLoc-Transport_cptac_source.tar -C cptac_analysis\data

   If extraction fails without -z, retry with -xzf instead of -xf.

   IMPORTANT: Do not extract these archives directly at the repository
   root without -C. The functional and import/export archives contain a
   top-level data/ directory, and the CPTAC archive contains a top-level
   source/ directory. The -C targets above restore those paths to the
   locations expected by the scripts.


Expected layout after extraction
--------------------------------

  phosloc-transport/
  |-- functional/
  |   `-- data/
  |       |-- model_artifacts/
  |       |-- TF_esm_embedding/
  |       |-- fasta/
  |       `-- ...
  |-- import_export/
  |   `-- data/
  |       |-- model_artifacts/
  |       |-- fasta/
  |       `-- ...
  `-- cptac_analysis/
      `-- data/
          `-- source/
              |-- 1.cpatac/
              |-- 4.chipaltas/
              `-- 5.regulons/


Verify installation (optional)
------------------------------

  functional/data/model_artifacts/          should exist
  import_export/data/model_artifacts/       should exist
  cptac_analysis/data/source/1.cpatac/      should exist


Run the pipelines
-----------------

  pip install -r requirements.txt
  # optional for Stage 3: uncomment pyensembl in requirements.txt, then pip install -r requirements.txt

  See the repository README and subproject README files:
    functional/README.md
    import_export/README.md
    cptac_analysis/README.md


Notes
-----

  - You only need all three archives to reproduce the finalized runs.
    For Stage 1 and Stage 2 only, download the first two archives.
  - Runtime outputs are written to each subproject's results/ directory
    and are not part of this bundle.
  - For the full data inventory, see DATA.md in the GitHub repository.
