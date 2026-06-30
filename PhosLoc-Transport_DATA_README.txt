PhosLoc-Transport 鈥?Data bundle README
========================================

This Zenodo record contains processed inputs and model artifacts for the
PhosLoc-Transport three-stage pipelines (functional classification,
import/export direction, and CPTAC target-regulation analysis).

Source code (not included here):
  https://github.com/TaoYySC/PhosLoc-TF

Clone the code repository first, then download and extract the archives
below into the repository root.


Files in this record
--------------------

  PhosLoc-Transport_functional_data.tar
    ~7 GB  鈥?Stage 1: features, ESM embeddings, AlphaFold PDBs,
             model checkpoints, precomputed CSVs
    Extracts to: functional/data/

  PhosLoc-Transport_import_export_data.tar
    ~5.6 GB 鈥?Stage 2: features, model artifacts, Platt calibrator,
              precomputed predictions and joint scores
    Extracts to: import_export/data/

  PhosLoc-Transport_cptac_source.tar.tar
    ~3.8 GB 鈥?Stage 3: CPTAC omics, ChIP-Atlas targets, regulons,
              UniProt idmapping
    Extracts to: cptac_analysis/data/source/


Quick start
-----------

1. Clone the code repository:

     git clone https://github.com/TaoYySC/PhosLoc-TF.git
     cd PhosLoc-TF

2. Place all four files from this Zenodo record (three archives + this
   README) in any location on your computer. The archives do not need
   to sit inside the cloned repository before extraction.

3. Open a terminal, change to the repository root (the folder that
   contains functional/, import_export/, and cptac_analysis/), then
   extract all three archives. Use the full path to each archive if it
   is stored outside the repo.

   Linux / macOS (from repository root):

     tar -xf /path/to/PhosLoc-Transport_functional_data.tar
     tar -xf /path/to/PhosLoc-Transport_import_export_data.tar
     tar -xf /path/to/PhosLoc-Transport_cptac_source.tar.tar

   If your system reports that the file is gzip-compressed, add -z:

     tar -xzf /path/to/PhosLoc-Transport_functional_data.tar
     tar -xzf /path/to/PhosLoc-Transport_import_export_data.tar
     tar -xzf /path/to/PhosLoc-Transport_cptac_source.tar.tar

   Windows PowerShell (from repository root):

     tar -xf D:\path\to\PhosLoc-Transport_functional_data.tar
     tar -xf D:\path\to\PhosLoc-Transport_import_export_data.tar
     tar -xf D:\path\to\PhosLoc-Transport_cptac_source.tar.tar

   If extraction fails without -z, retry with -xzf instead of -xf.

   IMPORTANT: Run these commands from the repository root (PhosLoc-TF/),
   not from inside functional/, import_export/, or cptac_analysis/.
   Each archive already contains the correct top-level path prefix
   (e.g. functional/data/...). Extracting at the repo root restores
   files to the locations expected by the scripts.


Expected layout after extraction
--------------------------------

  PhosLoc-TF/
  鈹溾攢鈹€ functional/
  鈹?  鈹斺攢鈹€ data/
  鈹?      鈹溾攢鈹€ model_artifacts/
  鈹?      鈹溾攢鈹€ TF_esm_embedding/
  鈹?      鈹溾攢鈹€ fasta/
  鈹?      鈹斺攢鈹€ ...
  鈹溾攢鈹€ import_export/
  鈹?  鈹斺攢鈹€ data/
  鈹?      鈹溾攢鈹€ model_artifacts/
  鈹?      鈹溾攢鈹€ fasta/
  鈹?      鈹斺攢鈹€ ...
  鈹斺攢鈹€ cptac_analysis/
      鈹斺攢鈹€ data/
          鈹斺攢鈹€ source/
              鈹溾攢鈹€ 1.cpatac/
              鈹溾攢鈹€ 4.chipaltas/
              鈹斺攢鈹€ 5.regulons/


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
    For Stage 1鈥? only, download the first two archives.
  - Runtime outputs are written to each subproject's results/ directory
    and are not part of this bundle.
  - For the full data inventory, see DATA.md in the GitHub repository.
