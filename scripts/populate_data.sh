#!/usr/bin/env bash
# Populate phosloc-transport data directories from local source projects.
# Run once after cloning the code repo; upload functional/data/ and import_export/data/ separately if needed.
#
# Environment overrides (all optional):
#   REPO       — monorepo root (default: parent of this scripts/ directory)
#   FUNC_SRC   — legacy phosloc-Func tree
#   IE_SRC     — legacy phosloc-ImportExport tree
#   TF_SRC     — legacy phosloc-TF tree (functional TF_family + CPTAC source)
#   CPTAC_SRC  — CPTAC / ChIP / regulon bundle (default: $TF_SRC/cancer_analysis/source)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PARENT="${PARENT:-$(dirname "$REPO")}"
FUNC_SRC="${FUNC_SRC:-$PARENT/phosloc-Func}"
IE_SRC="${IE_SRC:-$PARENT/phosloc-ImportExport}"
TF_SRC="${TF_SRC:-$PARENT/phosloc-TF}"
CPTAC_SRC="${CPTAC_SRC:-$TF_SRC/cancer_analysis/source}"

FUNC="$REPO/functional"
IE="$REPO/import_export"
CPTAC="$REPO/cptac_analysis"

copy_tree() {
  local src="$1"
  local dest="$2"
  if [[ ! -e "$src" ]]; then
    echo "[warn] missing source: $src" >&2
    return
  fi
  if [[ -L "$dest" ]]; then
    rm -f "$dest"
  fi
  mkdir -p "$(dirname "$dest")"
  echo "[rsync] $src -> $dest"
  rsync -a "$src/" "$dest/"
}

echo "=== Repo: $REPO ==="

echo "=== functional/data: large assets ==="
copy_tree "$FUNC_SRC/data/fasta" "$FUNC/data/fasta"
copy_tree "$FUNC_SRC/data/features" "$FUNC/data/features"
copy_tree "$FUNC_SRC/data/TF_esm_embedding" "$FUNC/data/TF_esm_embedding"
copy_tree "$FUNC_SRC/data/alphafold_tf_pdb" "$FUNC/data/alphafold_tf_pdb"
copy_tree "$TF_SRC/data/TF_family" "$FUNC/data/TF_family"

echo "=== functional/data: model artifacts (inference) ==="
copy_tree \
  "$FUNC_SRC/results/run_20260610_204935_ESM Window+Site+PDB/Functional_Transport/artifacts" \
  "$FUNC/data/model_artifacts/run_20260610_204935_ESM Window+Site+PDB/Functional_Transport/artifacts"

echo "=== functional/data: precomputed inputs (plotting / downstream) ==="
mkdir -p "$FUNC/data/precomputed"

if [[ -d "$REPO/functional/results/2_1_functional_classifier_results/predictions" ]]; then
  copy_tree \
    "$REPO/functional/results/2_1_functional_classifier_results/predictions" \
    "$FUNC/data/precomputed/2_1_functional_classifier_results/predictions"
fi

if [[ -d "$REPO/functional/results/3_figure3" ]]; then
  copy_tree "$REPO/functional/results/3_figure3" "$FUNC/data/precomputed/3_figure3"
elif [[ -d "$REPO/functional/results/3_fiugre3" ]]; then
  copy_tree "$REPO/functional/results/3_fiugre3" "$FUNC/data/precomputed/3_figure3"
fi

for run in \
  "run_20260610_204935_ESM Window+Site+PDB" \
  "run_20260610_210322_ESM Window+Site" \
  "run_20260610_210449_ESM Window"; do
  dest="$FUNC/data/precomputed/$run/Functional_Transport"
  mkdir -p "$dest"
  for f in metrics_all_folds.csv fixed_test_samples.csv fixed_test_predictions_by_fold.csv; do
    src="$REPO/functional/results/$run/Functional_Transport/$f"
    if [[ -f "$src" ]]; then
      cp -f "$src" "$dest/$f"
    fi
  done
done

echo "=== import_export/data: shared assets (copies from functional) ==="
copy_tree "$FUNC/data/fasta" "$IE/data/fasta"
copy_tree "$FUNC/data/TF_esm_embedding" "$IE/data/TF_esm_embedding"

echo "=== import_export/data: model artifacts (inference) ==="
IE_RUN_SRC="$IE_SRC/results/run_20260612_125646_esm_window_only_supcon_ce_import_pos/Import_vs_Export"
IE_RUN_DEST="$IE/data/model_artifacts/run_20260612_125646_esm_window_only_supcon_ce_import_pos/Import_vs_Export"
mkdir -p "$IE_RUN_DEST"
copy_tree "$IE_RUN_SRC/fold_artifacts" "$IE_RUN_DEST/fold_artifacts"
for f in platt_calibrator.json platt_calibration_meta.json run_meta.json; do
  if [[ -f "$IE_RUN_SRC/$f" ]]; then
    cp -f "$IE_RUN_SRC/$f" "$IE_RUN_DEST/$f"
  fi
done

echo "=== import_export/data: precomputed inputs ==="
mkdir -p "$IE/data/precomputed"

IE_PRE="$REPO/import_export/results"
if [[ -d "$IE_PRE/1_transport_classifier_results/esm_window_only_import_pos_predictions" ]]; then
  copy_tree \
    "$IE_PRE/1_transport_classifier_results/esm_window_only_import_pos_predictions" \
    "$IE/data/precomputed/1_transport_classifier_results/esm_window_only_import_pos_predictions"
fi
if [[ -d "$IE_PRE/1_transport_classifier_results/joint_score" ]]; then
  copy_tree \
    "$IE_PRE/1_transport_classifier_results/joint_score" \
    "$IE/data/precomputed/1_transport_classifier_results/joint_score"
fi

mkdir -p "$IE/data/precomputed/run_20260612_125646_esm_window_only_supcon_ce_import_pos/Import_vs_Export"
for f in metrics_all_runs.csv all_fold_test_predictions_platt.csv all_fold_test_predictions.csv; do
  src="$IE_PRE/run_20260612_125646_esm_window_only_supcon_ce_import_pos/Import_vs_Export/$f"
  if [[ -f "$src" ]]; then
    cp -f "$src" \
      "$IE/data/precomputed/run_20260612_125646_esm_window_only_supcon_ce_import_pos/Import_vs_Export/$f"
  fi
done

echo "=== cptac_analysis/data: CPTAC source copy ==="
if [[ -d "$CPTAC_SRC" ]]; then
  copy_tree "$CPTAC_SRC" "$CPTAC/data/source"
else
  echo "[warn] CPTAC source not found: $CPTAC_SRC" >&2
  echo "[warn] Copy CPTAC data manually — see cptac_analysis/data/README.md" >&2
fi

echo "=== cleanup legacy paths ==="
rm -rf "$REPO/external"
rm -f "$FUNC/results/run_20260610_204935_ESM Window+Site+PDB/Functional_Transport/artifacts" 2>/dev/null || true
rm -f "$IE/results/run_20260612_125646_esm_window_only_supcon_ce_import_pos/Import_vs_Export/fold_artifacts" 2>/dev/null || true

echo "=== Done ==="
