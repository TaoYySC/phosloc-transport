#!/usr/bin/env bash
# Manuscript Figure 5: four-arm CPTAC scan (Import/Export × activate/repress).
# Settings: mixed_pred_filter pure-direction catalog; regulon_only; two-sided Wilcoxon.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="$ROOT/scripts"
PY="${PYTHON:-python}"

SRC_HOTSPOT="${SRC_HOTSPOT:-$ROOT/results/import_target_regulation_hotspot_v11_147pos_d3_platt/hotspots_mixed_pred_filter}"
HOTSPOT_DIR="${HOTSPOT_DIR:-$ROOT/results/import_target_regulation_hotspot_v11_147pos_d3_platt/hotspots_mixed_pred_filter_pure_direction}"
OUT="${OUT:-$ROOT/results/hotspot_mixed_pred_filter_pure_regulon_only_twosided_20260929}"

IMP_ACT="$OUT/Import_activate"
IMP_REP="$OUT/Import_repress"
EXP_ACT="$OUT/Export_activate"
EXP_REP="$OUT/Export_repress"

export PYTHONPATH="${SCRIPTS}:${PYTHONPATH:-}"
mkdir -p "$OUT" "$SRC_HOTSPOT" "$HOTSPOT_DIR" \
  "$IMP_ACT/cptac" "$IMP_ACT/figure4" "$IMP_REP/figure4" \
  "$EXP_ACT/cptac" "$EXP_ACT/figure4" "$EXP_REP/figure4"
chmod +x "$0" 2>/dev/null || true

echo "======== [0] Catalog (mixed_pred_filter) ========"
"$PY" "$SCRIPTS/build_hotspots_mixed_pred_filter.py" \
  --output-dir "$SRC_HOTSPOT" \
  2>&1 | tee "$OUT/00_build_catalog.log"

echo "======== [0b] Pure-direction gates ========"
"$PY" "$SCRIPTS/reannotate_hotspot_pure_direction.py" \
  --src-dir "$SRC_HOTSPOT" \
  --output-dir "$HOTSPOT_DIR" \
  --distance 15 \
  2>&1 | tee "$OUT/00_catalog.log"

run_scan() {
  local direction="$1"
  local out_cptac="$2"
  local log="$3"
  "$PY" "$SCRIPTS/run_hotspot_target_regulation_analysis.py" \
    --hotspot-dir "$HOTSPOT_DIR" \
    --distance 15 \
    --activity mean_z \
    --phospho-split-mode median_nonmissing \
    --directions "$direction" \
    --signed-target-mode regulon_only \
    --test-alternative two-sided \
    --random-iterations 100 \
    --confounder-analysis none \
    --output-dir "$out_cptac" \
    2>&1 | tee "$log"
}

echo "======== [1] CPTAC Import (two-sided, regulon_only) ========"
run_scan "Nuclear Import" "$IMP_ACT/cptac" "$OUT/01_import_cptac.log"

echo "======== [2] Import × activate association panels ========"
"$PY" "$SCRIPTS/plot_hotspot_figure4.py" \
  --results-dir "$IMP_ACT/cptac" \
  --output-dir "$IMP_ACT/figure4" \
  --direction-short Import \
  --target-regulation activate \
  2>&1 | tee "$OUT/02_import_activate_figure4.log"

echo "======== [3] Import × repress association panels ========"
rm -rf "$IMP_REP/cptac"
ln -sfn "$IMP_ACT/cptac" "$IMP_REP/cptac"
"$PY" "$SCRIPTS/plot_hotspot_figure4.py" \
  --results-dir "$IMP_REP/cptac" \
  --output-dir "$IMP_REP/figure4" \
  --direction-short Import \
  --target-regulation repress \
  2>&1 | tee "$OUT/04_import_repress_figure4.log"

echo "======== [4] CPTAC Export (two-sided, regulon_only) ========"
run_scan "Nuclear Export" "$EXP_ACT/cptac" "$OUT/05_export_cptac.log"

echo "======== [5] Export × activate association panels ========"
"$PY" "$SCRIPTS/plot_hotspot_figure4.py" \
  --results-dir "$EXP_ACT/cptac" \
  --output-dir "$EXP_ACT/figure4" \
  --direction-short Export \
  --target-regulation activate \
  2>&1 | tee "$OUT/06_export_activate_figure4.log"

echo "======== [6] Export × repress association panels ========"
rm -rf "$EXP_REP/cptac"
ln -sfn "$EXP_ACT/cptac" "$EXP_REP/cptac"
"$PY" "$SCRIPTS/plot_hotspot_figure4.py" \
  --results-dir "$EXP_REP/cptac" \
  --output-dir "$EXP_REP/figure4" \
  --direction-short Export \
  --target-regulation repress \
  2>&1 | tee "$OUT/08_export_repress_figure4.log"

echo "======== [7] Figure 5 concordance + four-arm heatmap ========"
"$PY" "$SCRIPTS/analyze_two_sided_concordance.py" 2>&1 | tee "$OUT/09_concordance.log"
"$PY" "$SCRIPTS/plot_two_sided_four_arm_significance_heatmap.py" 2>&1 | tee "$OUT/10_heatmap.log"

echo "======== DONE ======== outputs under $OUT"
