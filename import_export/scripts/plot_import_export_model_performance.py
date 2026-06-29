# Panels: Figure 3(b)


# Monorepo path constants (monorepo relative paths)
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
FUNCTIONAL_ROOT = REPO_ROOT / "functional"
DATA_PRECOMPUTED = PROJECT_ROOT / "data" / "precomputed"
FUNCTIONAL_PRECOMPUTED = FUNCTIONAL_ROOT / "data" / "precomputed"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_path = str(DATA_PRECOMPUTED / "run_20260612_125646_esm_window_only_supcon_ce_import_pos" / "Import_vs_Export" / "metrics_all_runs.csv")
output_dir = str(PROJECT_ROOT / "results" / "1_transport_classifier_results" / "model_performance")
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_path)

metrics = ["test_auroc", "test_auprc", "test_f1", "test_mcc"]
metric_labels = ["AUROC", "AUPRC", "F1", "MCC"]
bar_colors = ["#d54e4e", "#e79e98", "#ffe4df", "#73b2df"]

summary = df[metrics].agg(["mean", "std"]).T.reset_index()
summary.columns = ["metric", "mean", "std"]
summary["label"] = metric_labels

FONT_SIZE_SHIFT = 3

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.family": "DejaVu Sans",
    "font.size": 10 + FONT_SIZE_SHIFT,
    "axes.labelsize": 10 + FONT_SIZE_SHIFT,
    "xtick.labelsize": 10 + FONT_SIZE_SHIFT,
    "ytick.labelsize": 10 + FONT_SIZE_SHIFT,
})

fig, ax = plt.subplots(figsize=(4.2, 3.2))

x = np.arange(len(summary))
bars = ax.bar(
    x,
    summary["mean"],
    yerr=summary["std"],
    capsize=4,
    width=0.65,
    color=bar_colors,
    edgecolor="black",
    linewidth=0.8,
)

for i, row in summary.iterrows():
    ax.text(
        i,
        row["mean"] + row["std"] + 0.025,
        f'{row["mean"]:.3f}',
        ha="center",
        va="bottom",
        fontsize=9 + FONT_SIZE_SHIFT,
    )

ax.set_xticks(x)
ax.set_xticklabels(summary["label"])
ax.set_ylabel("Test performance")
ax.set_ylim(0, 1.1)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(False)

plt.tight_layout()

summary_path = os.path.join(output_dir, "test_metrics_summary.csv")
pdf_path = os.path.join(output_dir, "test_metrics_barplot_colored.pdf")
svg_path = os.path.join(output_dir, "test_metrics_barplot_colored.svg")
png_path = os.path.join(output_dir, "test_metrics_barplot_colored.png")

summary.to_csv(summary_path, index=False)
plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(svg_path, bbox_inches="tight")
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {summary_path}")
print(f"Saved: {pdf_path}")
print(f"Saved: {svg_path}")
print(f"Saved: {png_path}")