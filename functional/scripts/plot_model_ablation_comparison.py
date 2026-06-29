# Panels: Figure 2(b)


# Monorepo path constants (monorepo relative paths)
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PRECOMPUTED = PROJECT_ROOT / "data" / "precomputed"
TF_FAMILY_PATH = PROJECT_ROOT / "data" / "TF_family" / "TF_Information.txt"

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RUN_DIRS = {
    "ESM Window+Site+PDB": str(DATA_PRECOMPUTED / "run_20260610_204935_ESM Window+Site+PDB" / "Functional_Transport"),
    "ESM Window+Site": str(DATA_PRECOMPUTED / "run_20260610_210322_ESM Window+Site" / "Functional_Transport"),
    "ESM Window": str(DATA_PRECOMPUTED / "run_20260610_210449_ESM Window" / "Functional_Transport"),
}

N_FOLDS = 5

OUTPUT_DIR = PROJECT_ROOT / "results" / "2_1_functional_classifier_results" / "model_ablation_comparison"

METRICS = {
    "test_auprc": "AUPRC",
    "test_auroc": "AUROC",
}

MODEL_BAR_COLORS = {
    "ESM Window+Site+PDB": "#C0392B",
    "ESM Window+Site": "#2471A3",
    "ESM Window": "#D68910",
}

MODEL_XLABELS = {
    "ESM Window+Site+PDB": "Window+Site+PDB",
    "ESM Window+Site": "Window+Site",
    "ESM Window": "Window",
}

FIG_W_MM = 110
FIG_H_MM = 34
FIGSIZE = (FIG_W_MM / 25.4, FIG_H_MM / 25.4)
SAVE_DPI = 300
PLOT_WIDTH_MM = 43.2
MARGIN_LEFT_MM = 16.0
SAVE_PAD_INCHES = 0.03

FONT_SIZE_BASE = 8
FONT_SIZE_TICK = 7
FONT_SIZE_ANNOT = 7
FONT_SIZE_LEGEND = 7
BAR_WIDTH = 0.28
METRIC_GROUP_GAP = BAR_WIDTH
AXES_LEFT = MARGIN_LEFT_MM / FIG_W_MM
AXES_RIGHT = (MARGIN_LEFT_MM + PLOT_WIDTH_MM) / FIG_W_MM
AXES_BOTTOM = 0.16
AXES_TOP = 0.96
LEGEND_LOC = "upper left"
LEGEND_BBOX_X = AXES_RIGHT + 0.012
X_PAD = 0.06


def _metric_group_centers(n_models):
    group_width = BAR_WIDTH * n_models
    return np.array([0.0, group_width + METRIC_GROUP_GAP])


def _save_figure(fig, output_dir, stem, legend=None):
    save_kwargs = {
        "facecolor": "white",
        "edgecolor": "none",
        "bbox_inches": "tight",
        "pad_inches": SAVE_PAD_INCHES,
    }
    if legend is not None:
        save_kwargs["bbox_extra_artists"] = [legend]
    fig.savefig(output_dir / f"{stem}.png", dpi=SAVE_DPI, **save_kwargs)
    fig.savefig(output_dir / f"{stem}.pdf", **save_kwargs)


def _metrics_all_folds_path(run_dir):
    return Path(run_dir) / "metrics_all_folds.csv"


def load_cv_summary_table(run_dirs, n_folds=N_FOLDS):
    records = []

    for model_name, run_dir in run_dirs.items():
        csv_path = _metrics_all_folds_path(run_dir)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

        df = pd.read_csv(csv_path)
        available_folds = sorted(df["fold"].unique().tolist())
        if len(available_folds) != n_folds:
            raise ValueError(
                f"Expected {n_folds} folds in {csv_path}, "
                f"found {len(available_folds)}: {available_folds}"
            )

        record = {
            "model": model_name,
            "n_folds": len(df),
            "source_csv": str(csv_path),
        }
        for metric_key in METRICS:
            if metric_key not in df.columns:
                raise ValueError(f"Missing required metric column {metric_key} in {csv_path}")
            values = df[metric_key].astype(float)
            record[f"{metric_key}_mean"] = float(values.mean())
            record[f"{metric_key}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )

        records.append(record)

    summary_df = pd.DataFrame(records)
    summary_df["model"] = pd.Categorical(
        summary_df["model"],
        categories=list(run_dirs.keys()),
        ordered=True,
    )
    return summary_df.sort_values("model").reset_index(drop=True)


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def _metric_bar_offsets(n_models):
    group_width = BAR_WIDTH * n_models
    return np.linspace(
        -group_width / 2 + BAR_WIDTH / 2,
        group_width / 2 - BAR_WIDTH / 2,
        n_models,
    )


def _axes_top_figure_y(ax):
    return ax.get_position().y1


def _apply_bar_plot_layout(fig, ax, n_models):
    group_width = BAR_WIDTH * n_models
    metric_x = _metric_group_centers(n_models)
    ax.set_xlim(
        metric_x[0] - group_width / 2 - X_PAD,
        metric_x[1] + group_width / 2 + X_PAD,
    )
    _style_axes(ax)
    fig.subplots_adjust(
        left=AXES_LEFT,
        right=AXES_RIGHT,
        top=AXES_TOP,
        bottom=AXES_BOTTOM,
    )
    legend = ax.legend(
        frameon=False,
        fontsize=FONT_SIZE_LEGEND,
        loc=LEGEND_LOC,
        bbox_to_anchor=(LEGEND_BBOX_X, _axes_top_figure_y(ax)),
        bbox_transform=fig.transFigure,
        handlelength=1.2,
        borderaxespad=0.0,
        labelspacing=0.22,
    )
    legend.set_clip_on(False)
    return metric_x, legend


MODEL_LABEL_Y_EXTRA = {
    "ESM Window+Site+PDB": {
        "test_auprc": 0.065,
        "test_auroc": 0.115,
    },
    "ESM Window": {
        "test_auroc": -0.010,
    },
}


def _label_y(mean, std, model_name=None, metric_key=None):
    y = mean + std + 0.025
    if model_name is None or metric_key is None:
        return y
    y += MODEL_LABEL_Y_EXTRA.get(model_name, {}).get(metric_key, 0.0)
    return y


def plot_prc_roc_panels(summary_df, output_dir):
    model_names = summary_df["model"].astype(str).tolist()
    legend_labels = model_names
    colors = [MODEL_BAR_COLORS.get(name, "#333333") for name in model_names]
    offsets = _metric_bar_offsets(len(model_names))

    auprc_means = summary_df["test_auprc_mean"].values
    auprc_stds = summary_df["test_auprc_std"].values
    auroc_means = summary_df["test_auroc_mean"].values
    auroc_stds = summary_df["test_auroc_std"].values
    metric_x = _metric_group_centers(len(model_names))

    error_kw = {"elinewidth": 0.8, "ecolor": "#333333", "capthick": 0.8}

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for i, (model_name, legend_label, color) in enumerate(
        zip(model_names, legend_labels, colors)
    ):
        auprc_x = metric_x[0] + offsets[i]
        auroc_x = metric_x[1] + offsets[i]

        ax.bar(
            auprc_x,
            auprc_means[i],
            width=BAR_WIDTH,
            yerr=auprc_stds[i],
            capsize=2,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=legend_label,
            error_kw=error_kw,
        )
        ax.bar(
            auroc_x,
            auroc_means[i],
            width=BAR_WIDTH,
            yerr=auroc_stds[i],
            capsize=2,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            error_kw=error_kw,
        )

        ax.text(
            auprc_x,
            _label_y(auprc_means[i], auprc_stds[i], model_name, "test_auprc"),
            f"{auprc_means[i]:.3f}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_ANNOT,
        )
        ax.text(
            auroc_x,
            _label_y(auroc_means[i], auroc_stds[i], model_name, "test_auroc"),
            f"{auroc_means[i]:.3f}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_ANNOT,
        )

    ax.set_xticks(metric_x)
    ax.set_xticklabels(
        [METRICS["test_auprc"], METRICS["test_auroc"]],
        fontsize=FONT_SIZE_TICK,
    )
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", labelpad=2)

    _apply_bar_plot_layout(fig, ax, len(model_names))

    _save_figure(fig, output_dir, "model_ablation_prc_roc_panels", ax.get_legend())
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot Functional Transport model ablation AUPRC/AUROC comparison "
            "using 5-fold CV mean ± std on the test set."
        )
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=N_FOLDS,
        help=f"Expected number of CV folds in metrics_all_folds.csv. Default: {N_FOLDS}.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    n_folds = int(args.n_folds)

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE_BASE,
            "axes.labelsize": FONT_SIZE_BASE,
            "xtick.labelsize": FONT_SIZE_TICK,
            "ytick.labelsize": FONT_SIZE_TICK,
        }
    )

    summary_df = load_cv_summary_table(RUN_DIRS, n_folds=n_folds)
    print(f"[INFO] Aggregating {n_folds}-fold CV test metrics (mean ± std)")

    plot_prc_roc_panels(summary_df, output_dir)
    summary_df.to_csv(output_dir / "model_ablation_comparison_summary.csv", index=False)

    print("Saved outputs to:")
    print(output_dir)
    print(
        summary_df[
            [
                "model",
                "n_folds",
                "test_auprc_mean",
                "test_auprc_std",
                "test_auroc_mean",
                "test_auroc_std",
            ]
        ]
    )


if __name__ == "__main__":
    main()
