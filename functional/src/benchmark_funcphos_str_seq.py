# Panels: Figure 2(c)


# Monorepo path constants (monorepo relative paths)
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PRECOMPUTED = PROJECT_ROOT / "data" / "precomputed"
TF_FAMILY_PATH = PROJECT_ROOT / "data" / "TF_family" / "TF_Information.txt"

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


DEFAULT_RUN_DIR = (
    str(DATA_PRECOMPUTED / "run_20260610_204935_ESM Window+Site+PDB")
)
DEFAULT_EXPECTED_FOLDS = 5
DEFAULT_STR_CSV = (
    str(DATA_PRECOMPUTED / "3_figure3" / "funcphos_str_scores.csv")
)
DEFAULT_SEQ_CSV = (
    str(DATA_PRECOMPUTED / "3_figure3" / "funcphos_seq_scores.csv")
)
DEFAULT_PSP_REGULATORY_SITES = (
    str(PROJECT_ROOT / "data" / "dataset_phos_site" / "Regulatory_sites")
)
DEFAULT_OUTPUT_DIR = (
    str(PROJECT_ROOT / "results" / "2_1_functional_classifier_results" / "benchmark_results")
)

MODEL_ORDER = ["PhosLoc", "FuncPhos-STR", "FuncPhos-SEQ"]
MODEL_DISPLAY_NAMES = {
    "PhosLoc": "FuncTransport",
    "FuncPhos-STR": "FuncPhos-STR",
    "FuncPhos-SEQ": "FuncPhos-SEQ",
}
MODEL_COLORS = {
    "PhosLoc": "#E64B35",
    "FuncPhos-STR": "#4DBBD5",
    "FuncPhos-SEQ": "#00A087",
}
MODEL_SCORE_COLS = {
    "PhosLoc": "phosloc_score",
    "FuncPhos-STR": "funcphos_str_score_num",
    "FuncPhos-SEQ": "funcphos_seq_score_num",
}
SCORE_COLS = list(MODEL_SCORE_COLS.values())

PLOT_FIG_W_MM = 110
PLOT_FIG_H_MM = 34
PLOT_FIGSIZE = (PLOT_FIG_W_MM / 25.4, PLOT_FIG_H_MM / 25.4)
SAVE_DPI = 600
PLOT_WIDTH_MM = 43.2
MARGIN_LEFT_MM = 16.0
SAVE_PAD_INCHES = 0.03

FONT_SIZE_BASE = 8
FONT_SIZE_TICK = 7
FONT_SIZE_ANNOT = 7
FONT_SIZE_LEGEND = 7
BAR_WIDTH = 0.28
METRIC_GROUP_GAP = BAR_WIDTH
AXES_LEFT = MARGIN_LEFT_MM / PLOT_FIG_W_MM
AXES_RIGHT = (MARGIN_LEFT_MM + PLOT_WIDTH_MM) / PLOT_FIG_W_MM
AXES_BOTTOM = 0.16
AXES_TOP = 0.96
LEGEND_LOC = "upper left"
LEGEND_BBOX_X = AXES_RIGHT + 0.012
X_PAD = 0.06

METRIC_GROUPS = {
    "auprc": "AUPRC",
    "auroc": "AUROC",
}


PSP_META_COLUMNS = [
    "GENE",
    "PROTEIN",
    "ACC_ID",
    "ORGANISM",
    "MOD_RSD",
    "SITE_GRP_ID",
    "ON_FUNCTION",
    "ON_PROCESS",
    "ON_PROT_INTERACT",
    "ON_OTHER_INTERACT",
    "PMIDs",
    "NOTES",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark FuncTransport (five-fold ensemble on fixed test) vs "
            "FuncPhos-STR/SEQ, with optional PSP regulatory-site negative filtering."
        )
    )
    parser.add_argument("--run_dir", type=str, default=DEFAULT_RUN_DIR)
    parser.add_argument("--str_csv", type=str, default=DEFAULT_STR_CSV)
    parser.add_argument("--seq_csv", type=str, default=DEFAULT_SEQ_CSV)
    parser.add_argument("--psp_regulatory_sites", type=str, default=DEFAULT_PSP_REGULATORY_SITES)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--expected_folds",
        type=int,
        default=DEFAULT_EXPECTED_FOLDS,
        help="Number of FuncTransport folds averaged for ensemble test predictions.",
    )
    parser.add_argument(
        "--psp_match_mode",
        type=str,
        default="strict",
        choices=["strict", "canonical", "both"],
        help=(
            "How to match PSP ACC_ID to test INDEX. "
            "strict uses the exact ACC_ID. canonical removes isoform suffixes. "
            "both uses either index."
        ),
    )
    parser.add_argument(
        "--skip_psp_filter",
        action="store_true",
        help="Run the benchmark without excluding PSP regulatory negative sites.",
    )
    parser.add_argument(
        "--make_strict_plots",
        action="store_true",
        help="Legacy option retained for compatibility. Strict and clean plots are always saved.",
    )
    return parser.parse_args()


def set_publication_style():
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "DejaVu Sans",
        "font.size": FONT_SIZE_BASE,
        "axes.labelsize": FONT_SIZE_BASE,
        "axes.titlesize": FONT_SIZE_BASE,
        "xtick.labelsize": FONT_SIZE_TICK,
        "ytick.labelsize": FONT_SIZE_TICK,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "savefig.dpi": SAVE_DPI,
    })


def clean_axis(ax):
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


def _metric_group_centers(n_models):
    group_width = BAR_WIDTH * n_models
    return np.array([0.0, group_width + METRIC_GROUP_GAP])


def _axes_top_figure_y(ax):
    return ax.get_position().y1


def _apply_bar_plot_layout(fig, ax, n_models):
    group_width = BAR_WIDTH * n_models
    metric_x = _metric_group_centers(n_models)
    ax.set_xlim(
        metric_x[0] - group_width / 2 - X_PAD,
        metric_x[1] + group_width / 2 + X_PAD,
    )
    clean_axis(ax)
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


def save_benchmark_figure(fig, output_dir, file_prefix, suffix, legend=None):
    png_path = output_dir / f"{file_prefix}_{suffix}.png"
    pdf_path = output_dir / f"{file_prefix}_{suffix}.pdf"
    save_kwargs = {
        "facecolor": "white",
        "edgecolor": "none",
        "bbox_inches": "tight",
        "pad_inches": SAVE_PAD_INCHES,
    }
    if legend is not None:
        save_kwargs["bbox_extra_artists"] = [legend]
    fig.savefig(png_path, dpi=SAVE_DPI, **save_kwargs)
    fig.savefig(pdf_path, **save_kwargs)
    plt.close(fig)
    return png_path, pdf_path


def plot_prc_roc_metric_bars(
    metrics_df,
    output_dir,
    file_prefix,
    title_suffix,
):
    models = [model for model in MODEL_ORDER if model in set(metrics_df["model"])]
    if not models:
        print(f"[WARN] Skip bar plot for {file_prefix}: no models found in metrics table.")
        return None, None

    if metrics_df[["auprc", "auroc"]].isna().any().any():
        print(f"[WARN] Skip bar plot for {file_prefix}: missing AUROC/AUPRC values.")
        return None, None

    legend_labels = [MODEL_DISPLAY_NAMES[model] for model in models]
    colors = [MODEL_COLORS[model] for model in models]
    offsets = _metric_bar_offsets(len(models))

    auprc_means = np.array([
        float(metrics_df.loc[metrics_df["model"] == model, "auprc"].iloc[0])
        for model in models
    ])
    auroc_means = np.array([
        float(metrics_df.loc[metrics_df["model"] == model, "auroc"].iloc[0])
        for model in models
    ])

    metric_x = _metric_group_centers(len(models))

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

    for i, (legend_label, color) in enumerate(zip(legend_labels, colors)):
        auprc_x = metric_x[0] + offsets[i]
        auroc_x = metric_x[1] + offsets[i]

        ax.bar(
            auprc_x,
            auprc_means[i],
            width=BAR_WIDTH,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=legend_label,
        )
        ax.bar(
            auroc_x,
            auroc_means[i],
            width=BAR_WIDTH,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )

        ax.text(
            auprc_x,
            auprc_means[i] + 0.025,
            f"{auprc_means[i]:.3f}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_ANNOT,
        )
        ax.text(
            auroc_x,
            auroc_means[i] + 0.025,
            f"{auroc_means[i]:.3f}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_ANNOT,
        )

    ax.set_xticks(metric_x)
    ax.set_xticklabels(
        [METRIC_GROUPS["auprc"], METRIC_GROUPS["auroc"]],
        fontsize=FONT_SIZE_TICK,
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", labelpad=2)

    _apply_bar_plot_layout(fig, ax, len(models))

    return save_benchmark_figure(fig, output_dir, file_prefix, "prc_roc_bars", ax.get_legend())

def format_title_suffix(title_suffix):
    if title_suffix == "PSP regulatory negatives removed":
        return "clean PSP-filtered"
    return str(title_suffix)


def read_text_lines(path):
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            return handle.readlines()
    except UnicodeDecodeError:
        with path.open("r", encoding="latin1") as handle:
            return handle.readlines()


def find_psp_header_row(psp_path):
    lines = read_text_lines(psp_path)
    for i, line in enumerate(lines):
        if line.startswith("GENE\tPROTEIN\t"):
            return i
    raise ValueError(f"Could not find PSP table header in: {psp_path}")


def read_psp_table(psp_path):
    psp_path = Path(psp_path)
    if not psp_path.exists():
        raise FileNotFoundError(f"Missing PSP Regulatory_sites file: {psp_path}")

    header_row = find_psp_header_row(psp_path)
    try:
        psp_df = pd.read_csv(
            psp_path,
            sep="\t",
            skiprows=header_row,
            dtype=str,
            low_memory=False,
            encoding="utf-8",
        )
    except UnicodeDecodeError:
        psp_df = pd.read_csv(
            psp_path,
            sep="\t",
            skiprows=header_row,
            dtype=str,
            low_memory=False,
            encoding="latin1",
        )

    psp_df.columns = [str(col).strip() for col in psp_df.columns]
    unnamed_cols = [col for col in psp_df.columns if col.startswith("Unnamed") or col == ""]
    if unnamed_cols:
        psp_df = psp_df.drop(columns=unnamed_cols)

    required_cols = ["GENE", "PROTEIN", "ACC_ID", "ORGANISM", "MOD_RSD"]
    missing_cols = [col for col in required_cols if col not in psp_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required PSP columns: {missing_cols}")

    for col in psp_df.columns:
        psp_df[col] = psp_df[col].fillna("").astype(str).str.strip()

    return psp_df


def parse_phosphosite_mod(mod_rsd):
    match = re.match(r"^([STY])(\d+)-p$", str(mod_rsd).strip(), flags=re.IGNORECASE)
    if match is None:
        return None, None
    residue = match.group(1).upper()
    position = int(match.group(2))
    return residue, position


def canonical_acc_id(acc_id):
    return str(acc_id).strip().split("-")[0]


def aggregate_unique(values):
    cleaned = []
    seen = set()
    for value in values:
        value = str(value).strip()
        if value == "" or value.lower() == "nan":
            continue
        if value not in seen:
            cleaned.append(value)
            seen.add(value)
    return "; ".join(cleaned)


def load_psp_regulatory_phosphosites(psp_path, match_mode="strict"):
    psp_df = read_psp_table(psp_path)
    human_df = psp_df[psp_df["ORGANISM"].str.lower() == "human"].copy()

    parsed = human_df["MOD_RSD"].apply(parse_phosphosite_mod)
    human_df["psp_residue"] = parsed.apply(lambda x: x[0])
    human_df["psp_position"] = parsed.apply(lambda x: x[1])
    phospho_df = human_df[
        human_df["psp_residue"].notna() & human_df["psp_position"].notna()
    ].copy()
    phospho_df["psp_position"] = phospho_df["psp_position"].astype(int)

    rows = []
    for _, row in phospho_df.iterrows():
        acc_id = str(row["ACC_ID"]).strip()
        residue = str(row["psp_residue"]).strip()
        position = int(row["psp_position"])
        strict_index = f"{acc_id}_{residue}{position}"
        canonical_index = f"{canonical_acc_id(acc_id)}_{residue}{position}"

        if match_mode == "strict":
            index_entries = [(strict_index, "strict")]
        elif match_mode == "canonical":
            index_entries = [(canonical_index, "canonical")]
        else:
            index_entries = [(strict_index, "strict")]
            if canonical_index != strict_index:
                index_entries.append((canonical_index, "canonical"))

        for psp_index, match_source in index_entries:
            out = {
                "INDEX": psp_index,
                "psp_match_source": match_source,
                "psp_residue": residue,
                "psp_position": position,
            }
            for col in PSP_META_COLUMNS:
                if col in phospho_df.columns:
                    out[f"psp_{col.lower()}"] = row[col]
            rows.append(out)

    if len(rows) == 0:
        return pd.DataFrame(columns=["INDEX", "psp_match_source"])

    site_df = pd.DataFrame(rows)
    aggregation = {col: aggregate_unique for col in site_df.columns if col != "INDEX"}
    site_df = site_df.groupby("INDEX", as_index=False).agg(aggregation)
    return site_df


def load_fixed_test_indices(run_dir):
    test_samples_path = Path(run_dir) / "Functional_Transport" / "fixed_test_samples.csv"
    if not test_samples_path.exists():
        raise FileNotFoundError(f"Missing fixed test sample file: {test_samples_path}")

    test_df = pd.read_csv(test_samples_path)
    test_df["INDEX"] = test_df["INDEX"].astype(str)
    return test_df


def load_phosloc_ensemble_fixed_test_scores(run_dir, expected_folds=DEFAULT_EXPECTED_FOLDS):
    pred_path = Path(run_dir) / "Functional_Transport" / "fixed_test_predictions_by_fold.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing fixed test prediction file: {pred_path}")

    pred_df = pd.read_csv(pred_path)
    pred_df["INDEX"] = pred_df["INDEX"].astype(str)
    pred_df["fold"] = pd.to_numeric(pred_df["fold"], errors="coerce").astype(int)
    pred_df["pred_prob"] = pd.to_numeric(pred_df["pred_prob"], errors="coerce")
    pred_df["true_label"] = pd.to_numeric(pred_df["true_label"], errors="coerce").astype(int)

    folds = sorted(pred_df["fold"].unique())
    if len(folds) != expected_folds:
        raise ValueError(
            f"Expected {expected_folds} folds for ensemble, found {len(folds)}: {folds}"
        )

    grouped = pred_df.groupby("INDEX", as_index=False).agg(
        true_label=("true_label", "first"),
        phosloc_score=("pred_prob", "mean"),
        n_folds=("fold", "nunique"),
    )
    incomplete = grouped[grouped["n_folds"] != expected_folds]
    if not incomplete.empty:
        raise ValueError(
            f"{len(incomplete)} sites do not have predictions from all {expected_folds} folds."
        )

    site_df = grouped[["INDEX", "true_label", "phosloc_score"]].copy()
    return site_df, expected_folds


def load_external_scores(score_csv, score_col):
    score_df = pd.read_csv(score_csv)
    score_df["INDEX"] = score_df["INDEX"].astype(str)
    score_df[score_col] = pd.to_numeric(score_df[score_col], errors="coerce")
    return score_df[["INDEX", score_col]].drop_duplicates(subset=["INDEX"]).copy()


def build_benchmark_table(run_dir, str_csv, seq_csv, expected_folds):
    test_df = load_fixed_test_indices(run_dir)
    test_indices = set(test_df["INDEX"].astype(str))

    phosloc_df, n_folds = load_phosloc_ensemble_fixed_test_scores(
        run_dir,
        expected_folds=expected_folds,
    )
    print(
        f"[INFO] Using FuncTransport five-fold ensemble mean on fixed test "
        f"(n_folds={n_folds})"
    )

    phosloc_df = phosloc_df[phosloc_df["INDEX"].isin(test_indices)].copy()

    str_df = load_external_scores(str_csv, "funcphos_str_score_num")
    seq_df = load_external_scores(seq_csv, "funcphos_seq_score_num")

    benchmark_df = phosloc_df.merge(str_df, on="INDEX", how="left")
    benchmark_df = benchmark_df.merge(seq_df, on="INDEX", how="left")

    benchmark_df["true_label"] = pd.to_numeric(
        benchmark_df["true_label"], errors="coerce"
    ).astype(int)

    missing_str = benchmark_df["funcphos_str_score_num"].isna().sum()
    missing_seq = benchmark_df["funcphos_seq_score_num"].isna().sum()
    if missing_str > 0 or missing_seq > 0:
        print(
            f"[WARN] Missing external scores on fixed test: "
            f"STR={missing_str}, SEQ={missing_seq}"
        )

    complete_df = benchmark_df.dropna(subset=SCORE_COLS).copy()

    print(f"[INFO] Fixed test sites in run: {len(test_indices)}")
    print(f"[INFO] PhosLoc scored sites: {len(phosloc_df)}")
    print(f"[INFO] Complete benchmark sites: {len(complete_df)}")
    print(f"[INFO] Positive sites: {int(complete_df['true_label'].sum())}")
    print(f"[INFO] Negative sites: {int((complete_df['true_label'] == 0).sum())}")

    return benchmark_df, complete_df, n_folds


def annotate_with_psp(benchmark_df, psp_site_df):
    annotated_df = benchmark_df.merge(psp_site_df, on="INDEX", how="left")
    annotated_df["is_psp_regulatory_phosphosite"] = annotated_df[
        "psp_match_source"
    ].notna()
    annotated_df["exclude_from_clean_benchmark"] = (
        (annotated_df["true_label"].astype(int) == 0)
        & annotated_df["is_psp_regulatory_phosphosite"]
    )
    return annotated_df


def make_clean_benchmark(complete_df):
    clean_df = complete_df[~complete_df["exclude_from_clean_benchmark"]].copy()
    excluded_df = complete_df[complete_df["exclude_from_clean_benchmark"]].copy()
    return clean_df, excluded_df


def compute_model_metrics(y_true, y_score):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    metrics = {
        "n": len(y_true),
        "pos": int(y_true.sum()),
        "neg": int((y_true == 0).sum()),
        "prevalence": float(np.mean(y_true)) if len(y_true) > 0 else np.nan,
    }

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        metrics.update({"auroc": np.nan, "auprc": np.nan})
        return metrics

    metrics.update({
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
    })
    return metrics


def compute_all_metrics(benchmark_df, benchmark_mode):
    rows = []
    y_true = benchmark_df["true_label"].to_numpy(dtype=int)

    for model in MODEL_ORDER:
        score_col = MODEL_SCORE_COLS[model]
        metrics = compute_model_metrics(y_true, benchmark_df[score_col].to_numpy())
        rows.append({"benchmark_mode": benchmark_mode, "model": model, **metrics})

    return pd.DataFrame(rows)


def compute_curve_tables(benchmark_df, benchmark_mode):
    y_true = benchmark_df["true_label"].to_numpy(dtype=int)
    curve_rows = []

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return pd.DataFrame(columns=[
            "benchmark_mode",
            "model",
            "curve_type",
            "x",
            "y",
            "threshold",
        ])

    for model in MODEL_ORDER:
        score_col = MODEL_SCORE_COLS[model]
        y_score = benchmark_df[score_col].to_numpy(dtype=float)

        fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)

        for i in range(len(fpr)):
            curve_rows.append({
                "benchmark_mode": benchmark_mode,
                "model": model,
                "curve_type": "roc",
                "x": float(fpr[i]),
                "y": float(tpr[i]),
                "threshold": float(roc_thresholds[i]) if i < len(roc_thresholds) else np.nan,
            })

        for i in range(len(recall)):
            curve_rows.append({
                "benchmark_mode": benchmark_mode,
                "model": model,
                "curve_type": "prc",
                "x": float(recall[i]),
                "y": float(precision[i]),
                "threshold": float(pr_thresholds[i]) if i < len(pr_thresholds) else np.nan,
            })

    return pd.DataFrame(curve_rows)


def add_phosloc_metadata(metrics_df, n_folds):
    metrics_df.insert(0, "phosloc_prediction_mode", "five_fold_ensemble_mean")
    metrics_df.insert(1, "phosloc_n_folds", n_folds)
    return metrics_df


def make_filter_summary(full_df, strict_df, clean_df, excluded_df, psp_site_df, n_folds):
    rows = [{
        "phosloc_prediction_mode": "five_fold_ensemble_mean",
        "phosloc_n_folds": n_folds,
        "psp_regulatory_phosphosite_index_count": len(psp_site_df),
        "all_scored_sites_before_complete_filter": len(full_df),
        "complete_strict_sites": len(strict_df),
        "complete_strict_positive_sites": int(strict_df["true_label"].sum()) if len(strict_df) else 0,
        "complete_strict_negative_sites": int((strict_df["true_label"] == 0).sum()) if len(strict_df) else 0,
        "psp_regulatory_sites_in_complete": int(strict_df["is_psp_regulatory_phosphosite"].sum()) if len(strict_df) else 0,
        "psp_regulatory_positive_sites_in_complete": int(
            ((strict_df["true_label"] == 1) & strict_df["is_psp_regulatory_phosphosite"]).sum()
        ) if len(strict_df) else 0,
        "excluded_psp_regulatory_negative_sites": len(excluded_df),
        "complete_clean_sites": len(clean_df),
        "complete_clean_positive_sites": int(clean_df["true_label"].sum()) if len(clean_df) else 0,
        "complete_clean_negative_sites": int((clean_df["true_label"] == 0).sum()) if len(clean_df) else 0,
        "complete_clean_prevalence": float(clean_df["true_label"].mean()) if len(clean_df) else np.nan,
    }]
    return pd.DataFrame(rows)


def print_metrics_summary(metrics_df, title):
    print(f"[INFO] {title}")
    print(metrics_df.to_string(index=False))


def main():
    args = parse_args()
    set_publication_style()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_df, complete_df, n_folds = build_benchmark_table(
        args.run_dir,
        args.str_csv,
        args.seq_csv,
        args.expected_folds,
    )

    if args.skip_psp_filter:
        psp_site_df = pd.DataFrame(columns=["INDEX", "psp_match_source"])
        full_annotated_df = full_df.copy()
        complete_annotated_df = complete_df.copy()
        full_annotated_df["is_psp_regulatory_phosphosite"] = False
        full_annotated_df["exclude_from_clean_benchmark"] = False
        complete_annotated_df["is_psp_regulatory_phosphosite"] = False
        complete_annotated_df["exclude_from_clean_benchmark"] = False
    else:
        psp_site_df = load_psp_regulatory_phosphosites(
            args.psp_regulatory_sites,
            match_mode=args.psp_match_mode,
        )
        full_annotated_df = annotate_with_psp(full_df, psp_site_df)
        complete_annotated_df = annotate_with_psp(complete_df, psp_site_df)

    strict_df = complete_annotated_df.copy()
    clean_df, excluded_df = make_clean_benchmark(strict_df)

    strict_metrics_df = add_phosloc_metadata(
        compute_all_metrics(strict_df, benchmark_mode="strict_all_non_transport_as_negative"),
        n_folds,
    )
    clean_metrics_df = add_phosloc_metadata(
        compute_all_metrics(clean_df, benchmark_mode="clean_psp_regulatory_negative_removed"),
        n_folds,
    )
    metrics_all_modes_df = pd.concat([strict_metrics_df, clean_metrics_df], ignore_index=True)

    strict_curve_df = compute_curve_tables(
        strict_df,
        benchmark_mode="strict_all_non_transport_as_negative",
    )
    clean_curve_df = compute_curve_tables(
        clean_df,
        benchmark_mode="clean_psp_regulatory_negative_removed",
    )
    curves_all_modes_df = pd.concat([strict_curve_df, clean_curve_df], ignore_index=True)

    summary_df = make_filter_summary(
        full_annotated_df,
        strict_df,
        clean_df,
        excluded_df,
        psp_site_df,
        n_folds,
    )

    psp_site_csv = output_dir / "psp_human_regulatory_phosphosite_indices.csv"
    full_csv = output_dir / "benchmark_fixed_test_scores_all_with_psp_annotation.csv"
    strict_csv = output_dir / "benchmark_fixed_test_scores_complete_strict.csv"
    clean_csv = output_dir / "benchmark_fixed_test_scores_complete_clean_psp_filtered.csv"
    excluded_csv = output_dir / "benchmark_fixed_test_excluded_psp_regulatory_negative_sites.csv"
    strict_metrics_csv = output_dir / "benchmark_fixed_test_metrics_strict.csv"
    clean_metrics_csv = output_dir / "benchmark_fixed_test_metrics_clean_psp_filtered.csv"
    metrics_all_modes_csv = output_dir / "benchmark_fixed_test_metrics_all_modes.csv"
    curves_all_modes_csv = output_dir / "benchmark_fixed_test_curve_points_all_modes.csv"
    summary_csv = output_dir / "benchmark_fixed_test_psp_filter_summary.csv"

    psp_site_df.to_csv(psp_site_csv, index=False)
    full_annotated_df.to_csv(full_csv, index=False)
    strict_df.to_csv(strict_csv, index=False)
    clean_df.to_csv(clean_csv, index=False)
    excluded_df.to_csv(excluded_csv, index=False)
    strict_metrics_df.to_csv(strict_metrics_csv, index=False)
    clean_metrics_df.to_csv(clean_metrics_csv, index=False)
    metrics_all_modes_df.to_csv(metrics_all_modes_csv, index=False)
    curves_all_modes_df.to_csv(curves_all_modes_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    clean_bars_png, clean_bars_pdf = plot_prc_roc_metric_bars(
        clean_metrics_df,
        output_dir,
        file_prefix="benchmark_fixed_test_clean_psp_filtered",
        title_suffix="PSP regulatory negatives removed",
    )

    strict_bars_png, strict_bars_pdf = plot_prc_roc_metric_bars(
        strict_metrics_df,
        output_dir,
        file_prefix="benchmark_fixed_test_strict",
        title_suffix="strict",
    )

    print(
        f"[DONE] FuncTransport prediction mode: five-fold ensemble mean "
        f"(n_folds={n_folds})"
    )
    print(f"[DONE] Run dir: {args.run_dir}")
    print(f"[DONE] PSP match mode: {args.psp_match_mode}")
    print(f"[DONE] PSP regulatory phosphosite index table: {psp_site_csv}")
    print(f"[DONE] Scores with PSP annotation: {full_csv}")
    print(f"[DONE] Strict complete scores: {strict_csv}")
    print(f"[DONE] Clean complete scores: {clean_csv}")
    print(f"[DONE] Excluded PSP regulatory negative sites: {excluded_csv}")
    print(f"[DONE] Strict metrics: {strict_metrics_csv}")
    print(f"[DONE] Clean metrics: {clean_metrics_csv}")
    print(f"[DONE] Metrics, all modes: {metrics_all_modes_csv}")
    print(f"[DONE] Curve points, all modes: {curves_all_modes_csv}")
    print(f"[DONE] PSP filter summary: {summary_csv}")
    print(f"[DONE] Clean AUPRC/AUROC bar plot: {clean_bars_png}")
    print(f"[DONE] Clean AUPRC/AUROC bar PDF: {clean_bars_pdf}")
    print(f"[DONE] Strict AUPRC/AUROC bar plot: {strict_bars_png}")
    print(f"[DONE] Strict AUPRC/AUROC bar PDF: {strict_bars_pdf}")

    print("[INFO] PSP filtering summary:")
    print(summary_df.to_string(index=False))
    print_metrics_summary(strict_metrics_df, "Strict metrics summary:")
    print_metrics_summary(clean_metrics_df, "Clean PSP-filtered metrics summary:")


if __name__ == "__main__":
    main()
