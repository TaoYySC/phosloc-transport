# Panels: Figure 3(c)


# Monorepo path constants (monorepo relative paths)
import argparse
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
import re
import numpy as np
import pandas as pd
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["text.usetex"] = False
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]

FONT_SIZE_SHIFT = 2
matplotlib.rcParams["font.size"] = 10 + FONT_SIZE_SHIFT
matplotlib.rcParams["axes.labelsize"] = 12 + FONT_SIZE_SHIFT
matplotlib.rcParams["xtick.labelsize"] = 10 + FONT_SIZE_SHIFT
matplotlib.rcParams["ytick.labelsize"] = 10 + FONT_SIZE_SHIFT
matplotlib.rcParams["legend.fontsize"] = 10 + FONT_SIZE_SHIFT

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DEFAULT_FUNCTIONAL_CSV = (
    FUNCTIONAL_PRECOMPUTED
    / "2_1_functional_classifier_results"
    / "predictions"
    / "esm_window_site_pdb_5_folds_ensemble_predictions.csv"
)
DEFAULT_IMPORT_EXPORT_CSV = (
    DATA_PRECOMPUTED
    / "1_transport_classifier_results"
    / "esm_window_only_import_pos_predictions"
    / "tf_all_phos_site_predictions_per_fold.csv"
)
DEFAULT_KNOWN_POSITIVE_CSV = (
    FUNCTIONAL_ROOT / "data" / "dataset_phos_site" / "TF_positive_phos_site_0608.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "1_transport_classifier_results" / "joint_score"
DEFAULT_MERGE_MODE = "matched_only"
DEFAULT_FUNCTIONAL_SCORE_THRESHOLD = 0.6

# Stability filter on 5-fold direction scores (same logic as 1_6).
MIN_VOTE = 4

IMPORT_COLOR = "#9a3f3f"
EXPORT_COLOR = "#4f7d95"
UNKNOWN_COLOR = "#bdbdbd"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Combine functional transport probabilities with direction predictions."
    )
    parser.add_argument("--functional_csv", type=Path, default=DEFAULT_FUNCTIONAL_CSV)
    parser.add_argument("--import_export_csv", type=Path, default=DEFAULT_IMPORT_EXPORT_CSV)
    parser.add_argument("--known_positive_csv", type=Path, default=DEFAULT_KNOWN_POSITIVE_CSV)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--merge_mode",
        choices=["matched_only", "keep_all_fill_zero"],
        default=DEFAULT_MERGE_MODE,
    )
    parser.add_argument(
        "--functional_score_threshold",
        type=float,
        default=DEFAULT_FUNCTIONAL_SCORE_THRESHOLD,
    )
    parser.add_argument("--min_vote", type=int, default=MIN_VOTE)
    return parser.parse_args(argv)


def format_threshold_tag(value):
    text = f"{float(value):g}".replace("-", "m").replace(".", "p")
    return f"gt{text}"


def ensure_index(df, acc_col="ACC_ID", site_col="POSITION"):
    df = df.copy()

    if "INDEX" not in df.columns:
        if acc_col not in df.columns or site_col not in df.columns:
            raise ValueError(f"Missing columns for INDEX construction: {acc_col}, {site_col}")
        df["INDEX"] = df[acc_col].astype(str) + "_" + df[site_col].astype(int).astype(str)

    df["INDEX"] = df["INDEX"].astype(str)
    return df


def check_unique_index(df, df_name):
    dup_mask = df["INDEX"].duplicated(keep=False)
    if dup_mask.any():
        examples = df.loc[dup_mask, "INDEX"].head(10).tolist()
        raise ValueError(f"{df_name} has duplicated INDEX values. Examples: {examples}")


def get_seed_fold_prob_cols(df, prefix):
    pattern = re.compile(rf"^{re.escape(prefix)}_seed(\d+)_fold(\d+)$")
    cols = [col for col in df.columns if pattern.match(col)]

    def sort_key(col):
        match = pattern.match(col)
        return int(match.group(1)), int(match.group(2))

    return sorted(cols, key=sort_key)


def get_seed_fold_key(col, prefix):
    pattern = re.compile(rf"^{re.escape(prefix)}_seed(\d+)_fold(\d+)$")
    match = pattern.match(col)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def get_functional_prob_fold_cols(df, prefix="prob_fold"):
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    cols = [col for col in df.columns if pattern.match(col)]

    def sort_key(col):
        match = pattern.match(col)
        return int(match.group(1))

    return sorted(cols, key=sort_key)


def get_merged_functional_prob_fold_cols(df):
    pattern = re.compile(r"^functional_prob_fold_(\d+)$")
    cols = [col for col in df.columns if pattern.match(col)]

    def sort_key(col):
        match = pattern.match(col)
        return int(match.group(1))

    return sorted(cols, key=sort_key)


def normalize_known_direction(value):
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()
    text = text.replace("-", "_").replace(" ", "_")

    import_specific_terms = [
        "promote_import",
        "promote_nuclear_import",
        "inhibit_export",
        "inhibit_nuclear_export",
        "nuclear_import",
    ]

    export_specific_terms = [
        "promote_export",
        "promote_nuclear_export",
        "inhibit_import",
        "inhibit_nuclear_import",
        "nuclear_export",
    ]

    for term in import_specific_terms:
        if term in text:
            return "Known Import"

    for term in export_specific_terms:
        if term in text:
            return "Known Export"

    if text == "import" or text.endswith("_import"):
        return "Known Import"

    if text == "export" or text.endswith("_export"):
        return "Known Export"

    return np.nan


def add_known_direction_to_known_df(known_df):
    known_df = known_df.copy()

    candidate_cols = [
        "Transport_Direction",
        "LABEL",
        "label",
        "direction",
        "Direction",
    ]

    source_col = None
    for col in candidate_cols:
        if col in known_df.columns:
            source_col = col
            break

    if source_col is None:
        known_df["known_direction_norm"] = np.nan
    else:
        known_df["known_direction_norm"] = known_df[source_col].apply(normalize_known_direction)

    return known_df


def calculate_per_fold_direction_scores(merged_df, common_seed_fold_keys, functional_prob_col="mean_prob_functional"):
    """Compute direction scores per IE fold, weighted by functional ensemble mean probability.

    Each per-fold direction score is:
        direction_score_fold = (2 * prob_import_fold - 1) * mean_prob_functional

    where mean_prob_functional is the average probability from the five-fold
    functional classifier ensemble (ESM Window+Site+PDB).
    """
    merged_df = merged_df.copy()

    if functional_prob_col not in merged_df.columns:
        raise ValueError(f"Missing functional probability column: {functional_prob_col}")

    if len(common_seed_fold_keys) == 0:
        raise ValueError(
            "No import/export seed-fold pairs were found. "
            "Expected columns like prob_import_seed42_fold1."
        )

    direction_raw_fold_cols = []
    direction_score_fold_cols = []

    for seed, fold in common_seed_fold_keys:
        import_col = f"prob_import_seed{seed}_fold{fold}"
        if import_col not in merged_df.columns:
            raise ValueError(f"Missing import probability column: {import_col}")

        direction_raw_col = f"direction_raw_seed{seed}_fold{fold}"
        direction_score_col = f"direction_score_seed{seed}_fold{fold}"

        merged_df[direction_raw_col] = 2.0 * merged_df[import_col] - 1.0
        merged_df[direction_score_col] = (
            merged_df[direction_raw_col] * merged_df[functional_prob_col]
        )

        direction_raw_fold_cols.append(direction_raw_col)
        direction_score_fold_cols.append(direction_score_col)

    merged_df["direction_raw"] = merged_df[direction_raw_fold_cols].mean(axis=1)
    merged_df["direction_raw_std"] = merged_df[direction_raw_fold_cols].std(axis=1)

    merged_df["direction_score"] = merged_df[direction_score_fold_cols].mean(axis=1)
    merged_df["direction_score_std"] = merged_df[direction_score_fold_cols].std(axis=1)
    merged_df["direction_abs_score"] = merged_df["direction_score"].abs()

    print(
        f"[INFO] Functional ensemble probability column: {functional_prob_col}. "
        f"Per-fold direction scores computed from {len(common_seed_fold_keys)} IE folds."
    )

    return merged_df, direction_raw_fold_cols, direction_score_fold_cols


def compute_known_positive_direction_thresholds(df, score_col="direction_score"):
    """Derive import/export mu thresholds from known positive direction scores.

    Import threshold = minimum direction score among known import sites.
    Export threshold = maximum direction score among known export sites.
    """
    known_import_scores = df.loc[
        df["known_direction_norm"] == "Known Import", score_col
    ].dropna()
    known_export_scores = df.loc[
        df["known_direction_norm"] == "Known Export", score_col
    ].dropna()

    if known_import_scores.empty:
        raise ValueError("No known import sites found to compute import threshold.")
    if known_export_scores.empty:
        raise ValueError("No known export sites found to compute export threshold.")

    import_mu_threshold = float(known_import_scores.min())
    export_mu_threshold = float(known_export_scores.max())

    return import_mu_threshold, export_mu_threshold


def apply_stability_filter(
    df,
    direction_score_fold_cols,
    import_mu_threshold,
    export_mu_threshold,
    min_vote=MIN_VOTE,
):
    """Filter sites by 5-fold direction score consistency (vote-based)."""
    df = df.copy()

    score_mat = df[direction_score_fold_cols]
    df["direction_score_mu"] = score_mat.mean(axis=1)
    df["direction_score_sigma"] = score_mat.std(axis=1)

    global_median = df["direction_score_mu"].median()
    df["direction_score_global_median"] = global_median

    df["vote_import"] = (score_mat > 0).sum(axis=1)
    df["vote_export"] = (score_mat < 0).sum(axis=1)

    df["stable_predicted_import"] = (
        (df["direction_score_mu"] >= import_mu_threshold)
        & (df["vote_import"] >= min_vote)
        & (df["direction_score_mu"] > global_median)
    )

    df["stable_predicted_export"] = (
        (df["direction_score_mu"] < export_mu_threshold)
        & (df["vote_export"] >= min_vote)
        & (df["direction_score_mu"] < global_median)
    )

    overlap_mask = df["stable_predicted_import"] & df["stable_predicted_export"]
    if overlap_mask.any():
        overlap_examples = df.loc[overlap_mask, "INDEX"].head(10).tolist()
        raise ValueError(
            f"Some sites satisfy both import and export stability rules. Examples: {overlap_examples}"
        )

    df["stable_predicted_direction"] = np.select(
        [
            df["stable_predicted_import"],
            df["stable_predicted_export"],
        ],
        [
            "Predicted Import",
            "Predicted Export",
        ],
        default="Unselected",
    )

    return df, global_median


def build_stability_rule_summary(
    df,
    global_median,
    import_mu_threshold,
    export_mu_threshold,
):
    rows = []

    for vote_threshold in [3, 4, 5]:
        import_mask = (
            (df["direction_score_mu"] >= import_mu_threshold)
            & (df["vote_import"] >= vote_threshold)
            & (df["direction_score_mu"] > global_median)
        )

        export_mask = (
            (df["direction_score_mu"] < export_mu_threshold)
            & (df["vote_export"] >= vote_threshold)
            & (df["direction_score_mu"] < global_median)
        )

        import_count = int(import_mask.sum())
        export_count = int(export_mask.sum())
        combined_count = int((import_mask | export_mask).sum())

        if export_count > 0:
            import_export_ratio = import_count / export_count
        else:
            import_export_ratio = np.nan

        rows.append(
            {
                "vote_min": vote_threshold,
                "import_mu_threshold": import_mu_threshold,
                "export_mu_threshold": export_mu_threshold,
                "global_median": global_median,
                "import_count": import_count,
                "export_count": export_count,
                "combined_count": combined_count,
                "import_export_ratio": import_export_ratio,
            }
        )

    return pd.DataFrame(rows)


def plot_direction_score_signed_scatter_known_direction(
    df,
    direction_score_cols,
    output_path,
    import_mu_threshold=None,
    export_mu_threshold=None,
):
    plot_df = df.copy()
    plot_df = plot_df.sort_values("direction_score", ascending=True).reset_index(drop=True)
    plot_df["rank_signed_direction_score"] = np.arange(1, len(plot_df) + 1)

    known_import_mask = plot_df["known_direction_norm"] == "Known Import"
    known_export_mask = plot_df["known_direction_norm"] == "Known Export"
    unknown_mask = ~(known_import_mask | known_export_mask)

    fig, ax = plt.subplots(figsize=(9.2, 5.6))

    n_seed_scores = len(direction_score_cols)
    x_offsets = np.linspace(-0.20, 0.20, n_seed_scores) if n_seed_scores > 1 else np.array([0.0])

    for i, col in enumerate(direction_score_cols):
        ax.scatter(
            plot_df.loc[known_export_mask, "rank_signed_direction_score"] + x_offsets[i],
            plot_df.loc[known_export_mask, col],
            s=14,
            alpha=0.28,
            color=EXPORT_COLOR,
            edgecolors="none",
            label="Known export per fold" if i == 0 else None,
            zorder=2
        )

        ax.scatter(
            plot_df.loc[known_import_mask, "rank_signed_direction_score"] + x_offsets[i],
            plot_df.loc[known_import_mask, col],
            s=14,
            alpha=0.28,
            color=IMPORT_COLOR,
            edgecolors="none",
            label="Known import per fold" if i == 0 else None,
            zorder=2
        )

        ax.scatter(
            plot_df.loc[unknown_mask, "rank_signed_direction_score"] + x_offsets[i],
            plot_df.loc[unknown_mask, col],
            s=14,
            alpha=0.28,
            color=UNKNOWN_COLOR,
            edgecolors="none",
            label="Other site per fold" if i == 0 else None,
            zorder=2
        )

    ax.scatter(
        plot_df.loc[unknown_mask, "rank_signed_direction_score"],
        plot_df.loc[unknown_mask, "direction_score"],
        s=24,
        alpha=0.80,
        color=UNKNOWN_COLOR,
        edgecolors="none",
        label="Other site mean",
        zorder=3
    )

    ax.scatter(
        plot_df.loc[known_export_mask, "rank_signed_direction_score"],
        plot_df.loc[known_export_mask, "direction_score"],
        s=38,
        alpha=0.92,
        color=EXPORT_COLOR,
        edgecolors="none",
        label="Known export mean",
        zorder=4
    )

    ax.scatter(
        plot_df.loc[known_import_mask, "rank_signed_direction_score"],
        plot_df.loc[known_import_mask, "direction_score"],
        s=38,
        alpha=0.92,
        color=IMPORT_COLOR,
        edgecolors="none",
        label="Known import mean",
        zorder=4
    )

    ax.axhline(0, color="black", linewidth=1.0)

    if import_mu_threshold is not None:
        ax.axhline(
            import_mu_threshold,
            color=IMPORT_COLOR,
            linewidth=1.0,
            linestyle="--",
            alpha=0.85,
        )

    if export_mu_threshold is not None:
        ax.axhline(
            export_mu_threshold,
            color=EXPORT_COLOR,
            linewidth=1.0,
            linestyle="--",
            alpha=0.85,
        )

    legend_handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            label="Known export mean",
            markerfacecolor=EXPORT_COLOR,
            markeredgecolor="none",
            markersize=7
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            label="Known import mean",
            markerfacecolor=IMPORT_COLOR,
            markeredgecolor="none",
            markersize=7
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            label="Other site mean",
            markerfacecolor=UNKNOWN_COLOR,
            markeredgecolor="none",
            markersize=6
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            label="Other site per fold",
            markerfacecolor=UNKNOWN_COLOR,
            markeredgecolor="none",
            alpha=0.28,
            markersize=4
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            label="Known export per fold",
            markerfacecolor=EXPORT_COLOR,
            markeredgecolor="none",
            alpha=0.28,
            markersize=4
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            label="Known import per fold",
            markerfacecolor=IMPORT_COLOR,
            markeredgecolor="none",
            alpha=0.28,
            markersize=4
        ),
    ]

    if import_mu_threshold is not None:
        legend_handles.append(
            Line2D(
                [0], [0],
                color=IMPORT_COLOR,
                lw=1.0,
                linestyle="--",
                label=f"Import threshold ({import_mu_threshold:.3f})",
            )
        )

    if export_mu_threshold is not None:
        legend_handles.append(
            Line2D(
                [0], [0],
                color=EXPORT_COLOR,
                lw=1.0,
                linestyle="--",
                label=f"Export threshold ({export_mu_threshold:.3f})",
            )
        )

    ax.set_xlabel("Sites ranked by signed direction score", fontsize=12 + FONT_SIZE_SHIFT)
    ax.set_ylabel("Direction score", fontsize=12 + FONT_SIZE_SHIFT)
    ax.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=10 + FONT_SIZE_SHIFT,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
    )

    fig.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    pdf_output_path = os.path.splitext(output_path)[0] + ".pdf"
    fig.savefig(pdf_output_path, bbox_inches="tight")

    plt.close(fig)

    print(f"[DONE] Saved plot: {output_path}")
    print(f"[DONE] Saved plot: {pdf_output_path}")



def main(argv=None):
    args = parse_args(argv)
    functional_csv = str(args.functional_csv)
    import_export_csv = str(args.import_export_csv)
    known_positive_csv = str(args.known_positive_csv)
    output_dir = str(args.output_dir)
    plot_dir = os.path.join(output_dir, "distribution_plots")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    threshold_tag = format_threshold_tag(args.functional_score_threshold)
    vote_tag = f"vote{args.min_vote}"
    output_csv = os.path.join(output_dir, "tf_all_phos_site_joint_direction_score.csv")
    stable_import_output_csv = os.path.join(output_dir, f"predicted_import_stable_{threshold_tag}_{vote_tag}.csv")
    stable_export_output_csv = os.path.join(output_dir, f"predicted_export_stable_{threshold_tag}_{vote_tag}.csv")
    stable_combined_output_csv = os.path.join(output_dir, f"predicted_import_export_stable_{threshold_tag}_{vote_tag}.csv")
    stable_rule_summary_csv = os.path.join(output_dir, f"stable_site_rule_count_summary_{threshold_tag}_{vote_tag}.csv")
    missing_functional_csv = os.path.join(output_dir, "sites_missing_functional_probability.csv")
    scatter_output_png = os.path.join(plot_dir, "direction_score_scatter_known_sites.png")

    merge_mode = args.merge_mode
    functional_score_threshold = float(args.functional_score_threshold)
    min_vote = int(args.min_vote)

    functional_df = pd.read_csv(functional_csv)
    import_export_df = pd.read_csv(import_export_csv)
    known_df = pd.read_csv(known_positive_csv)

    functional_df = ensure_index(functional_df)
    import_export_df = ensure_index(import_export_df)
    known_df = ensure_index(known_df)

    known_df = add_known_direction_to_known_df(known_df)

    check_unique_index(functional_df, "functional_csv")
    check_unique_index(import_export_df, "import_export_csv")

    if "mean_prob" not in functional_df.columns:
        raise ValueError("Missing column in functional file: mean_prob")

    import_prob_cols = get_seed_fold_prob_cols(import_export_df, "prob_import")
    export_prob_cols = get_seed_fold_prob_cols(import_export_df, "prob_export")

    if len(import_prob_cols) == 0:
        raise ValueError("No import probability columns were found. Expected columns like prob_import_seed42_fold1.")

    if len(export_prob_cols) == 0:
        raise ValueError("No export probability columns were found. Expected columns like prob_export_seed42_fold1.")

    for col in import_prob_cols + export_prob_cols:
        import_export_df[col] = pd.to_numeric(import_export_df[col], errors="coerce")

    if import_export_df[import_prob_cols + export_prob_cols].isna().any().any():
        na_summary = import_export_df[import_prob_cols + export_prob_cols].isna().sum()
        na_summary = na_summary[na_summary > 0].to_dict()
        raise ValueError(f"NaN values found in probability columns after numeric conversion: {na_summary}")

    functional_prob_cols_raw = get_functional_prob_fold_cols(functional_df, prefix="prob_fold")

    if len(functional_prob_cols_raw) == 0:
        raise ValueError(
            "No functional fold probability columns were found. Expected prob_fold_1 to prob_fold_5."
        )

    for col in functional_prob_cols_raw:
        functional_df[col] = pd.to_numeric(functional_df[col], errors="coerce")

    if functional_df[functional_prob_cols_raw].isna().any().any():
        na_summary = functional_df[functional_prob_cols_raw].isna().sum()
        na_summary = na_summary[na_summary > 0].to_dict()
        raise ValueError(f"NaN values found in functional probability columns after numeric conversion: {na_summary}")

    functional_df["mean_prob"] = pd.to_numeric(functional_df["mean_prob"], errors="coerce")
    functional_df["std_prob"] = pd.to_numeric(functional_df["std_prob"], errors="coerce")
    functional_mean_from_folds = functional_df[functional_prob_cols_raw].mean(axis=1)
    mean_prob_diff = (functional_df["mean_prob"] - functional_mean_from_folds).abs()
    if mean_prob_diff.max() > 1e-6:
        print(
            "[WARN] mean_prob differs from the average of prob_fold_* columns. "
            "Using mean_prob as the functional ensemble score."
        )

    import_export_df["mean_prob_import"] = import_export_df[import_prob_cols].mean(axis=1)
    import_export_df["std_prob_import"] = import_export_df[import_prob_cols].std(axis=1)

    import_export_df["mean_prob_export"] = import_export_df[export_prob_cols].mean(axis=1)
    import_export_df["std_prob_export"] = import_export_df[export_prob_cols].std(axis=1)

    import_col_map = {
        get_seed_fold_key(col, "prob_import"): col
        for col in import_prob_cols
    }

    export_col_map = {
        get_seed_fold_key(col, "prob_export"): col
        for col in export_prob_cols
    }

    common_seed_fold_keys = sorted(set(import_col_map.keys()) & set(export_col_map.keys()))

    if len(common_seed_fold_keys) == 0:
        raise ValueError("No matched import and export seed fold probability columns were found.")

    functional_keep_cols = [
        "INDEX",
        "ACC_ID",
        "MOD_RSD",
        "POSITION",
        "FULL_SEQUENCE",
        "LABEL",
        "prob_fold_1",
        "artifact_fold_1",
        "seed_fold_1",
        "prob_fold_2",
        "artifact_fold_2",
        "seed_fold_2",
        "prob_fold_3",
        "artifact_fold_3",
        "seed_fold_3",
        "prob_fold_4",
        "artifact_fold_4",
        "seed_fold_4",
        "prob_fold_5",
        "artifact_fold_5",
        "seed_fold_5",
        "mean_prob",
        "std_prob",
        "final_threshold",
        "pred_label",
    ]

    functional_keep_cols = [
        col for col in functional_keep_cols
        if col in functional_df.columns
    ]

    functional_keep = functional_df[functional_keep_cols].copy()

    functional_keep = functional_keep.rename(
        columns={
            "ACC_ID": "functional_ACC_ID",
            "MOD_RSD": "functional_MOD_RSD",
            "POSITION": "functional_POSITION",
            "FULL_SEQUENCE": "functional_FULL_SEQUENCE",
            "LABEL": "functional_LABEL",
            "prob_fold_1": "functional_prob_fold_1",
            "artifact_fold_1": "functional_artifact_fold_1",
            "seed_fold_1": "functional_seed_fold_1",
            "prob_fold_2": "functional_prob_fold_2",
            "artifact_fold_2": "functional_artifact_fold_2",
            "seed_fold_2": "functional_seed_fold_2",
            "prob_fold_3": "functional_prob_fold_3",
            "artifact_fold_3": "functional_artifact_fold_3",
            "seed_fold_3": "functional_seed_fold_3",
            "prob_fold_4": "functional_prob_fold_4",
            "artifact_fold_4": "functional_artifact_fold_4",
            "seed_fold_4": "functional_seed_fold_4",
            "prob_fold_5": "functional_prob_fold_5",
            "artifact_fold_5": "functional_artifact_fold_5",
            "seed_fold_5": "functional_seed_fold_5",
            "mean_prob": "mean_prob_functional",
            "std_prob": "std_prob_functional",
            "final_threshold": "functional_final_threshold",
            "pred_label": "functional_pred_label",
        }
    )

    merged_df = import_export_df.merge(
        functional_keep,
        on="INDEX",
        how="left",
        validate="one_to_one"
    )

    missing_functional_mask = merged_df["mean_prob_functional"].isna()
    missing_functional_n = int(missing_functional_mask.sum())

    if missing_functional_n > 0:
        missing_df = merged_df.loc[missing_functional_mask].copy()
        missing_df.to_csv(missing_functional_csv, index=False)

        print(f"[WARN] {missing_functional_n} sites cannot match functional probabilities.")
        print(f"[WARN] Saved unmatched sites to: {missing_functional_csv}")
        print("[WARN] Examples:")
        print(missing_df["INDEX"].head(10).tolist())

        if merge_mode == "matched_only":
            merged_df = merged_df.loc[~missing_functional_mask].copy()
            print(f"[INFO] MERGE_MODE=matched_only. Remaining sites: {len(merged_df)}")

        elif merge_mode == "keep_all_fill_zero":
            merged_df["mean_prob_functional"] = merged_df["mean_prob_functional"].fillna(0.0)
            merged_df["std_prob_functional"] = merged_df["std_prob_functional"].fillna(0.0)
            merged_df["functional_pred_label"] = merged_df["functional_pred_label"].fillna(0)
            merged_df["functional_final_threshold"] = merged_df["functional_final_threshold"].fillna(np.nan)

            for col in get_merged_functional_prob_fold_cols(merged_df):
                merged_df[col] = merged_df[col].fillna(0.0)

            print(f"[INFO] MERGE_MODE=keep_all_fill_zero. Kept all sites: {len(merged_df)}")

        else:
            raise ValueError(f"Unknown merge_mode: {merge_mode}")

    merged_df["functional_selected"] = (
        pd.to_numeric(merged_df["mean_prob_functional"], errors="coerce") >= functional_score_threshold
    )

    before_functional_filter_n = len(merged_df)
    merged_df = merged_df.loc[merged_df["functional_selected"]].copy()
    after_functional_filter_n = len(merged_df)

    print(
        f"[INFO] Functional filter: mean_prob_functional >= {functional_score_threshold}. "
        f"Before={before_functional_filter_n}, after={after_functional_filter_n}, "
        f"removed={before_functional_filter_n - after_functional_filter_n}"
    )

    known_cols = ["INDEX", "known_direction_norm"]

    for col in [
        "ACC_ID",
        "POSITION",
        "MOD_RSD",
        "RESIDUE",
        "LABEL",
        "Transport_Direction",
        "Gene",
        "GENE",
        "Gene Symbol",
        "PMID",
        "SOURCE",
    ]:
        if col in known_df.columns and col not in known_cols:
            known_cols.append(col)

    known_anno = known_df[known_cols].drop_duplicates(subset=["INDEX"]).copy()

    known_anno = known_anno.rename(
        columns={
            col: f"known_{col}"
            for col in known_anno.columns
            if col not in ["INDEX", "known_direction_norm"]
        }
    )

    merged_df = merged_df.merge(
        known_anno,
        on="INDEX",
        how="left"
    )

    merged_df["known_status"] = np.where(
        merged_df["known_direction_norm"].isin(["Known Import", "Known Export"]),
        "Known",
        "Unknown"
    )

    merged_df, direction_raw_fold_cols, direction_score_fold_cols = calculate_per_fold_direction_scores(
        merged_df,
        common_seed_fold_keys,
        functional_prob_col="mean_prob_functional",
    )

    import_mu_threshold, export_mu_threshold = compute_known_positive_direction_thresholds(merged_df)

    print(
        f"[INFO] Known-positive direction score thresholds: "
        f"import_mu >= {import_mu_threshold:.6f} (min known import), "
        f"export_mu < {export_mu_threshold:.6f} (max known export)"
    )

    merged_df, global_median = apply_stability_filter(
        merged_df,
        direction_score_fold_cols,
        import_mu_threshold=import_mu_threshold,
        export_mu_threshold=export_mu_threshold,
        min_vote=min_vote,
    )

    merged_df["direction_margin_raw"] = (
        merged_df["mean_prob_import"] - merged_df["mean_prob_export"]
    ).abs()

    merged_df["pred_direction_by_score"] = np.select(
        [
            merged_df["direction_score"] > 0,
            merged_df["direction_score"] < 0,
        ],
        [
            "Nuclear Import",
            "Nuclear Export",
        ],
        default="Tie"
    )

    merged_df["functional_import_score"] = np.where(
        merged_df["direction_score"] > 0,
        merged_df["direction_score"],
        0.0
    )

    merged_df["functional_export_score"] = np.where(
        merged_df["direction_score"] < 0,
        -merged_df["direction_score"],
        0.0
    )

    front_cols = [
        "INDEX",
        "ACC_ID",
        "POSITION",
        "functional_MOD_RSD",
        "known_status",
        "known_direction_norm",
        "mean_prob_functional",
        "std_prob_functional",
        "functional_selected",
        "functional_final_threshold",
        "functional_pred_label",
        "mean_prob_import",
        "std_prob_import",
        "mean_prob_export",
        "std_prob_export",
        "direction_raw",
        "direction_raw_std",
        "direction_score",
        "direction_score_std",
        "direction_score_mu",
        "direction_score_sigma",
        "direction_score_global_median",
        "vote_import",
        "vote_export",
        "stable_predicted_import",
        "stable_predicted_export",
        "stable_predicted_direction",
        "direction_abs_score",
        "direction_margin_raw",
        "pred_direction_by_score",
        "functional_import_score",
        "functional_export_score",
        "functional_prob_fold_1",
        "functional_prob_fold_2",
        "functional_prob_fold_3",
        "functional_prob_fold_4",
        "functional_prob_fold_5",
    ]

    front_cols = front_cols + direction_raw_fold_cols + direction_score_fold_cols

    known_annotation_cols = [
        col for col in merged_df.columns
        if col.startswith("known_") and col not in front_cols
    ]

    front_cols = front_cols + known_annotation_cols
    front_cols = [col for col in front_cols if col in merged_df.columns]
    front_cols = list(dict.fromkeys(front_cols))

    other_cols = [col for col in merged_df.columns if col not in front_cols]

    final_df = merged_df[front_cols + other_cols].copy()
    final_df = final_df.loc[:, ~final_df.columns.duplicated()].copy()

    final_df = final_df.sort_values(
        by=["direction_abs_score", "mean_prob_functional"],
        ascending=[False, False]
    )

    final_df.to_csv(output_csv, index=False)

    stable_import_df = final_df.loc[final_df["stable_predicted_import"]].copy()
    stable_export_df = final_df.loc[final_df["stable_predicted_export"]].copy()
    stable_combined_df = final_df.loc[
        final_df["stable_predicted_import"] | final_df["stable_predicted_export"]
    ].copy()

    stable_import_df = stable_import_df.sort_values(
        by=["direction_score_mu", "vote_import", "mean_prob_functional"],
        ascending=[False, False, False],
    )
    stable_export_df = stable_export_df.sort_values(
        by=["direction_score_mu", "vote_export", "mean_prob_functional"],
        ascending=[True, False, False],
    )
    stable_combined_df = stable_combined_df.sort_values(
        by=["stable_predicted_direction", "direction_abs_score", "mean_prob_functional"],
        ascending=[True, False, False],
    )

    stable_rule_summary_df = build_stability_rule_summary(
        final_df,
        global_median,
        import_mu_threshold,
        export_mu_threshold,
    )

    stable_import_df.to_csv(stable_import_output_csv, index=False)
    stable_export_df.to_csv(stable_export_output_csv, index=False)
    stable_combined_df.to_csv(stable_combined_output_csv, index=False)
    stable_rule_summary_df.to_csv(stable_rule_summary_csv, index=False)

    plot_direction_score_signed_scatter_known_direction(
        final_df,
        direction_score_fold_cols,
        scatter_output_png,
        import_mu_threshold=import_mu_threshold,
        export_mu_threshold=export_mu_threshold,
    )

    print(f"[DONE] Saved output to: {output_csv}")
    print(f"[DONE] Saved stable import sites to: {stable_import_output_csv}")
    print(f"[DONE] Saved stable export sites to: {stable_export_output_csv}")
    print(f"[DONE] Saved stable combined sites to: {stable_combined_output_csv}")
    print(f"[DONE] Saved stability rule summary to: {stable_rule_summary_csv}")
    print(f"[DONE] Saved scatter plot to: {scatter_output_png}")
    print(f"[INFO] Number of final scored sites: {len(final_df)}")
    print(f"[INFO] Number of stable import sites: {len(stable_import_df)}")
    print(f"[INFO] Number of stable export sites: {len(stable_export_df)}")
    print(f"[INFO] Number of stable combined sites: {len(stable_combined_df)}")
    print(f"[INFO] Global direction score median: {global_median}")
    print(f"[INFO] Number of sites missing functional probability: {missing_functional_n}")
    print(f"[INFO] Functional probability threshold: {functional_score_threshold}")
    print(f"[INFO] Stability import rule: mu >= {import_mu_threshold}, vote_import >= {min_vote}, mu > global_median")
    print(f"[INFO] Stability export rule: mu < {export_mu_threshold}, vote_export >= {min_vote}, mu < global_median")
    print(f"[INFO] Import probability columns: {len(import_prob_cols)}")
    print(f"[INFO] Export probability columns: {len(export_prob_cols)}")
    print(f"[INFO] Matched seed fold pairs: {len(common_seed_fold_keys)}")
    print(f"[INFO] Per-fold direction score columns: {len(direction_score_fold_cols)}")

    print("[INFO] Stable predicted direction counts:")
    print(final_df["stable_predicted_direction"].value_counts())

    print("[INFO] Stability rule count summary:")
    print(stable_rule_summary_df)

    print("[INFO] Known direction counts:")
    print(final_df["known_direction_norm"].fillna("Unknown").value_counts())

    print("[INFO] Predicted direction counts:")
    print(final_df["pred_direction_by_score"].value_counts())

    print("[INFO] Direction raw summary:")
    print(final_df["direction_raw"].describe())

    print("[INFO] Direction score summary:")
    print(final_df["direction_score"].describe())

    print("[INFO] Direction score standard deviation summary:")
    print(final_df["direction_score_std"].describe())

    print("[INFO] Top 10 sites by absolute direction score:")
    print(
        final_df[
            [
                "INDEX",
                "known_direction_norm",
                "mean_prob_functional",
                "mean_prob_import",
                "mean_prob_export",
                "direction_raw",
                "direction_raw_std",
                "direction_score",
                "direction_score_std",
                "direction_abs_score",
                "pred_direction_by_score",
            ]
        ].head(10)
    )


if __name__ == "__main__":
    main()
