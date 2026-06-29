# Panels: Supplementary Figure 2(b)


# Monorepo path constants (monorepo relative paths)
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PRECOMPUTED = PROJECT_ROOT / "data" / "precomputed"
TF_FAMILY_PATH = PROJECT_ROOT / "data" / "TF_family" / "TF_Information.txt"

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


FUNCTIONAL_CSV = DATA_PRECOMPUTED / "2_1_functional_classifier_results" / "predictions" / "esm_window_site_pdb_5_folds_ensemble_predictions.csv"

POSITIVE_CSV = PROJECT_ROOT / "data" / "dataset_phos_site" / "TF_positive_phos_site_0608.csv"

BACKGROUND_CSV = PROJECT_ROOT / "data" / "dataset_phos_site" / "TF_deepmvp_negative_phos_site_tf_only.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "2_1_functional_classifier_results" / "distribution" / "functional_score_distribution_5_folds_ensemble_esm_window_site_pdb"

SCORE_COL = "mean_prob"
MIN_DISTANCE_TO_POSITIVE = 30
EXCLUDE_PREDICTED_POSITIVE_FROM_NEGATIVE = False

# Match palette used in model ablation / window-cluster plots
NEGATIVE_COLOR = "#95b0b5"
POSITIVE_COLOR = "#ad7551"
HIST_ALPHA = 0.35
FILL_ALPHA = 0.20
FONT_SIZE_SHIFT = 4


def setup_plot_style():
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "DejaVu Sans",
        "font.size": 10 + FONT_SIZE_SHIFT,
        "axes.titlesize": 11 + FONT_SIZE_SHIFT,
        "axes.labelsize": 10 + FONT_SIZE_SHIFT,
        "xtick.labelsize": 9 + FONT_SIZE_SHIFT,
        "ytick.labelsize": 9 + FONT_SIZE_SHIFT,
        "legend.fontsize": 8 + FONT_SIZE_SHIFT,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "figure.dpi": 300,
    })


def normalize_position(series):
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def ensure_index(df):
    df = df.copy()

    if "ACC_ID" not in df.columns:
        raise ValueError("ACC_ID column is required.")

    df["ACC_ID"] = df["ACC_ID"].astype(str)

    if "POSITION" in df.columns:
        df["POSITION"] = normalize_position(df["POSITION"])

    if "INDEX" in df.columns:
        df["INDEX"] = df["INDEX"].astype(str)
        return df

    if "MOD_RSD" in df.columns:
        df["MOD_RSD"] = df["MOD_RSD"].astype(str)
        df["INDEX"] = df["ACC_ID"] + "_" + df["MOD_RSD"]
        return df

    if "POSITION" in df.columns and "FULL_SEQUENCE" in df.columns:
        residues = []
        for _, row in df.iterrows():
            seq = str(row["FULL_SEQUENCE"])
            pos = row["POSITION"]
            if pd.isna(pos):
                residues.append("X")
            else:
                pos_int = int(pos)
                if 1 <= pos_int <= len(seq):
                    residues.append(seq[pos_int - 1])
                else:
                    residues.append("X")
        df["INDEX"] = df["ACC_ID"] + "_" + pd.Series(residues, index=df.index) + df["POSITION"].astype(str)
        return df

    if "POSITION" in df.columns:
        df["INDEX"] = df["ACC_ID"] + "_" + df["POSITION"].astype(str)
        return df

    raise ValueError("Cannot build INDEX. Need INDEX, MOD_RSD, or POSITION.")


def read_table(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    df = ensure_index(df)
    return df


def make_site_key(df):
    if "ACC_ID" not in df.columns or "POSITION" not in df.columns:
        raise ValueError("ACC_ID and POSITION columns are required for site key generation.")
    positions = normalize_position(df["POSITION"])
    return df["ACC_ID"].astype(str) + "|" + positions.astype(str)


def get_final_threshold(pred_df):
    threshold_cols = ["threshold_model_1", "final_threshold"]
    for col in threshold_cols:
        if col in pred_df.columns:
            vals = pd.to_numeric(pred_df[col], errors="coerce").dropna()
            if len(vals) > 0:
                return float(vals.median())
    return 0.5


def kde_curve(values, grid, bandwidth=None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return np.zeros_like(grid)

    if bandwidth is None:
        std = np.std(values, ddof=1)
        n = len(values)
        bandwidth = 1.06 * std * (n ** (-1 / 5))
        if not np.isfinite(bandwidth) or bandwidth <= 0:
            bandwidth = 0.03

    diff = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * diff ** 2).sum(axis=1)
    density = density / (len(values) * bandwidth * np.sqrt(2 * np.pi))
    return density


def plot_hist_density(plot_df, output_dir):
    positive_scores = plot_df.loc[plot_df["group"] == "Positive", SCORE_COL].dropna().astype(float).values
    negative_scores = plot_df.loc[plot_df["group"] == "Negative", SCORE_COL].dropna().astype(float).values

    fig, ax1 = plt.subplots(figsize=(5.2, 4.0))
    bins = np.linspace(0, 1, 31)

    ax1.hist(
        negative_scores,
        bins=bins,
        density=True,
        alpha=HIST_ALPHA,
        color=NEGATIVE_COLOR,
        label=f"Negative hist. (n={len(negative_scores)})",
        edgecolor="white",
        linewidth=0.4,
    )

    ax1.hist(
        positive_scores,
        bins=bins,
        density=True,
        alpha=HIST_ALPHA,
        color=POSITIVE_COLOR,
        label=f"Positive hist. (n={len(positive_scores)})",
        edgecolor="white",
        linewidth=0.4,
    )

    grid = np.linspace(0, 1, 500)
    neg_density = kde_curve(negative_scores, grid)
    pos_density = kde_curve(positive_scores, grid)

    ax1.plot(grid, neg_density, linewidth=2.0, color=NEGATIVE_COLOR, label="Negative density")
    ax1.plot(grid, pos_density, linewidth=2.0, color=POSITIVE_COLOR, label="Positive density")

    ax1.set_xlabel("Functional prediction score")
    ax1.set_ylabel("Density")
    ax1.set_xlim(0, 1)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(frameon=False, fontsize=8 + FONT_SIZE_SHIFT)

    plt.tight_layout()

    fig.savefig(output_dir / "functional_score_positive_negative_hist_density.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "functional_score_positive_negative_hist_density.png", bbox_inches="tight", dpi=600)
    plt.close(fig)


def plot_density_only(plot_df, output_dir):
    positive_scores = plot_df.loc[plot_df["group"] == "Positive", SCORE_COL].dropna().astype(float).values
    negative_scores = plot_df.loc[plot_df["group"] == "Negative", SCORE_COL].dropna().astype(float).values

    grid = np.linspace(0, 1, 500)
    neg_density = kde_curve(negative_scores, grid)
    pos_density = kde_curve(positive_scores, grid)

    fig, ax = plt.subplots(figsize=(5.2, 4.0))

    ax.plot(grid, neg_density, linewidth=2.2, color=NEGATIVE_COLOR, label=f"Negative (n={len(negative_scores)})")
    ax.plot(grid, pos_density, linewidth=2.2, color=POSITIVE_COLOR, label=f"Positive (n={len(positive_scores)})")

    ax.fill_between(grid, 0, neg_density, alpha=FILL_ALPHA, color=NEGATIVE_COLOR)
    ax.fill_between(grid, 0, pos_density, alpha=FILL_ALPHA, color=POSITIVE_COLOR)

    ax.set_xlabel("Functional prediction score")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8 + FONT_SIZE_SHIFT)

    plt.tight_layout()

    fig.savefig(output_dir / "functional_score_positive_negative_density_only.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "functional_score_positive_negative_density_only.png", bbox_inches="tight", dpi=600)
    plt.close(fig)


def add_nearest_positive_distance(bg_df, pos_df):
    bg_df = bg_df.copy()
    pos_df = pos_df.copy()

    bg_df["POSITION"] = pd.to_numeric(bg_df["POSITION"], errors="coerce")
    pos_df["POSITION"] = pd.to_numeric(pos_df["POSITION"], errors="coerce")

    bg_df = bg_df.dropna(subset=["ACC_ID", "POSITION"]).copy()
    pos_df = pos_df.dropna(subset=["ACC_ID", "POSITION"]).copy()

    bg_df["POSITION"] = bg_df["POSITION"].astype(int)
    pos_df["POSITION"] = pos_df["POSITION"].astype(int)

    pos_position_map = {
        acc: group["POSITION"].to_numpy(dtype=int)
        for acc, group in pos_df.groupby("ACC_ID")
    }

    nearest_distances = []

    for acc, pos in zip(bg_df["ACC_ID"], bg_df["POSITION"]):
        positive_positions = pos_position_map.get(acc)

        if positive_positions is None or len(positive_positions) == 0:
            nearest_distances.append(np.nan)
            continue

        nearest_distance = np.min(np.abs(positive_positions - pos))
        nearest_distances.append(nearest_distance)

    bg_df["nearest_positive_distance"] = nearest_distances
    return bg_df


def main():
    setup_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pred_df = read_table(FUNCTIONAL_CSV)
    pos_df = read_table(POSITIVE_CSV)
    bg_df = read_table(BACKGROUND_CSV)

    if SCORE_COL not in pred_df.columns:
        raise ValueError(f"{SCORE_COL} is missing from functional prediction file.")

    pred_df[SCORE_COL] = pd.to_numeric(pred_df[SCORE_COL], errors="coerce")
    pred_df = pred_df.dropna(subset=[SCORE_COL])

    pos_keys = set(pos_df["INDEX"].astype(str))
    pos_site_keys = set(make_site_key(pos_df))
    pos_proteins = set(pos_df["ACC_ID"].astype(str))

    n_bg_original = len(bg_df)

    bg_df = bg_df[bg_df["ACC_ID"].astype(str).isin(pos_proteins)].copy()
    n_bg_after_protein_filter = len(bg_df)

    bg_df["site_key"] = make_site_key(bg_df)
    bg_df = bg_df[
        ~bg_df["INDEX"].astype(str).isin(pos_keys)
        & ~bg_df["site_key"].isin(pos_site_keys)
    ].copy()
    bg_df = bg_df.drop(columns=["site_key"])
    n_bg_after_positive_site_removal = len(bg_df)

    bg_df = add_nearest_positive_distance(bg_df, pos_df)
    bg_df = bg_df[bg_df["nearest_positive_distance"] > MIN_DISTANCE_TO_POSITIVE].copy()
    n_bg_after_distance_filter = len(bg_df)

    pred_keep_cols = [
        "INDEX",
        "ACC_ID",
        "POSITION",
        "MOD_RSD",
        "LABEL",
        "prob_fold_1",
        "prob_fold_2",
        "prob_fold_3",
        "prob_fold_4",
        "prob_fold_5",
        "mean_prob",
        "std_prob",
    ]
    pred_keep_cols = [c for c in pred_keep_cols if c in pred_df.columns]

    pos_score_df = pos_df[["INDEX", "ACC_ID", "POSITION"]].drop_duplicates("INDEX").merge(
        pred_df[pred_keep_cols],
        on="INDEX",
        how="inner",
        suffixes=("", "_pred"),
    )
    pos_score_df["group"] = "Positive"

    neg_base_cols = ["INDEX", "ACC_ID", "POSITION", "nearest_positive_distance"]
    neg_score_df = bg_df[neg_base_cols].drop_duplicates("INDEX").merge(
        pred_df[pred_keep_cols],
        on="INDEX",
        how="inner",
        suffixes=("", "_pred"),
    )
    neg_score_df["group"] = "Negative"

    if EXCLUDE_PREDICTED_POSITIVE_FROM_NEGATIVE:
        threshold = get_final_threshold(pred_df)

        if "pred_label" in neg_score_df.columns:
            neg_score_df = neg_score_df[
                pd.to_numeric(neg_score_df["pred_label"], errors="coerce") != 1
            ].copy()
        else:
            neg_score_df = neg_score_df[neg_score_df[SCORE_COL] < threshold].copy()

    plot_df = pd.concat([pos_score_df, neg_score_df], axis=0, ignore_index=True)
    plot_df = plot_df.dropna(subset=[SCORE_COL])

    filtered_path = OUTPUT_DIR / "functional_score_positive_negative_filtered_sites.csv"
    plot_df.to_csv(filtered_path, index=False)

    negative_path = OUTPUT_DIR / "functional_negative_sites_distance_gt30.csv"
    neg_score_df.to_csv(negative_path, index=False)

    summary_df = (
        plot_df.groupby("group")[SCORE_COL]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )

    summary_path = OUTPUT_DIR / "functional_score_positive_negative_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    plot_hist_density(plot_df, OUTPUT_DIR)
    plot_density_only(plot_df, OUTPUT_DIR)

    print("[DONE] Saved filtered sites to:", filtered_path)
    print("[DONE] Saved negative sites to:", negative_path)
    print("[DONE] Saved summary to:", summary_path)
    print("[DONE] Saved figures to:", OUTPUT_DIR)

    print("[INFO] Score column:", SCORE_COL)
    print("[INFO] Minimum distance to positive site:", MIN_DISTANCE_TO_POSITIVE)
    print("[INFO] Positive proteins:", len(pos_proteins))
    print("[INFO] Positive sites in positive CSV:", len(pos_keys))
    print("[INFO] Background sites before filter:", n_bg_original)
    print("[INFO] Background sites after positive protein filter:", n_bg_after_protein_filter)
    print("[INFO] Background sites after removing exact positive sites:", n_bg_after_positive_site_removal)
    print("[INFO] Background sites after distance > 30 filter:", n_bg_after_distance_filter)
    print("[INFO] Positive sites matched to prediction:", len(pos_score_df))
    print("[INFO] Negative sites matched to prediction:", len(neg_score_df))

    print(summary_df)


if __name__ == "__main__":
    main()
