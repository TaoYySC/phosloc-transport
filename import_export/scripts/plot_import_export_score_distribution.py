# Panels: Supplementary Figure 4(a,b)


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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import mannwhitneyu
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ESM Window only + SupCon+CE (Import=LABEL 1), 5-fold CV OOF probabilities
# Each site appears once in its held-out outer fold (124 sites, 0608 dataset).
run_dir = (
    str(DATA_PRECOMPUTED / "run_20260612_125646_esm_window_only_supcon_ce_import_pos" / "Import_vs_Export")
)
oof_csv = f"{run_dir}/all_fold_test_predictions_platt.csv"
oof_csv_fallback = f"{run_dir}/all_fold_test_predictions.csv"

# Func 5-fold ESM Window+Site+PDB ensemble predictions (mean_prob as functional score)
functional_csv = (
    str(FUNCTIONAL_PRECOMPUTED / "2_1_functional_classifier_results" / "predictions" / "esm_window_site_pdb_5_folds_ensemble_predictions.csv")
)

output_dir = (
    str(PROJECT_ROOT / "results" / "1_transport_classifier_results" / "esm_window_only_supcon_ce_import_pos_score_distribution_platt")
)

aggregate_by_site = True
selected_region_fraction = 0.20
annotation_top_fraction = 0.10

IMPORT_COLOR = "#9a3f3f"
EXPORT_COLOR = "#4f7d95"

FONT_SIZE_SHIFT = 4

os.makedirs(output_dir, exist_ok=True)


def set_publication_style():
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "DejaVu Sans",
        "font.size": 10 + FONT_SIZE_SHIFT,
        "axes.titlesize": 11 + FONT_SIZE_SHIFT,
        "axes.labelsize": 10 + FONT_SIZE_SHIFT,
        "xtick.labelsize": 9 + FONT_SIZE_SHIFT,
        "ytick.labelsize": 9 + FONT_SIZE_SHIFT,
        "legend.fontsize": 9 + FONT_SIZE_SHIFT,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "savefig.dpi": 300,
    })


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_panel_label(ax, label):
    ax.text(
        -0.16,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=13 + FONT_SIZE_SHIFT,
        fontweight="bold",
        va="top",
        ha="left",
    )


def p_to_stars(p):
    if pd.isna(p):
        return ""
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def add_significance_bar(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.1, c="black")
    ax.text((x1 + x2) / 2, y + h + 0.015, text, ha="center", va="bottom", fontsize=11 + FONT_SIZE_SHIFT)


def make_site_key(df):
    df = df.copy()

    if "INDEX" in df.columns:
        df["_site_key"] = df["INDEX"].astype(str)
        return df

    if {"ACC_ID", "POSITION", "RESIDUE"}.issubset(df.columns):
        df["_site_key"] = (
            df["ACC_ID"].astype(str).str.strip()
            + "_"
            + df["RESIDUE"].astype(str).str.strip()
            + df["POSITION"].astype(str).str.strip()
        )
        return df

    if {"ACC_ID", "POSITION"}.issubset(df.columns):
        df["_site_key"] = (
            df["ACC_ID"].astype(str).str.strip()
            + "_"
            + df["POSITION"].astype(str).str.strip()
        )
        return df

    raise ValueError("Cannot build site key. Expected INDEX or ACC_ID and POSITION columns.")


def find_functional_score_column(df):
    candidates = [
        "functional_score",
        "mean_prob",
        "prob_mean",
        "mean_probability",
        "functional_prob",
        "y_prob_mean",
        "probability",
        "prob",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = {"POSITION", "rank", "n_oof"}
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    if len(numeric_cols) == 1:
        return numeric_cols[0]

    raise ValueError(
        "Cannot determine the functional score column. "
        "Please rename the score column to functional_score or mean_prob."
    )


def prepare_site_level_table(df, aggregate_by_site=True):
    required_cols = ["prob_import", "Transport_Direction"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if aggregate_by_site:
        group_cols = [
            c for c in [
                "INDEX",
                "ACC_ID",
                "POSITION",
                "RESIDUE",
                "LABEL",
                "Transport_Direction",
            ]
            if c in df.columns
        ]

        if len(group_cols) == 0:
            raise ValueError("No valid site identifier columns were found for aggregation.")

        agg_dict = {
            "prob_import": ("prob_import", "mean"),
            "n_oof": ("prob_import", "size"),
        }

        if "prob_export" in df.columns:
            agg_dict["prob_export"] = ("prob_export", "mean")

        site_df = df.groupby(group_cols, as_index=False).agg(**agg_dict)

        if "prob_export" not in site_df.columns:
            site_df["prob_export"] = 1.0 - site_df["prob_import"]

    else:
        site_df = df.copy()

        if "prob_export" not in site_df.columns:
            site_df["prob_export"] = 1.0 - site_df["prob_import"]

        site_df["n_oof"] = 1

    site_df = make_site_key(site_df)
    site_df["true_direction"] = site_df["Transport_Direction"].astype(str).str.strip()
    site_df = site_df[site_df["true_direction"].isin(["Nuclear Import", "Nuclear Export"])].copy()

    site_df["direction_raw"] = 2.0 * site_df["prob_import"] - 1.0
    site_df["is_import"] = (site_df["true_direction"] == "Nuclear Import").astype(int)
    site_df["is_export"] = (site_df["true_direction"] == "Nuclear Export").astype(int)

    return site_df


def prepare_functional_scores(functional_csv_path):
    func_df = pd.read_csv(functional_csv_path)
    func_df = make_site_key(func_df)

    score_col = find_functional_score_column(func_df)

    keep_cols = ["_site_key", score_col]
    keep_cols.extend([c for c in ["INDEX", "ACC_ID", "POSITION", "RESIDUE"] if c in func_df.columns])

    func_df = func_df[keep_cols].copy()
    func_df = func_df.rename(columns={score_col: "functional_score"})
    func_df["functional_score"] = pd.to_numeric(func_df["functional_score"], errors="coerce")
    func_df["functional_score"] = func_df["functional_score"].clip(lower=0.0, upper=1.0)

    agg_cols = {"functional_score": "mean"}

    for col in ["INDEX", "ACC_ID", "POSITION", "RESIDUE"]:
        if col in func_df.columns:
            agg_cols[col] = "first"

    func_df = func_df.groupby("_site_key", as_index=False).agg(agg_cols)
    return func_df


def merge_functional_scores(site_df, func_df):
    merged = site_df.merge(
        func_df[["_site_key", "functional_score"]],
        on="_site_key",
        how="left",
    )

    missing_count = merged["functional_score"].isna().sum()

    if missing_count > 0:
        print(f"[WARN] Missing functional_score for {missing_count} sites. Filling with 0.0.")
        merged["functional_score"] = merged["functional_score"].fillna(0.0)

    merged["direction_score"] = merged["direction_raw"] * merged["functional_score"]

    merged = merged.sort_values("direction_score", ascending=False).reset_index(drop=True)
    merged["rank"] = np.arange(1, len(merged) + 1)
    merged["rank_percentile"] = 100.0 * merged["rank"] / len(merged)

    return merged


def calculate_rank_class_ratio(site_df):
    ranked_df = site_df.sort_values("direction_score", ascending=False).reset_index(drop=True)

    n_sites = len(ranked_df)
    ranks = np.arange(1, n_sites + 1)

    cumulative_import_count = ranked_df["is_import"].cumsum().to_numpy()
    cumulative_export_count = ranked_df["is_export"].cumsum().to_numpy()

    import_ratio = cumulative_import_count / ranks
    export_ratio = cumulative_export_count / ranks

    ratio_df = pd.DataFrame({
        "rank": ranks,
        "direction_score": ranked_df["direction_score"].to_numpy(),
        "cumulative_import_count": cumulative_import_count,
        "cumulative_export_count": cumulative_export_count,
        "import_ratio_before_rank": import_ratio,
        "export_ratio_before_rank": export_ratio,
    })

    return ratio_df


def draw_import_export_boxplot(ax, site_df):
    import_scores = site_df.loc[
        site_df["true_direction"] == "Nuclear Import",
        "direction_score",
    ].values

    export_scores = site_df.loc[
        site_df["true_direction"] == "Nuclear Export",
        "direction_score",
    ].values

    data = [import_scores, export_scores]

    labels = [
        f"Import\n(n={len(import_scores)})",
        f"Export\n(n={len(export_scores)})",
    ]

    box = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.4},
        whiskerprops={"linewidth": 1.1},
        capprops={"linewidth": 1.1},
        boxprops={"linewidth": 1.1},
    )

    colors = [IMPORT_COLOR, EXPORT_COLOR]

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)

    rng = np.random.default_rng(42)

    for i, values in enumerate(data, start=1):
        x = rng.normal(i, 0.045, size=len(values))
        ax.scatter(
            x,
            values,
            s=30,
            color=colors[i - 1],
            alpha=0.65,
            edgecolor="white",
            linewidth=0.3,
            zorder=3,
        )

    ax.axhline(0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels)
    ax.set_ylabel("Direction score")
    ax.set_ylim(-1.05, 1.12)
    clean_axis(ax)

    if HAS_SCIPY and len(import_scores) > 0 and len(export_scores) > 0:
        p = mannwhitneyu(
            import_scores,
            export_scores,
            alternative="two-sided",
        ).pvalue
        stars = p_to_stars(p)

        if stars != "":
            add_significance_bar(ax, 1, 2, 0.98, 0.04, stars)


def draw_ranking_scatter(ax, site_df, selected_fraction):
    import_df = site_df[site_df["true_direction"] == "Nuclear Import"]
    export_df = site_df[site_df["true_direction"] == "Nuclear Export"]

    ax.scatter(
        import_df["rank_percentile"],
        import_df["direction_score"],
        s=36,
        alpha=0.70,
        color=IMPORT_COLOR,
        label=f"Import (n={len(import_df)})",
        edgecolor="white",
        linewidth=0.3,
    )

    ax.scatter(
        export_df["rank_percentile"],
        export_df["direction_score"],
        s=36,
        alpha=0.70,
        color=EXPORT_COLOR,
        label=f"Export (n={len(export_df)})",
        edgecolor="white",
        linewidth=0.3,
    )

    left_cut = 100 * selected_fraction
    right_cut = 100 * (1 - selected_fraction)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1.0)
    ax.axvspan(0, left_cut, color="0.92", zorder=0)
    ax.axvspan(right_cut, 100, color="0.92", zorder=0)
    ax.axvline(left_cut, color="gray", linestyle=":", linewidth=1.0)
    ax.axvline(right_cut, color="gray", linestyle=":", linewidth=1.0)

    ax.set_xlabel("Rank percentile")
    ax.set_ylabel("Direction score")
    ax.set_xlim(0, 100)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(frameon=False, loc="upper right")
    clean_axis(ax)


def draw_enrichment_curve(ax, site_df, top_fraction):
    ratio_df = calculate_rank_class_ratio(site_df)

    top_rank = max(1, int(np.ceil(len(site_df) * top_fraction)))
    top_row = ratio_df.iloc[top_rank - 1]

    ax.plot(
        ratio_df["rank"],
        ratio_df["import_ratio_before_rank"],
        color=IMPORT_COLOR,
        linewidth=2.0,
        label="Import enrichment score",
    )

    ax.plot(
        ratio_df["rank"],
        ratio_df["export_ratio_before_rank"],
        color=EXPORT_COLOR,
        linewidth=2.0,
        label="Export enrichment score",
    )

    ax.scatter(
        [top_rank],
        [top_row["import_ratio_before_rank"]],
        s=30,
        color=IMPORT_COLOR,
        zorder=3,
    )

    ax.scatter(
        [top_rank],
        [top_row["export_ratio_before_rank"]],
        s=30,
        color=EXPORT_COLOR,
        zorder=3,
    )

    ax.axvline(top_rank, color="gray", linestyle=":", linewidth=1.0)
    ax.axvspan(1, top_rank, color="0.94", zorder=0)

    ax.set_xlabel("Rank")
    ax.set_ylabel("Enrichment score")
    ax.set_xlim(1, len(site_df))
    ax.set_ylim(0, 1.02)

    ax.legend(frameon=False, loc="upper right")

    clean_axis(ax)


def save_outputs(site_df, resolved_oof_csv):
    ratio_df = calculate_rank_class_ratio(site_df)

    top_rank = max(1, int(np.ceil(len(site_df) * annotation_top_fraction)))
    top_row = ratio_df.iloc[top_rank - 1]
    top_percent_label = int(round(annotation_top_fraction * 100))

    site_df.to_csv(
        os.path.join(output_dir, "site_level_direction_scores_sorted.csv"),
        index=False,
    )

    ratio_df.to_csv(
        os.path.join(output_dir, "direction_score_rank_class_ratio_curve.csv"),
        index=False,
    )

    import_scores = site_df.loc[
        site_df["true_direction"] == "Nuclear Import",
        "direction_score",
    ].values

    export_scores = site_df.loc[
        site_df["true_direction"] == "Nuclear Export",
        "direction_score",
    ].values

    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(f"Run dir: {run_dir}\n")
        f.write(f"OOF input: {resolved_oof_csv}\n")
        f.write("Calibration: Platt scaling on OOF decision_function\n")
        f.write(f"Functional input: {functional_csv}\n")
        f.write(f"Total sites: {len(site_df)}\n")
        f.write(f"Import sites: {len(import_scores)}\n")
        f.write(f"Export sites: {len(export_scores)}\n")
        f.write(f"Overall import fraction: {site_df['is_import'].mean():.4f}\n")
        f.write(f"Overall export fraction: {site_df['is_export'].mean():.4f}\n")
        f.write(f"Import score mean: {np.mean(import_scores):.4f}\n")
        f.write(f"Export score mean: {np.mean(export_scores):.4f}\n")
        f.write(f"Import score median: {np.median(import_scores):.4f}\n")
        f.write(f"Export score median: {np.median(export_scores):.4f}\n")
        f.write(f"Top {top_percent_label} percent cutoff rank: {top_rank}\n")
        f.write(f"Top {top_percent_label} percent import ratio: {top_row['import_ratio_before_rank']:.4f}\n")
        f.write(f"Top {top_percent_label} percent export ratio: {top_row['export_ratio_before_rank']:.4f}\n")


def save_single_panel(draw_func, site_df, panel_name, figsize=(5.0, 4.4)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if panel_name == "A":
        draw_func(ax, site_df)
    elif panel_name == "B":
        draw_func(ax, site_df, selected_region_fraction)
    elif panel_name == "C":
        draw_func(ax, site_df, annotation_top_fraction)
    else:
        raise ValueError(f"Unknown panel_name: {panel_name}")

    fig.tight_layout()

    base = os.path.join(output_dir, f"panel_{panel_name}")

    fig.savefig(base + ".png", bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".svg", bbox_inches="tight")

    plt.close(fig)


def resolve_oof_csv():
    if os.path.exists(oof_csv):
        return oof_csv
    if os.path.exists(oof_csv_fallback):
        print(
            f"[WARN] Platt-calibrated OOF not found: {oof_csv}\n"
            f"       Falling back to raw probabilities: {oof_csv_fallback}\n"
            f"       Run: python scripts/fit_platt_calibration_import_export.py"
        )
        return oof_csv_fallback
    raise FileNotFoundError(
        f"Neither {oof_csv} nor {oof_csv_fallback} exists. "
        "Run fit_platt_calibration_import_export.py first."
    )


def main():
    set_publication_style()

    resolved_oof_csv = resolve_oof_csv()
    pred_df = pd.read_csv(resolved_oof_csv)
    site_df = prepare_site_level_table(pred_df, aggregate_by_site=aggregate_by_site)

    func_df = prepare_functional_scores(functional_csv)
    site_df = merge_functional_scores(site_df, func_df)

    print(f"[INFO] Run dir: {run_dir}")
    print(f"[INFO] OOF input: {resolved_oof_csv}")
    print(f"[INFO] Functional input: {functional_csv}")
    print(f"[INFO] Total sites: {len(site_df)}")
    print(f"[INFO] Import sites: {(site_df['true_direction'] == 'Nuclear Import').sum()}")
    print(f"[INFO] Export sites: {(site_df['true_direction'] == 'Nuclear Export').sum()}")

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.6))

    draw_import_export_boxplot(axes[0], site_df)
    draw_ranking_scatter(axes[1], site_df, selected_region_fraction)
    draw_enrichment_curve(axes[2], site_df, annotation_top_fraction)

    add_panel_label(axes[0], "A")
    add_panel_label(axes[1], "B")
    add_panel_label(axes[2], "C")

    fig.tight_layout()

    base = os.path.join(output_dir, "direction_score_panels_A_B_C")

    fig.savefig(base + ".png", bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".svg", bbox_inches="tight")

    plt.close(fig)

    save_single_panel(draw_import_export_boxplot, site_df, "A", figsize=(4.8, 4.4))
    save_single_panel(draw_ranking_scatter, site_df, "B", figsize=(5.2, 4.4))
    save_single_panel(draw_enrichment_curve, site_df, "C", figsize=(5.6, 4.4))

    save_outputs(site_df, resolved_oof_csv)

    print("[DONE] Finished.")
    print(f"[DONE] Output directory: {output_dir}")


if __name__ == "__main__":
    main()
