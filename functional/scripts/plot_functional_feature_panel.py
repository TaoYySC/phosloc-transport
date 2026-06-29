# Panels: Supplementary Figure 3(a-i)

"""Draw a combined selected-feature panel for functional site groups."""

# Monorepo path constants (monorepo relative paths)
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PRECOMPUTED = PROJECT_ROOT / "data" / "precomputed"
TF_FAMILY_PATH = PROJECT_ROOT / "data" / "TF_family" / "TF_Information.txt"


import re
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import fisher_exact, mannwhitneyu


mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"

PREDICTION_CSV = DATA_PRECOMPUTED / "2_1_functional_classifier_results" / "predictions" / "esm_window_site_pdb_5_folds_ensemble_predictions.csv"

POSITIVE_CSV = PROJECT_ROOT / "data" / "dataset_phos_site" / "TF_positive_phos_site_0608.csv"

NEG_SITE_CSV = PROJECT_ROOT / "data" / "dataset_phos_site" / "TF_deepmvp_negative_phos_site_tf_only.csv"

FEATURE_DIR = PROJECT_ROOT / "data" / "features"

FEATURE_TABLE_FILES = {
    "motif1433": "1433_features_all_sty.csv",
    "alphamissense": "alphamissense_features_all_sty.csv",
    "domain": "domain_features_all_sty.csv",
    "evolution": "evolution_features_all_sty.csv",
    "idr": "idr_features_all_sty.csv",
    "kinase": "kinase_features_all_sty.csv",
    "nes": "nes_features_all_sty.csv",
    "nls": "nls_features_all_sty.csv",
    "sequence": "sequence_features_all_sty.csv",
}

OUTPUT_ROOT = PROJECT_ROOT / "results" / "2_1_functional_classifier_results" / "feature_boxplot_stacked_barplot" / "functional_selected_panel"

INDEX_COL = "INDEX"
PROTEIN_COL = "ACC_ID"
SCORE_COL = "mean_prob"

KNOWN_FUNCTIONAL_COLOR = "#ad7551"
NEW_FUNCTIONAL_COLOR = "#c9a085"
SHARED_NEG_COLOR = "#bdbdbd"

POS_ONE_COLOR = "#8f4f2f"
POS_ZERO_COLOR = "#d9b59f"
NEG_ONE_COLOR = "#555555"
NEG_ZERO_COLOR = "#d0d0d0"

BOX_LABEL_FONTSIZE = 17.2
YTICK_FONTSIZE = 15.2
PVALUE_FONTSIZE = 15.8

GROUP_ORDER = [
    "new_functional",
    "known_functional",
    "shared_neg",
]

GROUP_COLORS = {
    "new_functional": NEW_FUNCTIONAL_COLOR,
    "known_functional": KNOWN_FUNCTIONAL_COLOR,
    "shared_neg": SHARED_NEG_COLOR,
}

POSITION_MAP_BOX = {
    "new_functional": 0.88,
    "known_functional": 1.18,
    "shared_neg": 1.48,
}

POSITION_MAP_BAR = {
    "new_functional": 0.90,
    "known_functional": 1.20,
    "shared_neg": 1.50,
}

BINARY_NAME_PATTERNS = ["flag", "bool", "binary"]

# Main figure: 3 continuous boxplot features.
MAIN_PANEL_FEATURES: List[str] = [
    "FUNC_AlphaMissense_PathMax",
    "FUNC_Evolution_Site_Score",
    "MOTIF_IDR_Site_DisorderScore",
]

# Supplementary figure: 3 box + 3 flag.
SUPP_PANEL_FEATURES: List[str] = [
    "FUNC_Kinase_PWM_MSS_CK1_Max",
    "SEQ_Sequence_Window_AAFrac_Lys",
    "SEQ_Sequence_Window_Hydropathy_MeanKD",
    "MOTIF_Domain_DBD_Inside_Flag",
    "MOTIF_Domain_Linker_Inside_Flag",
    "MOTIF_IDR_Inside_Flag",
]

COMBINED_PANEL_FEATURES: List[str] = MAIN_PANEL_FEATURES + SUPP_PANEL_FEATURES

SELECTED_FEATURES: List[str] = []
APPLY_PANEL_FILTER_ON_MANUAL = False

P_VALUE_THRESHOLD = 0.05
LEGEND_FONTSIZE = 15.5
PREDICTION_THRESHOLD = 0.6

COMBINED_PANEL_BASENAME = "functional_selected_feature_combined_panel_shared_negative"


def ensure_index(df, acc_col="ACC_ID", site_col="POSITION"):
    df = df.copy()
    if "INDEX" not in df.columns:
        if acc_col not in df.columns or site_col not in df.columns:
            raise ValueError(f"Missing columns for INDEX construction: {acc_col}, {site_col}")
        df["INDEX"] = df[acc_col].astype(str) + "_" + df[site_col].astype(int).astype(str)
    df["INDEX"] = df["INDEX"].astype(str)
    return df


def is_binary_feature_name(feature_name: str) -> bool:
    feature_lower = str(feature_name).lower()
    if any(pattern in feature_lower for pattern in BINARY_NAME_PATTERNS):
        return True
    tokens = re.split(r"[_\-\s]+", feature_lower)
    return any(token in {"0", "1", "yes", "no", "true", "false"} for token in tokens)


def load_known_functional_sites() -> Tuple[pd.DataFrame, set]:
    if not POSITIVE_CSV.exists():
        raise FileNotFoundError(f"Positive file not found: {POSITIVE_CSV}")

    df = pd.read_csv(POSITIVE_CSV)
    df = ensure_index(df)
    df = df.drop_duplicates(subset=[INDEX_COL]).copy()
    df[INDEX_COL] = df[INDEX_COL].astype(str).str.strip()

    known_indices = set(df[INDEX_COL])
    print(f"[INFO] Loaded positive file: {POSITIVE_CSV}")
    print(f"[INFO] Known functional sites: {len(df)}")
    return df, known_indices


def load_new_functional_predictions(known_indices: set) -> pd.DataFrame:
    if not PREDICTION_CSV.exists():
        raise FileNotFoundError(f"Prediction file not found: {PREDICTION_CSV}")

    df = pd.read_csv(PREDICTION_CSV)
    df = ensure_index(df)
    df = df.drop_duplicates(subset=[INDEX_COL]).copy()
    df[INDEX_COL] = df[INDEX_COL].astype(str).str.strip()

    before = len(df)

    if SCORE_COL not in df.columns:
        raise ValueError(f"Missing score column in prediction file: {SCORE_COL}")
    df[SCORE_COL] = pd.to_numeric(df[SCORE_COL], errors="coerce")
    df = df.dropna(subset=[SCORE_COL]).copy()
    df = df[df[SCORE_COL] >= PREDICTION_THRESHOLD].copy()
    selection_rule = f"{SCORE_COL}>={PREDICTION_THRESHOLD:.1f}"

    after_pred = len(df)
    df = df[~df[INDEX_COL].isin(known_indices)].copy()
    removed_known = after_pred - len(df)

    print(f"[INFO] Loaded prediction file: {PREDICTION_CSV}")
    print(f"[INFO] Rows before dedup: {before}")
    print(f"[INFO] Predicted functional ({selection_rule}): {after_pred}")
    print(f"[INFO] Removed known positive sites: {removed_known}")
    print(f"[INFO] Final new functional sites: {len(df)}")
    return df


def load_negative_site_table() -> pd.DataFrame:
    if not NEG_SITE_CSV.exists():
        raise FileNotFoundError(f"Negative file not found: {NEG_SITE_CSV}")

    df = pd.read_csv(NEG_SITE_CSV)
    df = ensure_index(df)
    df = df.drop_duplicates(subset=[INDEX_COL]).copy()
    df[INDEX_COL] = df[INDEX_COL].astype(str).str.strip()
    df[PROTEIN_COL] = df[PROTEIN_COL].astype(str).str.strip()

    print(f"[INFO] Loaded negative file: {NEG_SITE_CSV} | rows={len(df)}")
    return df[[INDEX_COL, PROTEIN_COL]]


def build_shared_negative_index_set(negative_site_df: pd.DataFrame, excluded_indices: set) -> set:
    negative_site_df = negative_site_df.copy()
    excluded_indices = {str(index).strip() for index in excluded_indices if pd.notna(index)}

    before = len(negative_site_df)
    shared_negative_df = negative_site_df[~negative_site_df[INDEX_COL].isin(excluded_indices)].copy()
    after = len(shared_negative_df.drop_duplicates(subset=[INDEX_COL]))

    print("[INFO] Shared negative filtering rule")
    print(f"Negative sites are selected from {NEG_SITE_CSV}.")
    print("Known functional and new predicted functional sites are excluded.")
    print(f"[INFO] Negative before exclusion: {before}")
    print(f"[INFO] Negative after exclusion: {after}")
    print(f"[INFO] Removed from negative: {before - after}")
    return set(shared_negative_df[INDEX_COL].astype(str).str.strip())


def build_sample_group_df(
    known_functional_df: pd.DataFrame,
    new_functional_df: pd.DataFrame,
    shared_negative_index_set: set,
) -> pd.DataFrame:
    parts = []

    for df, group_name in [
        (new_functional_df, "new_functional"),
        (known_functional_df, "known_functional"),
    ]:
        if len(df) == 0:
            continue
        out = df[[INDEX_COL]].copy()
        out[INDEX_COL] = out[INDEX_COL].astype(str).str.strip()
        out["group"] = group_name
        parts.append(out.drop_duplicates())

    neg_df = pd.DataFrame({INDEX_COL: list(shared_negative_index_set)})
    neg_df[INDEX_COL] = neg_df[INDEX_COL].astype(str).str.strip()
    neg_df["group"] = "shared_neg"
    parts.append(neg_df.drop_duplicates())

    sample_df = pd.concat(parts, ignore_index=True)
    return sample_df.drop_duplicates(subset=[INDEX_COL, "group"]).copy()


def load_all_feature_tables() -> Dict[str, pd.DataFrame]:
    tables = {}
    for family_name, file_name in FEATURE_TABLE_FILES.items():
        path = FEATURE_DIR / file_name
        if not path.exists():
            raise FileNotFoundError(f"Feature file not found: {path}")
        df = pd.read_csv(path)
        df = ensure_index(df)
        df = df.drop_duplicates(subset=[INDEX_COL]).copy()
        tables[family_name] = df
        print(f"[INFO] Loaded family={family_name} path={path} rows={len(df)} cols={df.shape[1]}")
    return tables


def format_feature_label(feature_name: str, width: int = 36) -> str:
    return "\n".join(textwrap.wrap(feature_name.replace("_", " "), width=width)[:4])


def pvalue_to_stars(p_value: float) -> str:
    if pd.isna(p_value):
        return "ns"
    if p_value < 1e-4:
        return "****"
    if p_value < 1e-3:
        return "***"
    if p_value < 1e-2:
        return "**"
    if p_value < 5e-2:
        return "*"
    return "ns"


def convert_feature_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(float)
    mapped = series.copy()
    if mapped.dtype == object:
        mapped = mapped.replace(
            {
                "True": 1, "TRUE": 1, "true": 1,
                "False": 0, "FALSE": 0, "false": 0,
                "Yes": 1, "YES": 1, "yes": 1,
                "No": 0, "NO": 0, "no": 0,
            }
        )
    return pd.to_numeric(mapped, errors="coerce")


def get_feature_columns_for_family(df: pd.DataFrame) -> List[str]:
    exclude_cols = {
        INDEX_COL, PROTEIN_COL, "POSITION", "RESIDUE", "MOD_RSD", "FULL_SEQUENCE",
        "Transport_Direction", "LABEL", "gene_symbol", "site_pos", "site_aa",
        "position", "acc_id", "full_sequence", "protein_id", "uniprot_id",
        "UniProt_ID", "Gene", "Gene_Symbol", "Protein", "Entry",
    }
    exclude_lower = {name.lower() for name in exclude_cols}
    return [
        col for col in df.columns
        if col not in exclude_cols and col.lower() not in exclude_lower
    ]


def infer_plot_type(feature_name: str, data_dict: dict) -> str:
    all_values = []
    for group_name in GROUP_ORDER:
        all_values.extend(pd.Series(data_dict.get(group_name, [])).dropna().astype(float).tolist())
    if len(all_values) == 0:
        return "continuous_boxplot"
    values = pd.Series(all_values).dropna().astype(float)
    unique_values = sorted(set(np.round(values.to_numpy(), 8).tolist()))
    is_value_binary = len(unique_values) > 0 and all(value in [0.0, 1.0] for value in unique_values)
    if is_binary_feature_name(feature_name) and is_value_binary:
        return "binary_stackedbar"
    return "continuous_boxplot"


def collect_feature_data_from_family(
    sample_df: pd.DataFrame,
    family_name: str,
    family_df: pd.DataFrame,
) -> Dict[str, dict]:
    family_df = family_df.copy()
    family_df[INDEX_COL] = family_df[INDEX_COL].astype(str).str.strip()
    family_df = family_df.drop_duplicates(subset=[INDEX_COL])
    feature_cols = get_feature_columns_for_family(family_df)
    merged_df = sample_df.merge(family_df, on=INDEX_COL, how="left")

    feature_data = {}
    for feature in feature_cols:
        group_values = {}
        total_valid = 0
        for group_name in GROUP_ORDER:
            raw_values = merged_df.loc[merged_df["group"].eq(group_name), feature]
            values = convert_feature_series(raw_values).dropna().astype(float).tolist()
            group_values[group_name] = values
            total_valid += len(values)
        if total_valid == 0:
            continue
        group_values["family"] = family_name
        group_values["plot_type"] = infer_plot_type(feature, group_values)
        feature_data[feature] = group_values
    return feature_data


def compute_continuous_p_value(group1, group2):
    g1 = pd.Series(group1).dropna().astype(float).tolist()
    g2 = pd.Series(group2).dropna().astype(float).tolist()
    if len(g1) == 0 or len(g2) == 0:
        return np.nan
    try:
        _, p_value = mannwhitneyu(g1, g2, alternative="two-sided")
        return float(p_value)
    except Exception:
        return np.nan


def clean_binary_values(values) -> pd.Series:
    s = convert_feature_series(pd.Series(values)).dropna()
    return s[s.isin([0, 1])].astype(int)


def compute_binary_p_value(group1, group2):
    g1 = clean_binary_values(group1)
    g2 = clean_binary_values(group2)
    if len(g1) == 0 or len(g2) == 0:
        return np.nan
    table = [[int((g1 == 1).sum()), int((g1 == 0).sum())], [int((g2 == 1).sum()), int((g2 == 0).sum())]]
    try:
        _, p_value = fisher_exact(table, alternative="two-sided")
        return float(p_value)
    except Exception:
        return np.nan


def summarize_binary_group(values):
    s = clean_binary_values(values)
    n_total = len(s)
    n_one = int((s == 1).sum())
    n_zero = int((s == 0).sum())
    if n_total == 0:
        return {"n_total": 0, "n_one": 0, "n_zero": 0, "prop_one": np.nan, "prop_zero": np.nan}
    return {
        "n_total": n_total,
        "n_one": n_one,
        "n_zero": n_zero,
        "prop_one": n_one / n_total,
        "prop_zero": n_zero / n_total,
    }


def add_significance_annotation(ax, x1, x2, y, h, text, fontsize=PVALUE_FONTSIZE, lw=1.15):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=lw, c="black")
    ax.text((x1 + x2) * 0.5, y + h + h * 0.10, text, ha="center", va="bottom", fontsize=fontsize)


def draw_single_boxplot(ax, feature_name: str, data_dict: dict):
    box_width = 0.11
    valid_groups, valid_positions, valid_colors = [], [], []

    for group_name in GROUP_ORDER:
        values = pd.Series(data_dict.get(group_name, [])).dropna().astype(float).tolist()
        if len(values) == 0:
            continue
        valid_groups.append(values)
        valid_positions.append(POSITION_MAP_BOX[group_name])
        valid_colors.append(GROUP_COLORS[group_name])

    if len(valid_groups) == 0:
        ax.axis("off")
        return

    box = ax.boxplot(
        valid_groups,
        positions=valid_positions,
        widths=box_width,
        vert=True,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        medianprops={"color": "white", "linewidth": 1.7},
        boxprops={"edgecolor": "black", "linewidth": 0.95},
        whiskerprops={"color": "black", "linewidth": 0.85},
        capprops={"color": "black", "linewidth": 0.85},
    )
    for patch, color in zip(box["boxes"], valid_colors):
        patch.set_facecolor(color)

    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(format_feature_label(feature_name), fontsize=BOX_LABEL_FONTSIZE, pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_linewidth(0.95)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=YTICK_FONTSIZE, pad=2, width=0.85)

    all_values = []
    for group_name in GROUP_ORDER:
        all_values.extend(pd.Series(data_dict.get(group_name, [])).dropna().astype(float).tolist())
    if len(all_values) == 0:
        return

    y_min = float(np.min(all_values))
    y_max = float(np.max(all_values))
    y_range = y_max - y_min if y_max > y_min else max(abs(y_max), 1.0) * 0.1

    line_h = y_range * 0.028
    row_gap = y_range * 0.14
    base_y = y_max + y_range * 0.06
    neg_gap = 0.045
    neg_left_anchor = POSITION_MAP_BOX["shared_neg"] - neg_gap

    p_known = compute_continuous_p_value(data_dict["known_functional"], data_dict["shared_neg"])
    p_new = compute_continuous_p_value(data_dict["new_functional"], data_dict["shared_neg"])

    annotation_specs = [
        ("known_functional", neg_left_anchor, base_y, p_known),
        ("new_functional", neg_left_anchor, base_y + row_gap, p_new),
    ]
    for group_name, neg_anchor, y_pos, p_value in annotation_specs:
        if len(data_dict[group_name]) == 0 or len(data_dict["shared_neg"]) == 0:
            continue
        add_significance_annotation(
            ax,
            POSITION_MAP_BOX[group_name],
            neg_anchor,
            y_pos,
            line_h,
            pvalue_to_stars(p_value),
        )

    ax.set_ylim(y_min - y_range * 0.06, base_y + row_gap + line_h + y_range * 0.18)
    ax.set_xlim(0.56, 1.80)


def draw_single_stackedbar(ax, feature_name: str, data_dict: dict):
    bar_width = 0.13
    for group_name in GROUP_ORDER:
        summary = summarize_binary_group(data_dict.get(group_name, []))
        prop_zero, prop_one = summary["prop_zero"], summary["prop_one"]
        if pd.isna(prop_zero) or pd.isna(prop_one):
            continue
        x = POSITION_MAP_BAR[group_name]
        if group_name == "shared_neg":
            zero_color, one_color = NEG_ZERO_COLOR, NEG_ONE_COLOR
        else:
            zero_color, one_color = POS_ZERO_COLOR, POS_ONE_COLOR
        ax.bar(x, prop_zero, width=bar_width, color=zero_color, edgecolor="black", linewidth=0.8)
        ax.bar(x, prop_one, width=bar_width, bottom=prop_zero, color=one_color, edgecolor="black", linewidth=0.8)

    ax.set_xlim(0.56, 1.84)
    ax.set_ylim(0.0, 1.28)
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax.set_ylabel("Proportion", fontsize=15.5)
    ax.set_xticks([POSITION_MAP_BAR[g] for g in GROUP_ORDER])
    ax.set_xticklabels(["New", "Known", "Negative"], fontsize=14.0)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.tick_params(axis="y", labelsize=YTICK_FONTSIZE, pad=2, width=0.85)
    ax.set_title(format_feature_label(feature_name), fontsize=BOX_LABEL_FONTSIZE, pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    neg_gap = 0.045
    neg_left_anchor = POSITION_MAP_BAR["shared_neg"] - neg_gap
    p_known = compute_binary_p_value(data_dict["known_functional"], data_dict["shared_neg"])
    p_new = compute_binary_p_value(data_dict["new_functional"], data_dict["shared_neg"])

    annotation_specs = [
        ("known_functional", neg_left_anchor, 1.02, p_known),
        ("new_functional", neg_left_anchor, 1.10, p_new),
    ]
    for group_name, neg_anchor, y_pos, p_value in annotation_specs:
        if len(clean_binary_values(data_dict[group_name])) == 0 or len(clean_binary_values(data_dict["shared_neg"])) == 0:
            continue
        add_significance_annotation(
            ax,
            POSITION_MAP_BAR[group_name],
            neg_anchor,
            y_pos,
            0.018,
            pvalue_to_stars(p_value),
        )


def values_are_binary_for_plot(data_dict: dict) -> bool:
    all_values = []
    for group_name in GROUP_ORDER:
        values = convert_feature_series(pd.Series(data_dict.get(group_name, []))).dropna()
        all_values.extend(values.astype(float).tolist())
    if len(all_values) == 0:
        return False
    unique_values = sorted(set(np.round(np.asarray(all_values, dtype=float), 8).tolist()))
    return all(value in [0.0, 1.0] for value in unique_values)


def normalize_feature_plot_item(feature_name: str, data_dict: dict):
    data_dict = dict(data_dict)
    plot_type = data_dict.get("plot_type", "continuous_boxplot")
    is_flag_name = is_binary_feature_name(feature_name)
    is_binary_value = values_are_binary_for_plot(data_dict)

    if plot_type == "binary_stackedbar" or (is_flag_name and is_binary_value):
        data_dict["plot_type"] = "binary_stackedbar"
    else:
        data_dict["plot_type"] = "continuous_boxplot"
    return feature_name, data_dict


def compute_feature_record(feature_name: str, data_dict: dict):
    plot_type = data_dict.get("plot_type", "continuous_boxplot")
    record = {
        "feature": feature_name,
        "family": data_dict.get("family"),
        "plot_type": plot_type,
        "known_functional_n": len(data_dict["known_functional"]),
        "new_functional_n": len(data_dict["new_functional"]),
        "shared_neg_n": len(data_dict["shared_neg"]),
    }

    if plot_type == "binary_stackedbar":
        record["known_functional_vs_shared_neg_p"] = compute_binary_p_value(
            data_dict["known_functional"], data_dict["shared_neg"]
        )
        record["new_functional_vs_shared_neg_p"] = compute_binary_p_value(
            data_dict["new_functional"], data_dict["shared_neg"]
        )
        record["known_vs_new_functional_p"] = compute_binary_p_value(
            data_dict["known_functional"], data_dict["new_functional"]
        )
        for group_name in GROUP_ORDER:
            summary = summarize_binary_group(data_dict[group_name])
            record[f"{group_name}_n_one"] = summary["n_one"]
            record[f"{group_name}_n_zero"] = summary["n_zero"]
            record[f"{group_name}_prop_one"] = summary["prop_one"]
            record[f"{group_name}_prop_zero"] = summary["prop_zero"]
    else:
        record["known_functional_vs_shared_neg_p"] = compute_continuous_p_value(
            data_dict["known_functional"], data_dict["shared_neg"]
        )
        record["new_functional_vs_shared_neg_p"] = compute_continuous_p_value(
            data_dict["new_functional"], data_dict["shared_neg"]
        )
        record["known_vs_new_functional_p"] = compute_continuous_p_value(
            data_dict["known_functional"], data_dict["new_functional"]
        )

    return record


def draw_combined_functional_panel(
    selected_feature_data: Dict[str, dict],
    output_root: Path,
    out_base_name: str = COMBINED_PANEL_BASENAME,
    panel_title: str = "combined",
):
    available_features = [name for name in COMBINED_PANEL_FEATURES if name in selected_feature_data]
    if len(available_features) == 0:
        print(f"[WARNING] No selected features available for {panel_title} panel plotting.")
        return pd.DataFrame()

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 15.5

    n_rows = 3
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.45 * n_cols, 4.65 * n_rows + 1.15), squeeze=False)
    summary_records = []
    row_feature_names: List[List[str]] = [[], [], []]

    for idx, feature_name in enumerate(COMBINED_PANEL_FEATURES):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        if feature_name not in selected_feature_data:
            axes[row_idx, col_idx].axis("off")
            continue

        feature_name, data_dict = normalize_feature_plot_item(
            feature_name, selected_feature_data[feature_name]
        )
        if data_dict["plot_type"] == "binary_stackedbar":
            draw_single_stackedbar(axes[row_idx, col_idx], feature_name, data_dict)
        else:
            draw_single_boxplot(axes[row_idx, col_idx], feature_name, data_dict)
        summary_records.append(compute_feature_record(feature_name, data_dict))
        row_feature_names[row_idx].append(feature_name)

    for idx in range(len(COMBINED_PANEL_FEATURES), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")

    box_legend_handles = [
        Patch(facecolor=NEW_FUNCTIONAL_COLOR, edgecolor="white", label="New Functional"),
        Patch(facecolor=KNOWN_FUNCTIONAL_COLOR, edgecolor="white", label="Known Functional"),
        Patch(facecolor=SHARED_NEG_COLOR, edgecolor="white", label="Negative"),
    ]
    stacked_legend_handles = [
        Patch(facecolor=POS_ZERO_COLOR, edgecolor="white", label="Functional 0"),
        Patch(facecolor=POS_ONE_COLOR, edgecolor="white", label="Functional 1"),
        Patch(facecolor=NEG_ZERO_COLOR, edgecolor="white", label="Negative 0"),
        Patch(facecolor=NEG_ONE_COLOR, edgecolor="white", label="Negative 1"),
    ]

    fig.legend(
        handles=box_legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        bbox_to_anchor=(0.5, 0.048),
        handlelength=1.35,
        columnspacing=1.05,
        handletextpad=0.40,
    )
    fig.legend(
        handles=stacked_legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        bbox_to_anchor=(0.5, 0.008),
        handlelength=1.25,
        columnspacing=0.90,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.045, right=0.995, top=0.965, bottom=0.120, wspace=0.24, hspace=0.36)

    output_root.mkdir(parents=True, exist_ok=True)
    out_base_path = output_root / out_base_name
    png_path = out_base_path.with_suffix(".png")
    pdf_path = out_base_path.with_suffix(".pdf")
    svg_path = out_base_path.with_suffix(".svg")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] Saved {panel_title} panel PNG: {png_path}")
    print(f"[DONE] Saved {panel_title} panel PDF: {pdf_path}")
    print(f"[DONE] Saved {panel_title} panel SVG: {svg_path}")
    print(f"[INFO] {panel_title} panel layout: 3 rows x 3 cols")
    for row_idx, feature_names in enumerate(row_feature_names, start=1):
        print(f"Row {row_idx} features: {feature_names}")

    summary_df = pd.DataFrame(summary_records)
    summary_df["panel"] = panel_title
    summary_df["png_path"] = str(png_path)
    summary_df["pdf_path"] = str(pdf_path)
    summary_df["svg_path"] = str(svg_path)
    return summary_df


def arm_summary_stat(values, plot_type: str):
    if plot_type == "binary_stackedbar":
        s = pd.Series(values).dropna()
        s = s[s.isin([0, 1])].astype(int)
        if len(s) == 0:
            return np.nan
        return float((s == 1).mean())
    s = pd.Series(values).dropna().astype(float)
    if len(s) == 0:
        return np.nan
    return float(s.median())


def arm_known_new_close(known_stat, new_stat, neg_stat):
    if any(pd.isna(v) for v in [known_stat, new_stat, neg_stat]):
        return False
    delta_known = known_stat - neg_stat
    delta_new = new_stat - neg_stat
    if delta_known == 0 or delta_new == 0:
        return False
    if np.sign(delta_known) != np.sign(delta_new):
        return False
    gap_pos = abs(known_stat - new_stat)
    return gap_pos < min(abs(delta_known), abs(delta_new))


def known_vs_neg_significant(data_dict: dict, plot_type: str):
    if plot_type == "binary_stackedbar":
        p_known = compute_binary_p_value(data_dict["known_functional"], data_dict["shared_neg"])
    else:
        p_known = compute_continuous_p_value(data_dict["known_functional"], data_dict["shared_neg"])

    known_stat = arm_summary_stat(data_dict["known_functional"], plot_type)
    new_stat = arm_summary_stat(data_dict["new_functional"], plot_type)
    neg_stat = arm_summary_stat(data_dict["shared_neg"], plot_type)

    known_sig = pd.notna(p_known) and p_known < P_VALUE_THRESHOLD
    close_ok = arm_known_new_close(known_stat, new_stat, neg_stat)
    passed = known_sig and close_ok

    return passed, {
        "known_functional_vs_shared_neg_p": p_known,
        "known_functional_vs_neg_significant": known_sig,
        "known_new_close": close_ok,
    }


def filter_features_known_vs_neg_required(
    selected_features: List[str],
    selected_feature_data: Dict[str, dict],
):
    kept_features = []
    skipped_records = []

    for feature_name in selected_features:
        if feature_name not in selected_feature_data:
            continue

        data_dict = selected_feature_data[feature_name]
        plot_type = data_dict.get("plot_type", "continuous_boxplot")
        passed, info = known_vs_neg_significant(data_dict, plot_type)

        if passed:
            kept_features.append(feature_name)
            continue

        reasons = []
        if not info["known_functional_vs_neg_significant"]:
            reasons.append("known_functional_vs_neg_not_significant")
        if not info["known_new_close"]:
            reasons.append("known_new_not_close")

        skipped_records.append(
            {
                "feature": feature_name,
                "plot_type": plot_type,
                "reason": ";".join(reasons) if reasons else "failed_panel_filter",
                **info,
            }
        )
        print(
            f"[SKIP] {feature_name}: "
            f"known_p={info['known_functional_vs_shared_neg_p']:.4g}, "
            f"known_new_close={info['known_new_close']}"
        )

    kept_data = {name: selected_feature_data[name] for name in kept_features}
    return kept_features, kept_data, pd.DataFrame(skipped_records)


def collect_all_feature_data(sample_df: pd.DataFrame, feature_tables: Dict[str, pd.DataFrame]):
    all_feature_names = []
    all_feature_data = {}

    for family_name, family_df in feature_tables.items():
        feature_data = collect_feature_data_from_family(
            sample_df=sample_df,
            family_name=family_name,
            family_df=family_df,
        )
        for feature_name, data_dict in feature_data.items():
            all_feature_names.append(feature_name)
            all_feature_data[feature_name] = data_dict

    return all_feature_names, all_feature_data


def auto_select_panel_features(all_feature_names: List[str], all_feature_data: Dict[str, dict]):
    selected_names = []
    filter_records = []

    for feature_name in sorted(all_feature_names):
        data_dict = all_feature_data[feature_name]
        plot_type = data_dict.get("plot_type", "continuous_boxplot")
        passed, info = known_vs_neg_significant(data_dict, plot_type)

        record = {
            "feature": feature_name,
            "family": data_dict.get("family"),
            "plot_type": plot_type,
            "selected_for_panel": passed,
            **info,
        }
        filter_records.append(record)

        if passed:
            selected_names.append(feature_name)

    return selected_names, pd.DataFrame(filter_records)


def split_features_for_two_row_panel(selected_features: List[str], selected_feature_data: Dict[str, dict]):
    box_items = []
    flag_items = []

    for feature_name in selected_features:
        if feature_name not in selected_feature_data:
            continue

        data_dict = selected_feature_data[feature_name]
        plot_type = data_dict.get("plot_type", "continuous_boxplot")
        is_flag_name = is_binary_feature_name(feature_name)
        is_binary_value = values_are_binary_for_plot(data_dict)

        if plot_type == "binary_stackedbar" or (is_flag_name and is_binary_value):
            data_dict["plot_type"] = "binary_stackedbar"
            flag_items.append((feature_name, data_dict))
        else:
            data_dict["plot_type"] = "continuous_boxplot"
            box_items.append((feature_name, data_dict))

    return box_items, flag_items


def draw_selected_feature_two_row_panel(
    selected_features: List[str],
    selected_feature_data: Dict[str, dict],
    output_root: Path,
    out_base_name: str = "selected_functional_feature_two_row_box_flag_panel_shared_negative",
    panel_title: str = "manual",
):
    if len(selected_feature_data) == 0:
        print(f"[WARNING] No selected features available for {panel_title} panel plotting.")
        return pd.DataFrame()

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 15.5

    box_items, flag_items = split_features_for_two_row_panel(selected_features, selected_feature_data)
    n_cols = max(len(box_items), len(flag_items), 1)
    box_only = len(flag_items) == 0

    if box_only:
        fig, axes = plt.subplots(1, n_cols, figsize=(4.45 * n_cols, 5.35), squeeze=False)
        plot_axes = axes.reshape(1, n_cols)
        summary_records = []

        for col_idx, (feature_name, data_dict) in enumerate(box_items):
            draw_single_boxplot(plot_axes[0, col_idx], feature_name, data_dict)
            summary_records.append(compute_feature_record(feature_name, data_dict))

        for col_idx in range(len(box_items), n_cols):
            plot_axes[0, col_idx].axis("off")
    else:
        fig, axes = plt.subplots(2, n_cols, figsize=(4.45 * n_cols, 9.25), squeeze=False)
        plot_axes = axes
        summary_records = []

        for col_idx, (feature_name, data_dict) in enumerate(box_items):
            draw_single_boxplot(plot_axes[0, col_idx], feature_name, data_dict)
            summary_records.append(compute_feature_record(feature_name, data_dict))

        for col_idx in range(len(box_items), n_cols):
            plot_axes[0, col_idx].axis("off")

        for col_idx, (feature_name, data_dict) in enumerate(flag_items):
            draw_single_stackedbar(plot_axes[1, col_idx], feature_name, data_dict)
            summary_records.append(compute_feature_record(feature_name, data_dict))

        for col_idx in range(len(flag_items), n_cols):
            plot_axes[1, col_idx].axis("off")

    box_legend_handles = [
        Patch(facecolor=NEW_FUNCTIONAL_COLOR, edgecolor="white", label="New Functional"),
        Patch(facecolor=KNOWN_FUNCTIONAL_COLOR, edgecolor="white", label="Known Functional"),
        Patch(facecolor=SHARED_NEG_COLOR, edgecolor="white", label="Negative"),
    ]

    if box_only:
        fig.legend(
            handles=box_legend_handles,
            loc="lower center",
            ncol=3,
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
            bbox_to_anchor=(0.5, 0.020),
            handlelength=1.35,
            columnspacing=1.05,
            handletextpad=0.40,
        )
        fig.subplots_adjust(left=0.045, right=0.995, top=0.950, bottom=0.120, wspace=0.24)
    else:
        stacked_legend_handles = [
            Patch(facecolor=POS_ZERO_COLOR, edgecolor="white", label="Functional 0"),
            Patch(facecolor=POS_ONE_COLOR, edgecolor="white", label="Functional 1"),
            Patch(facecolor=NEG_ZERO_COLOR, edgecolor="white", label="Negative 0"),
            Patch(facecolor=NEG_ONE_COLOR, edgecolor="white", label="Negative 1"),
        ]

        fig.legend(
            handles=box_legend_handles,
            loc="lower center",
            ncol=3,
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
            bbox_to_anchor=(0.5, 0.060),
            handlelength=1.35,
            columnspacing=1.05,
            handletextpad=0.40,
        )

        fig.legend(
            handles=stacked_legend_handles,
            loc="lower center",
            ncol=4,
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
            bbox_to_anchor=(0.5, 0.020),
            handlelength=1.25,
            columnspacing=0.90,
            handletextpad=0.35,
        )

        fig.subplots_adjust(left=0.045, right=0.995, top=0.950, bottom=0.155, wspace=0.24, hspace=0.42)

    output_root.mkdir(parents=True, exist_ok=True)
    out_base_path = output_root / out_base_name

    png_path = out_base_path.with_suffix(".png")
    pdf_path = out_base_path.with_suffix(".pdf")
    svg_path = out_base_path.with_suffix(".svg")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] Saved {panel_title} panel PNG: {png_path}")
    print(f"[DONE] Saved {panel_title} panel PDF: {pdf_path}")
    print(f"[DONE] Saved {panel_title} panel SVG: {svg_path}")

    summary_df = pd.DataFrame(summary_records)
    summary_df["panel"] = panel_title
    summary_df["png_path"] = str(png_path)
    summary_df["pdf_path"] = str(pdf_path)
    summary_df["svg_path"] = str(svg_path)
    return summary_df


def resolve_panel_feature_data(
    all_feature_data: Dict[str, dict],
    feature_names: List[str],
):
    panel_feature_data = {}
    missing_records = []

    for feature_name in feature_names:
        if feature_name not in all_feature_data:
            missing_records.append(
                {"feature": feature_name, "reason": "feature not found in all feature tables"}
            )
            continue
        panel_feature_data[feature_name] = all_feature_data[feature_name]

    return feature_names, panel_feature_data, pd.DataFrame(missing_records)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    known_functional_df, known_indices = load_known_functional_sites()
    new_functional_df = load_new_functional_predictions(known_indices)
    negative_site_df = load_negative_site_table()

    excluded_indices = (
        set(known_functional_df[INDEX_COL].astype(str).str.strip())
        | set(new_functional_df[INDEX_COL].astype(str).str.strip())
    )
    shared_negative_index_set = build_shared_negative_index_set(
        negative_site_df=negative_site_df,
        excluded_indices=excluded_indices,
    )

    print("[INFO] Group sizes")
    print(f"New Functional: {len(new_functional_df)}")
    print(f"Known Functional: {len(known_functional_df)}")
    print(f"Shared Negative: {len(shared_negative_index_set)}")

    sample_df = build_sample_group_df(
        known_functional_df=known_functional_df,
        new_functional_df=new_functional_df,
        shared_negative_index_set=shared_negative_index_set,
    )
    sample_group_path = OUTPUT_ROOT / "selected_site_groups_shared_negative.csv"
    sample_df.to_csv(sample_group_path, index=False)
    print(f"[DONE] Saved selected site group table to: {sample_group_path}")

    feature_tables = load_all_feature_tables()
    all_feature_names, all_feature_data = collect_all_feature_data(
        sample_df=sample_df,
        feature_tables=feature_tables,
    )

    if SELECTED_FEATURES:
        candidate_features = SELECTED_FEATURES
        candidate_data = {}
        missing_records = []

        for feature_name in candidate_features:
            if feature_name not in all_feature_data:
                missing_records.append(
                    {"feature": feature_name, "reason": "feature not found in all feature tables"}
                )
                continue
            candidate_data[feature_name] = all_feature_data[feature_name]

        panel_features, panel_feature_data, known_neg_skipped_df = filter_features_known_vs_neg_required(
            selected_features=candidate_features,
            selected_feature_data=candidate_data,
        )
        if not APPLY_PANEL_FILTER_ON_MANUAL:
            panel_features = [name for name in candidate_features if name in candidate_data]
            panel_feature_data = {name: candidate_data[name] for name in panel_features}
            print("[INFO] Manual feature list: plotting all available features without panel filter")
        missing_df = pd.DataFrame(missing_records)
        print(f"[INFO] Manual candidate features: {len(SELECTED_FEATURES)}")

        summary_df = draw_selected_feature_two_row_panel(
            selected_features=panel_features,
            selected_feature_data=panel_feature_data,
            output_root=OUTPUT_ROOT,
            panel_title="manual",
        )
    elif MAIN_PANEL_FEATURES or SUPP_PANEL_FEATURES:
        combined_features, combined_feature_data, missing_df = resolve_panel_feature_data(
            all_feature_data=all_feature_data,
            feature_names=COMBINED_PANEL_FEATURES,
        )
        print(f"[INFO] Combined panel features: {len(combined_features)}")

        summary_df = draw_combined_functional_panel(
            selected_feature_data=combined_feature_data,
            output_root=OUTPUT_ROOT,
            panel_title="combined",
        )

        with open(OUTPUT_ROOT / "combined_panel_features.txt", "w", encoding="utf-8") as handle:
            handle.write("\n".join(combined_features) + ("\n" if combined_features else ""))
    else:
        panel_features, filter_summary_df = auto_select_panel_features(
            all_feature_names=all_feature_names,
            all_feature_data=all_feature_data,
        )
        panel_feature_data = {name: all_feature_data[name] for name in panel_features}
        missing_df = pd.DataFrame()
        print(f"[INFO] Auto-selected from all features: {len(all_feature_names)}")

        summary_df = draw_selected_feature_two_row_panel(
            selected_features=panel_features,
            selected_feature_data=panel_feature_data,
            output_root=OUTPUT_ROOT,
            panel_title="auto",
        )

    if SELECTED_FEATURES or (not MAIN_PANEL_FEATURES and not SUPP_PANEL_FEATURES):
        print("[INFO] Panel inclusion rule:")
        print("  1) Known Functional vs Neg, p < 0.05")
        print("  2) Known and New close:")
        print("     |Known-New| < min(|Known-Neg|, |New-Neg|), same trend")

    missing_path = OUTPUT_ROOT / "selected_feature_missing_or_skipped.csv"
    missing_df.to_csv(missing_path, index=False)
    print(f"[DONE] Saved missing or skipped feature table to: {missing_path}")

    filter_summary_path = OUTPUT_ROOT / "all_features_panel_filter_summary.csv"
    if SELECTED_FEATURES and not APPLY_PANEL_FILTER_ON_MANUAL:
        auto_names, auto_filter_df = auto_select_panel_features(
            all_feature_names=all_feature_names,
            all_feature_data=all_feature_data,
        )
        auto_filter_df.to_csv(filter_summary_path, index=False)
        print(f"[INFO] Auto panel filter summary (reference): {len(auto_names)} features would pass")
    elif not SELECTED_FEATURES:
        auto_names, auto_filter_df = auto_select_panel_features(
            all_feature_names=all_feature_names,
            all_feature_data=all_feature_data,
        )
        auto_filter_df.to_csv(filter_summary_path, index=False)
        print(f"[INFO] Auto panel filter summary (reference): {len(auto_names)} features would pass")
    print(f"[DONE] Saved all-features panel filter summary to: {filter_summary_path}")

    summary_path = OUTPUT_ROOT / "selected_feature_panel_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[DONE] Saved selected feature panel summary to: {summary_path}")
    print(f"[DONE] Saved all outputs to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
