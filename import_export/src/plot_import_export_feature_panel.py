# Panels: Figure 3(d); Supplementary Figure 4(d)

"""Draw selected-feature panels for Import/Export site groups (self-contained)."""

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

PREDICTED_IMPORT_CSV = DATA_PRECOMPUTED / "1_transport_classifier_results" / "joint_score" / "predicted_import_stable_gt0p6_vote4.csv"

PREDICTED_EXPORT_CSV = DATA_PRECOMPUTED / "1_transport_classifier_results" / "joint_score" / "predicted_export_stable_gt0p6_vote4.csv"

POSITIVE_CSV = PROJECT_ROOT / "data" / "dataset_phos_site" / "TF_positive_phos_site_0608.csv"

NEG_SITE_CSV = FUNCTIONAL_ROOT / "data" / "dataset_phos_site" / "TF_deepmvp_negative_phos_site_tf_only.csv"

FEATURE_DIR = FUNCTIONAL_ROOT / "data" / "features"

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

OUTPUT_ROOT = PROJECT_ROOT / "results" / "4_1_feature_boxplot_stacked_barplot"

INDEX_COL = "INDEX"
PROTEIN_COL = "ACC_ID"

IMPORT_LABEL = "Nuclear Import"
EXPORT_LABEL = "Nuclear Export"

KNOWN_IMPORT_COLOR = "#9a3f3f"
NEW_IMPORT_COLOR = "#c98585"

KNOWN_EXPORT_COLOR = "#4f7d95"
NEW_EXPORT_COLOR = "#9ab9c9"

SHARED_NEG_COLOR = "#bdbdbd"

IMPORT_ONE_COLOR = "#9f1334"
IMPORT_ZERO_COLOR = "#cfa0aa"
EXPORT_ONE_COLOR = "#0b6497"
EXPORT_ZERO_COLOR = "#8fc1e3"
NEG_ONE_COLOR = "#555555"
NEG_ZERO_COLOR = "#d0d0d0"

BOX_LABEL_FONTSIZE = 12.2
YTICK_FONTSIZE = 10.2
LEGEND_FONTSIZE = 10.6
PVALUE_FONTSIZE = 10.8

GROUP_ORDER = [
    "new_import",
    "known_import",
    "shared_neg",
    "known_export",
    "new_export",
]

GROUP_ORDER_POSITIVE_ONLY = [
    "known_import",
    "known_export",
    "new_import",
    "new_export",
]

POSITION_MAP_BOX_POSITIVE_ONLY = {
    "known_import": 0.82,
    "known_export": 1.02,
    "new_import": 1.38,
    "new_export": 1.58,
}

POSITION_MAP_BAR_POSITIVE_ONLY = {
    "known_import": 0.84,
    "known_export": 1.04,
    "new_import": 1.40,
    "new_export": 1.60,
}

GROUP_DISPLAY_NAMES = {
    "new_import": "New Import",
    "known_import": "Known Import",
    "shared_neg": "Negative",
    "known_export": "Known Export",
    "new_export": "New Export",
}

GROUP_COLORS = {
    "new_import": NEW_IMPORT_COLOR,
    "known_import": KNOWN_IMPORT_COLOR,
    "shared_neg": SHARED_NEG_COLOR,
    "known_export": KNOWN_EXPORT_COLOR,
    "new_export": NEW_EXPORT_COLOR,
}

BINARY_NAME_PATTERNS = [
    "flag",
    "bool",
    "binary",
    "hasany",
]


def is_binary_feature_name(feature_name: str) -> bool:
    feature_lower = str(feature_name).lower()

    if any(pattern in feature_lower for pattern in BINARY_NAME_PATTERNS):
        return True

    tokens = re.split(r"[_\-\s]+", feature_lower)

    if "is" in tokens:
        return True

    return False


def resolve_first_existing_path(path_candidates: List[Path]) -> Path:
    for path in path_candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "None of the candidate files exists:\n"
        + "\n".join(str(path) for path in path_candidates)
    )


def build_index_from_columns(df: pd.DataFrame, acc_col: str, site_col: str) -> pd.Series:
    acc = df[acc_col].astype(str).str.strip()
    pos = pd.to_numeric(df[site_col], errors="coerce").astype("Int64").astype(str)

    if "MOD_RSD" in df.columns:
        mod_rsd = df["MOD_RSD"].astype(str).str.strip()
        valid = mod_rsd.str.match(r"^[A-Za-z][0-9]+$", na=False)
        out = acc + "_" + pos
        out.loc[valid] = acc.loc[valid] + "_" + mod_rsd.loc[valid]
        return out

    if "RESIDUE" in df.columns:
        residue = df["RESIDUE"].astype(str).str.strip()
        valid = residue.str.match(r"^[A-Za-z]$", na=False)
        out = acc + "_" + pos
        out.loc[valid] = acc.loc[valid] + "_" + residue.loc[valid] + pos.loc[valid]
        return out

    return acc + "_" + pos


def ensure_index(df: pd.DataFrame, acc_col: str = "ACC_ID", site_col: str = "POSITION") -> pd.DataFrame:
    df = df.copy()

    if INDEX_COL not in df.columns:
        if acc_col not in df.columns or site_col not in df.columns:
            raise ValueError(f"Missing columns for INDEX construction: {acc_col}, {site_col}")

        df[INDEX_COL] = build_index_from_columns(df, acc_col=acc_col, site_col=site_col)

    df[INDEX_COL] = df[INDEX_COL].astype(str).str.strip()

    if PROTEIN_COL not in df.columns:
        df[PROTEIN_COL] = df[INDEX_COL].astype(str).str.rsplit("_", n=1).str[0]

    df[PROTEIN_COL] = df[PROTEIN_COL].astype(str).str.strip()

    return df


def normalize_direction(value):
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()
    text = text.replace("-", "_").replace(" ", "_")

    import_terms = {
        "nuclear_import",
        "import",
        "promote_import",
        "promote_nuclear_import",
        "inhibit_export",
        "inhibit_nuclear_export",
        "known_import",
        "predicted_import",
    }

    export_terms = {
        "nuclear_export",
        "export",
        "promote_export",
        "promote_nuclear_export",
        "inhibit_import",
        "inhibit_nuclear_import",
        "known_export",
        "predicted_export",
    }

    if text in import_terms:
        return IMPORT_LABEL

    if text in export_terms:
        return EXPORT_LABEL

    if "import" in text and "export" not in text:
        return IMPORT_LABEL

    if "export" in text and "import" not in text:
        return EXPORT_LABEL

    return np.nan


def get_first_existing_direction(df: pd.DataFrame, candidate_cols: List[str]) -> pd.Series:
    direction = pd.Series(np.nan, index=df.index, dtype="object")

    for col in candidate_cols:
        if col not in df.columns:
            continue

        current = df[col].apply(normalize_direction)
        direction = direction.where(direction.notna(), current)

    return direction


def load_known_positive_groups() -> Tuple[pd.DataFrame, pd.DataFrame, set]:
    if not POSITIVE_CSV.exists():
        raise FileNotFoundError(f"Positive file not found: {POSITIVE_CSV}")

    df = pd.read_csv(POSITIVE_CSV)
    df = ensure_index(df)

    direction_cols = [
        "Transport_Direction",
        "LABEL",
        "label",
        "direction",
        "Direction",
    ]

    df["known_direction_norm"] = get_first_existing_direction(df, direction_cols)

    known_import_df = df[df["known_direction_norm"].eq(IMPORT_LABEL)].copy()
    known_export_df = df[df["known_direction_norm"].eq(EXPORT_LABEL)].copy()

    known_import_df = known_import_df.drop_duplicates(subset=[INDEX_COL]).copy()
    known_export_df = known_export_df.drop_duplicates(subset=[INDEX_COL]).copy()

    known_indices = set(
        pd.concat(
            [
                known_import_df[[INDEX_COL]],
                known_export_df[[INDEX_COL]],
            ],
            ignore_index=True,
        )[INDEX_COL].astype(str).str.strip()
    )

    print(f"[INFO] Loaded positive file: {POSITIVE_CSV}")
    print(f"[INFO] Known Import from positive CSV: {len(known_import_df)}")
    print(f"[INFO] Known Export from positive CSV: {len(known_export_df)}")
    print(f"[INFO] Known Import and Export total unique sites: {len(known_indices)}")

    return known_import_df, known_export_df, known_indices


def load_new_prediction_file(path: Path, direction_label: str, known_indices: set) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"New prediction file not found: {path}")

    df = pd.read_csv(path)
    df = ensure_index(df)

    before_dedup = len(df)
    df = df.drop_duplicates(subset=[INDEX_COL]).copy()

    df[INDEX_COL] = df[INDEX_COL].astype(str).str.strip()

    before_known_exclusion = len(df)
    df = df[~df[INDEX_COL].isin(known_indices)].copy()
    removed_known = before_known_exclusion - len(df)

    df["new_direction_norm"] = direction_label

    if direction_label == IMPORT_LABEL and "stable_predicted_import" in df.columns:
        df = df[df["stable_predicted_import"].astype(bool)].copy()

    if direction_label == EXPORT_LABEL and "stable_predicted_export" in df.columns:
        df = df[df["stable_predicted_export"].astype(bool)].copy()

    print(f"[INFO] Loaded new prediction file: {path}")
    print(f"[INFO] Rows before dedup: {before_dedup}")
    print(f"[INFO] Rows after dedup: {before_known_exclusion}")
    print(f"[INFO] Removed known positive sites: {removed_known}")
    print(f"[INFO] Final new sites: {len(df)}")

    return df


def load_prediction_groups() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    known_import_df, known_export_df, known_indices = load_known_positive_groups()

    new_import_df = load_new_prediction_file(
        path=PREDICTED_IMPORT_CSV,
        direction_label=IMPORT_LABEL,
        known_indices=known_indices,
    )

    new_export_df = load_new_prediction_file(
        path=PREDICTED_EXPORT_CSV,
        direction_label=EXPORT_LABEL,
        known_indices=known_indices,
    )

    return known_import_df, new_import_df, known_export_df, new_export_df


def load_negative_site_table() -> pd.DataFrame:
    if not NEG_SITE_CSV.exists():
        raise FileNotFoundError(f"Negative file not found: {NEG_SITE_CSV}")

    df = pd.read_csv(NEG_SITE_CSV)
    df = ensure_index(df)

    required_cols = [INDEX_COL, PROTEIN_COL]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in negative file: {missing}")

    df = df.drop_duplicates(subset=[INDEX_COL]).copy()
    df[INDEX_COL] = df[INDEX_COL].astype(str).str.strip()
    df[PROTEIN_COL] = df[PROTEIN_COL].astype(str).str.strip()

    print(f"[INFO] Loaded negative file: {NEG_SITE_CSV} | rows={len(df)}")

    return df[[INDEX_COL, PROTEIN_COL]]


def build_shared_negative_index_set(negative_site_df: pd.DataFrame,
    excluded_indices,
):
    negative_site_df = negative_site_df.copy()
    negative_site_df[INDEX_COL] = negative_site_df[INDEX_COL].astype(str).str.strip()

    excluded_indices = {
        str(index).strip()
        for index in excluded_indices
        if pd.notna(index)
    }

    before = len(negative_site_df)

    shared_negative_df = negative_site_df[
        ~negative_site_df[INDEX_COL].isin(excluded_indices)
    ].copy()

    shared_negative_df = shared_negative_df.drop_duplicates(subset=[INDEX_COL]).copy()

    after = len(shared_negative_df)

    print("[INFO] Shared negative filtering rule")
    print(f"Negative sites are selected from {NEG_SITE_CSV}.")
    print("Known import, known export, new import, and new export sites are excluded.")
    print(f"[INFO] Negative before exclusion: {before}")
    print(f"[INFO] Negative after exclusion: {after}")
    print(f"[INFO] Removed from negative: {before - after}")

    return set(shared_negative_df[INDEX_COL].astype(str).str.strip())


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

        print(
            f"[INFO] Loaded family={family_name} "
            f"path={path} rows={len(df)} cols={df.shape[1]}"
        )

    return tables


def make_group_df(df: pd.DataFrame, group_name: str) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame(columns=[INDEX_COL, "group"])

    out = df[[INDEX_COL]].copy()
    out[INDEX_COL] = out[INDEX_COL].astype(str).str.strip()
    out["group"] = group_name

    return out.drop_duplicates()


def make_negative_group_df(index_set, group_name: str) -> pd.DataFrame:
    out = pd.DataFrame({INDEX_COL: list(index_set)})
    out[INDEX_COL] = out[INDEX_COL].astype(str).str.strip()
    out["group"] = group_name

    return out.drop_duplicates()


def build_sample_group_df(known_import_df: pd.DataFrame,
    new_import_df: pd.DataFrame,
    known_export_df: pd.DataFrame,
    new_export_df: pd.DataFrame,
    shared_negative_index_set,
) -> pd.DataFrame:
    sample_df = pd.concat(
        [
            make_group_df(new_import_df, "new_import"),
            make_group_df(known_import_df, "known_import"),
            make_negative_group_df(shared_negative_index_set, "shared_neg"),
            make_group_df(known_export_df, "known_export"),
            make_group_df(new_export_df, "new_export"),
        ],
        ignore_index=True,
    )

    sample_df[INDEX_COL] = sample_df[INDEX_COL].astype(str).str.strip()
    sample_df["group"] = sample_df["group"].astype(str).str.strip()

    return sample_df.drop_duplicates(subset=[INDEX_COL, "group"]).copy()


def build_sample_group_df_positive_only(known_import_df: pd.DataFrame,
    new_import_df: pd.DataFrame,
    known_export_df: pd.DataFrame,
    new_export_df: pd.DataFrame,
) -> pd.DataFrame:
    sample_df = pd.concat(
        [
            make_group_df(new_import_df, "new_import"),
            make_group_df(known_import_df, "known_import"),
            make_group_df(known_export_df, "known_export"),
            make_group_df(new_export_df, "new_export"),
        ],
        ignore_index=True,
    )
    sample_df[INDEX_COL] = sample_df[INDEX_COL].astype(str).str.strip()
    sample_df["group"] = sample_df["group"].astype(str).str.strip()
    return sample_df.drop_duplicates(subset=[INDEX_COL, "group"]).copy()


def format_feature_label(feature_name: str, width: int = 36) -> str:
    text = feature_name.replace("_", " ")
    wrapped = textwrap.wrap(text, width=width)

    return "\n".join(wrapped[:4])


def sanitize_filename(text: str, max_len: int = 180) -> str:
    text = str(text)
    text = re.sub(r"[^\w\.\-]+", "_", text)
    text = text.strip("_")

    if len(text) > max_len:
        text = text[:max_len].rstrip("_")

    if len(text) == 0:
        text = "feature"

    return text


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
                "True": 1,
                "TRUE": 1,
                "true": 1,
                "False": 0,
                "FALSE": 0,
                "false": 0,
                "Yes": 1,
                "YES": 1,
                "yes": 1,
                "No": 0,
                "NO": 0,
                "no": 0,
            }
        )

    return pd.to_numeric(mapped, errors="coerce")


def get_feature_columns_for_family(df: pd.DataFrame) -> List[str]:
    exclude_cols = {
        INDEX_COL,
        PROTEIN_COL,
        "POSITION",
        "RESIDUE",
        "MOD_RSD",
        "FULL_SEQUENCE",
        "Transport_Direction",
        "LABEL",
        "gene_symbol",
        "site_pos",
        "site_aa",
        "position",
        "acc_id",
        "full_sequence",
        "protein_id",
        "uniprot_id",
        "UniProt_ID",
        "Gene",
        "Gene_Symbol",
        "Protein",
        "Entry",
    }

    exclude_lower = {name.lower() for name in exclude_cols}
    feature_cols = []

    for col in df.columns:
        if col in exclude_cols:
            continue

        if col.lower() in exclude_lower:
            continue

        feature_cols.append(col)

    return feature_cols


def infer_plot_type(feature_name: str, data_dict: dict) -> str:
    all_values = []

    for group_name in GROUP_ORDER:
        all_values.extend(pd.Series(data_dict.get(group_name, [])).dropna().astype(float).tolist())

    if len(all_values) == 0:
        return "continuous_boxplot"

    values = pd.Series(all_values).dropna().astype(float)
    unique_values = sorted(set(np.round(values.to_numpy(), 8).tolist()))

    is_value_binary = len(unique_values) > 0 and all(value in [0.0, 1.0] for value in unique_values)

    if is_value_binary:
        return "binary_stackedbar"

    return "continuous_boxplot"


def collect_feature_data_from_family(sample_df: pd.DataFrame,
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
    group1 = pd.Series(group1).dropna().astype(float).tolist()
    group2 = pd.Series(group2).dropna().astype(float).tolist()

    if len(group1) == 0 or len(group2) == 0:
        return np.nan

    try:
        _, p_value = mannwhitneyu(group1, group2, alternative="two-sided")
        return float(p_value)
    except Exception:
        return np.nan


def clean_binary_values(values) -> pd.Series:
    s = pd.Series(values).dropna()
    s = convert_feature_series(s).dropna()
    s = s[s.isin([0, 1])].astype(int)

    return s


def compute_binary_p_value(group1, group2):
    g1 = clean_binary_values(group1)
    g2 = clean_binary_values(group2)

    if len(g1) == 0 or len(g2) == 0:
        return np.nan

    g1_one = int((g1 == 1).sum())
    g1_zero = int((g1 == 0).sum())
    g2_one = int((g2 == 1).sum())
    g2_zero = int((g2 == 0).sum())

    table = [[g1_one, g1_zero], [g2_one, g2_zero]]

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
        return {
            "n_total": 0,
            "n_one": 0,
            "n_zero": 0,
            "prop_one": np.nan,
            "prop_zero": np.nan,
        }

    return {
        "n_total": n_total,
        "n_one": n_one,
        "n_zero": n_zero,
        "prop_one": n_one / n_total,
        "prop_zero": n_zero / n_total,
    }


def add_significance_annotation(ax,
    x1,
    x2,
    y,
    h,
    text,
    fontsize=PVALUE_FONTSIZE,
    lw=1.15,
    text_offset_ratio=0.10,
):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=lw, c="black")
    ax.text(
        (x1 + x2) * 0.5,
        y + h + h * text_offset_ratio,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
    )

def draw_single_boxplot(ax, feature_name: str, data_dict: dict):
    position_map = {
        "new_import": 0.70,
        "known_import": 0.88,
        "shared_neg": 1.18,
        "known_export": 1.48,
        "new_export": 1.66,
    }

    box_width = 0.11

    valid_groups = []
    valid_positions = []
    valid_colors = []

    for group_name in GROUP_ORDER:
        values = pd.Series(data_dict.get(group_name, [])).dropna().astype(float).tolist()

        if len(values) == 0:
            continue

        valid_groups.append(values)
        valid_positions.append(position_map[group_name])
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
    y_range = y_max - y_min

    if y_range == 0:
        y_range = max(abs(y_max), 1.0) * 0.1

    line_h = y_range * 0.028
    row_gap = y_range * 0.14
    base_y = y_max + y_range * 0.06

    p_known_import = compute_continuous_p_value(data_dict["known_import"], data_dict["shared_neg"])
    p_new_import = compute_continuous_p_value(data_dict["new_import"], data_dict["shared_neg"])
    p_known_export = compute_continuous_p_value(data_dict["known_export"], data_dict["shared_neg"])
    p_new_export = compute_continuous_p_value(data_dict["new_export"], data_dict["shared_neg"])

    row1_y = base_y
    row2_y = base_y + row_gap
    neg_gap = 0.045
    neg_left_anchor = position_map["shared_neg"] - neg_gap
    neg_right_anchor = position_map["shared_neg"] + neg_gap

    annotation_specs = [
        ("known_import", neg_left_anchor, row1_y, p_known_import),
        ("known_export", neg_right_anchor, row1_y, p_known_export),
        ("new_import", neg_left_anchor, row2_y, p_new_import),
        ("new_export", neg_right_anchor, row2_y, p_new_export),
    ]

    for group1, neg_anchor_x, y_pos, p_value in annotation_specs:
        if len(data_dict[group1]) == 0 or len(data_dict["shared_neg"]) == 0:
            continue

        add_significance_annotation(
            ax,
            position_map[group1],
            neg_anchor_x,
            y_pos,
            line_h,
            pvalue_to_stars(p_value),
        )

    upper_y = row2_y + line_h + y_range * 0.18
    lower_y = y_min - y_range * 0.06

    ax.set_ylim(lower_y, upper_y)
    ax.set_xlim(0.56, 1.80)


def get_binary_colors_for_group(group_name: str):
    if group_name in ["known_import", "new_import"]:
        return IMPORT_ZERO_COLOR, IMPORT_ONE_COLOR

    if group_name in ["known_export", "new_export"]:
        return EXPORT_ZERO_COLOR, EXPORT_ONE_COLOR

    return NEG_ZERO_COLOR, NEG_ONE_COLOR


def draw_single_stackedbar(ax, feature_name: str, data_dict: dict):
    position_map = {
        "new_import": 0.70,
        "known_import": 0.90,
        "shared_neg": 1.20,
        "known_export": 1.50,
        "new_export": 1.70,
    }

    bar_width = 0.13

    for group_name in GROUP_ORDER:
        summary = summarize_binary_group(data_dict.get(group_name, []))

        prop_zero = summary["prop_zero"]
        prop_one = summary["prop_one"]

        if pd.isna(prop_zero) or pd.isna(prop_one):
            continue

        x = position_map[group_name]
        zero_color, one_color = get_binary_colors_for_group(group_name)

        ax.bar(
            x,
            prop_zero,
            width=bar_width,
            color=zero_color,
            edgecolor="black",
            linewidth=0.8,
        )
        ax.bar(
            x,
            prop_one,
            width=bar_width,
            bottom=prop_zero,
            color=one_color,
            edgecolor="black",
            linewidth=0.8,
        )

    ax.set_xlim(0.56, 1.84)
    ax.set_ylim(0.0, 1.28)
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax.set_ylabel("Proportion", fontsize=10.5)

    ax.set_xticks([position_map[group_name] for group_name in GROUP_ORDER])
    ax.set_xticklabels(["New", "Known", "Neg", "Known", "New"], fontsize=9.0)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.tick_params(axis="y", labelsize=YTICK_FONTSIZE, pad=2, width=0.85)

    ax.set_title(format_feature_label(feature_name), fontsize=BOX_LABEL_FONTSIZE, pad=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    p_known_import = compute_binary_p_value(data_dict["known_import"], data_dict["shared_neg"])
    p_new_import = compute_binary_p_value(data_dict["new_import"], data_dict["shared_neg"])
    p_known_export = compute_binary_p_value(data_dict["known_export"], data_dict["shared_neg"])
    p_new_export = compute_binary_p_value(data_dict["new_export"], data_dict["shared_neg"])

    row1_y = 1.02
    row2_y = 1.10
    line_h = 0.018

    neg_gap = 0.045
    neg_left_anchor = position_map["shared_neg"] - neg_gap
    neg_right_anchor = position_map["shared_neg"] + neg_gap

    annotation_specs = [
        ("known_import", neg_left_anchor, row1_y, p_known_import),
        ("known_export", neg_right_anchor, row1_y, p_known_export),
        ("new_import", neg_left_anchor, row2_y, p_new_import),
        ("new_export", neg_right_anchor, row2_y, p_new_export),
    ]

    for group1, neg_anchor_x, y_pos, p_value in annotation_specs:
        if len(clean_binary_values(data_dict[group1])) == 0 or len(clean_binary_values(data_dict["shared_neg"])) == 0:
            continue

        add_significance_annotation(
            ax,
            position_map[group1],
            neg_anchor_x,
            y_pos,
            line_h,
            pvalue_to_stars(p_value),
        )

    ax.text(0.80, -0.18, "Import", ha="center", va="center", fontsize=10.5, transform=ax.transData)
    ax.text(1.20, -0.18, "Negative", ha="center", va="center", fontsize=10.5, transform=ax.transData)
    ax.text(1.60, -0.18, "Export", ha="center", va="center", fontsize=10.5, transform=ax.transData)


def draw_single_boxplot_positive_only(ax, feature_name: str, data_dict: dict):
    position_map = POSITION_MAP_BOX_POSITIVE_ONLY
    box_width = 0.11

    valid_groups = []
    valid_positions = []
    valid_colors = []

    for group_name in GROUP_ORDER_POSITIVE_ONLY:
        values = pd.Series(data_dict.get(group_name, [])).dropna().astype(float).tolist()
        if len(values) == 0:
            continue
        valid_groups.append(values)
        valid_positions.append(position_map[group_name])
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
    for group_name in GROUP_ORDER_POSITIVE_ONLY:
        all_values.extend(pd.Series(data_dict.get(group_name, [])).dropna().astype(float).tolist())
    if len(all_values) == 0:
        return

    y_min = float(np.min(all_values))
    y_max = float(np.max(all_values))
    y_range = y_max - y_min if y_max > y_min else max(abs(y_max), 1.0) * 0.1

    line_h = y_range * 0.028
    base_y = y_max + y_range * 0.06

    p_known_import_vs_known_export = compute_continuous_p_value(
        data_dict["known_import"], data_dict["known_export"]
    )
    p_new_import_vs_new_export = compute_continuous_p_value(
        data_dict["new_import"], data_dict["new_export"]
    )
    if len(data_dict["known_import"]) > 0 and len(data_dict["known_export"]) > 0:
        add_significance_annotation(
            ax,
            position_map["known_import"],
            position_map["known_export"],
            base_y,
            line_h,
            pvalue_to_stars(p_known_import_vs_known_export),
        )
    if len(data_dict["new_import"]) > 0 and len(data_dict["new_export"]) > 0:
        add_significance_annotation(
            ax,
            position_map["new_import"],
            position_map["new_export"],
            base_y,
            line_h,
            pvalue_to_stars(p_new_import_vs_new_export),
        )

    ax.set_ylim(y_min - y_range * 0.06, base_y + line_h + y_range * 0.18)
    ax.set_xlim(0.62, 1.68)


def draw_single_stackedbar_positive_only(ax, feature_name: str, data_dict: dict):
    position_map = POSITION_MAP_BAR_POSITIVE_ONLY
    bar_width = 0.13

    for group_name in GROUP_ORDER_POSITIVE_ONLY:
        summary = summarize_binary_group(data_dict.get(group_name, []))
        prop_zero, prop_one = summary["prop_zero"], summary["prop_one"]
        if pd.isna(prop_zero) or pd.isna(prop_one):
            continue
        x = position_map[group_name]
        zero_color, one_color = get_binary_colors_for_group(group_name)
        ax.bar(x, prop_zero, width=bar_width, color=zero_color, edgecolor="black", linewidth=0.8)
        ax.bar(x, prop_one, width=bar_width, bottom=prop_zero, color=one_color, edgecolor="black", linewidth=0.8)

    ax.set_xlim(0.62, 1.68)
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax.set_ylabel("Proportion", fontsize=10.5)
    ax.set_xticks([position_map[g] for g in GROUP_ORDER_POSITIVE_ONLY])
    ax.set_xticklabels(["Import", "Export", "Import", "Export"], fontsize=9.0)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.tick_params(axis="y", labelsize=YTICK_FONTSIZE, pad=2, width=0.85)
    ax.set_title(format_feature_label(feature_name), fontsize=BOX_LABEL_FONTSIZE, pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    p_known_import_vs_known_export = compute_binary_p_value(
        data_dict["known_import"], data_dict["known_export"]
    )
    p_new_import_vs_new_export = compute_binary_p_value(
        data_dict["new_import"], data_dict["new_export"]
    )
    sig_y = 1.06
    line_h = 0.018
    if (
        len(clean_binary_values(data_dict["known_import"])) > 0
        and len(clean_binary_values(data_dict["known_export"])) > 0
    ):
        add_significance_annotation(
            ax,
            position_map["known_import"],
            position_map["known_export"],
            sig_y,
            line_h,
            pvalue_to_stars(p_known_import_vs_known_export),
        )
    if (
        len(clean_binary_values(data_dict["new_import"])) > 0
        and len(clean_binary_values(data_dict["new_export"])) > 0
    ):
        add_significance_annotation(
            ax,
            position_map["new_import"],
            position_map["new_export"],
            sig_y,
            line_h,
            pvalue_to_stars(p_new_import_vs_new_export),
        )

    ax.set_ylim(0.0, max(1.28, sig_y + line_h + 0.10))

    ax.text(0.93, -0.18, "Known", ha="center", va="center", fontsize=10.5, transform=ax.transData)
    ax.text(1.50, -0.18, "New", ha="center", va="center", fontsize=10.5, transform=ax.transData)


OUTPUT_ROOT = PROJECT_ROOT / "results" / "4_1_feature_boxplot_stacked_barplot" / "importexport_selected_panel_no_negative"

# Exclude shared negative; significance only Known Import vs Known Export.
POSITIVE_ONLY_PANEL = True

# Main figure: 3 features in one row.
MAIN_PANEL_FEATURES: List[str] = [
    "MOTIF_1433_Nearest_Distance",
    "FUNC_Kinase_PWM_MSS_CAMK_Max",
    "MOTIF_Domain_DBD_Within50AA_Flag",
]

# Supplementary figure: 2 x 2 layout.
SUPP_PANEL_ROW1: List[str] = [
    "SEQ_Sequence_Window_AAFrac_Lys",
    "SEQ_Sequence_Window_AAFrac_Ile",
]
SUPP_PANEL_ROW2: List[str] = [
    "SEQ_Sequence_Window_AAFrac_Gln",
    "MOTIF_Domain_Linker_HasSegment_Flag",
]
SUPP_PANEL_FEATURES: List[str] = SUPP_PANEL_ROW1 + SUPP_PANEL_ROW2

# Leave empty when using MAIN_PANEL_FEATURES / SUPP_PANEL_FEATURES split above.
SELECTED_FEATURES: List[str] = []

# When SELECTED_FEATURES is manually set, skip auto filter and plot all listed features.
APPLY_PANEL_FILTER_ON_MANUAL = False

P_VALUE_THRESHOLD = 0.05

FONT_SIZE_SHIFT = 5
LEGEND_FONTSIZE = 10.5 + FONT_SIZE_SHIFT

# Extra space between subplot x-labels and figure legends.
PANEL_BOTTOM_MARGIN_SINGLE = 0.138
PANEL_LEGEND_ROW_HEIGHT = 0.034
PANEL_LEGEND_ANCHOR_LOWER = -0.010
PANEL_LEGEND_ANCHOR_UPPER = PANEL_LEGEND_ANCHOR_LOWER + 2 * PANEL_LEGEND_ROW_HEIGHT
PANEL_BOTTOM_MARGIN_DUAL = 0.228
PANEL_LEGEND_ANCHOR_SINGLE = 0.008
STACKED_GROUP_LABEL_Y = -0.11


def apply_panel_font_style():
    global BOX_LABEL_FONTSIZE, YTICK_FONTSIZE, LEGEND_FONTSIZE, PVALUE_FONTSIZE
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 10.5 + FONT_SIZE_SHIFT
    BOX_LABEL_FONTSIZE = 12.2 + FONT_SIZE_SHIFT
    YTICK_FONTSIZE = 10.2 + FONT_SIZE_SHIFT
    LEGEND_FONTSIZE = 10.6 + FONT_SIZE_SHIFT
    PVALUE_FONTSIZE = 10.8 + FONT_SIZE_SHIFT


def boost_stacked_axis_fonts(ax):
    shift = FONT_SIZE_SHIFT
    ax.yaxis.label.set_fontsize(10.5 + shift)
    for label in ax.get_xticklabels():
        label.set_fontsize(9.0 + shift)
    for text in ax.texts:
        text.set_fontsize(10.5 + shift)
        if text.get_text() in {"Known", "New"}:
            x, _ = text.get_position()
            text.set_position((x, STACKED_GROUP_LABEL_Y))
            text.set_clip_on(False)


def group_order_for_panel(positive_only: bool):
    if positive_only:
        return GROUP_ORDER_POSITIVE_ONLY
    return GROUP_ORDER


def values_are_binary_for_plot(data_dict: dict, positive_only: bool = False):
    all_values = []
    for group_name in group_order_for_panel(positive_only):
        values = convert_feature_series(pd.Series(data_dict.get(group_name, []))).dropna()
        all_values.extend(values.astype(float).tolist())
    if len(all_values) == 0:
        return False
    unique_values = sorted(set(np.round(np.asarray(all_values, dtype=float), 8).tolist()))
    return all(value in [0.0, 1.0] for value in unique_values)


def find_feature_family(feature_tables: Dict[str, pd.DataFrame], feature_name: str, index_col: str):
    for family_name, df in feature_tables.items():
        if feature_name in df.columns:
            return family_name
    return None


def collect_selected_feature_data(sample_df: pd.DataFrame,
    feature_tables: Dict[str, pd.DataFrame],
    selected_features: List[str],
):
    selected_data = {}
    missing_records = []

    for feature_name in selected_features:
        family_name = find_feature_family(feature_tables, feature_name, INDEX_COL)
        if family_name is None:
            print(f"[WARNING] Selected feature not found in feature tables: {feature_name}")
            missing_records.append({"feature": feature_name, "reason": "feature not found in feature tables"})
            continue

        family_df = feature_tables[family_name].copy()
        family_df[INDEX_COL] = family_df[INDEX_COL].astype(str).str.strip()
        family_df = family_df.drop_duplicates(subset=[INDEX_COL])

        merged_df = sample_df.merge(family_df[[INDEX_COL, feature_name]], on=INDEX_COL, how="left")

        group_values = {}
        total_valid = 0
        for group_name in GROUP_ORDER:
            raw_values = merged_df.loc[merged_df["group"].eq(group_name), feature_name]
            values = convert_feature_series(raw_values).dropna().astype(float).tolist()
            group_values[group_name] = values
            total_valid += len(values)

        if total_valid == 0:
            print(f"[WARNING] Selected feature has no numeric values: {feature_name}")
            missing_records.append(
                {"feature": feature_name, "family": family_name, "reason": "no numeric values in selected groups"}
            )
            continue

        group_values["family"] = family_name
        group_values["plot_type"] = infer_plot_type(feature_name, group_values)
        selected_data[feature_name] = group_values

    return selected_data, pd.DataFrame(missing_records)


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
        p_known_import = compute_binary_p_value(
            data_dict["known_import"], data_dict["shared_neg"]
        )
        p_known_export = compute_binary_p_value(
            data_dict["known_export"], data_dict["shared_neg"]
        )
    else:
        p_known_import = compute_continuous_p_value(
            data_dict["known_import"], data_dict["shared_neg"]
        )
        p_known_export = compute_continuous_p_value(
            data_dict["known_export"], data_dict["shared_neg"]
        )

    import_known_sig = pd.notna(p_known_import) and p_known_import < P_VALUE_THRESHOLD
    export_known_sig = pd.notna(p_known_export) and p_known_export < P_VALUE_THRESHOLD

    import_known_stat = arm_summary_stat(data_dict["known_import"], plot_type)
    import_new_stat = arm_summary_stat(data_dict["new_import"], plot_type)
    import_neg_stat = arm_summary_stat(data_dict["shared_neg"], plot_type)
    export_known_stat = arm_summary_stat(data_dict["known_export"], plot_type)
    export_new_stat = arm_summary_stat(data_dict["new_export"], plot_type)
    export_neg_stat = arm_summary_stat(data_dict["shared_neg"], plot_type)

    import_close = arm_known_new_close(import_known_stat, import_new_stat, import_neg_stat)
    export_close = arm_known_new_close(export_known_stat, export_new_stat, export_neg_stat)

    known_sig_any = import_known_sig or export_known_sig
    passed = known_sig_any and import_close and export_close

    passing_arms = []
    if import_known_sig:
        passing_arms.append("import")
    if export_known_sig:
        passing_arms.append("export")

    return passed, passing_arms, {
        "known_import_vs_shared_neg_p": p_known_import,
        "known_export_vs_shared_neg_p": p_known_export,
        "known_import_vs_neg_significant": import_known_sig,
        "known_export_vs_neg_significant": export_known_sig,
        "import_known_new_close": import_close,
        "export_known_new_close": export_close,
        "both_arms_known_new_close": import_close and export_close,
        "passing_arms": ",".join(passing_arms),
    }


def filter_features_known_vs_neg_required(selected_features: List[str],
    selected_feature_data: Dict[str, dict],
):
    kept_features = []
    skipped_records = []

    for feature_name in selected_features:
        if feature_name not in selected_feature_data:
            continue

        data_dict = selected_feature_data[feature_name]
        plot_type = data_dict.get("plot_type", "continuous_boxplot")
        passed, passing_arms, info = known_vs_neg_significant( data_dict, plot_type)

        if passed:
            kept_features.append(feature_name)
            continue

        reasons = []
        if not info["known_import_vs_neg_significant"] and not info["known_export_vs_neg_significant"]:
            reasons.append("neither_known_import_nor_known_export_vs_neg_significant")
        if not info["import_known_new_close"]:
            reasons.append("import_known_new_not_close")
        if not info["export_known_new_close"]:
            reasons.append("export_known_new_not_close")

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
            f"import_known_p={info['known_import_vs_shared_neg_p']:.4g}, "
            f"export_known_p={info['known_export_vs_shared_neg_p']:.4g}, "
            f"passing_arms={info['passing_arms'] or 'none'}"
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
        passed, passing_arms, info = known_vs_neg_significant( data_dict, plot_type)

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


def split_features_for_two_row_panel(selected_features: List[str], selected_feature_data: Dict[str, dict], positive_only: bool = False
):
    box_items = []
    flag_items = []

    for feature_name in selected_features:
        if feature_name not in selected_feature_data:
            continue

        data_dict = selected_feature_data[feature_name]
        plot_type = data_dict.get("plot_type", "continuous_boxplot")
        is_flag_name = is_binary_feature_name(feature_name)
        is_binary_value = values_are_binary_for_plot( data_dict, positive_only=positive_only)

        if plot_type == "binary_stackedbar" or is_binary_value:
            data_dict["plot_type"] = "binary_stackedbar"
            flag_items.append((feature_name, data_dict))
        else:
            data_dict["plot_type"] = "continuous_boxplot"
            box_items.append((feature_name, data_dict))

    return box_items, flag_items


def compute_feature_record(feature_name: str, data_dict: dict, positive_only: bool = False):
    plot_type = data_dict.get("plot_type", "continuous_boxplot")
    record = {
        "feature": feature_name,
        "family": data_dict.get("family"),
        "plot_type": plot_type,
        "new_import_n": len(data_dict["new_import"]),
        "known_import_n": len(data_dict["known_import"]),
        "known_export_n": len(data_dict["known_export"]),
        "new_export_n": len(data_dict["new_export"]),
    }
    if not positive_only:
        record["shared_neg_n"] = len(data_dict.get("shared_neg", []))

    if plot_type == "binary_stackedbar":
        record["known_import_vs_known_export_p"] = compute_binary_p_value(
            data_dict["known_import"], data_dict["known_export"]
        )
        if not positive_only:
            record["known_import_vs_shared_neg_p"] = compute_binary_p_value(
                data_dict["known_import"], data_dict["shared_neg"]
            )
            record["new_import_vs_shared_neg_p"] = compute_binary_p_value(
                data_dict["new_import"], data_dict["shared_neg"]
            )
            record["known_export_vs_shared_neg_p"] = compute_binary_p_value(
                data_dict["known_export"], data_dict["shared_neg"]
            )
            record["new_export_vs_shared_neg_p"] = compute_binary_p_value(
                data_dict["new_export"], data_dict["shared_neg"]
            )
        group_names = group_order_for_panel( positive_only)
        for group_name in group_names:
            summary = summarize_binary_group(data_dict[group_name])
            record[f"{group_name}_n_one"] = summary["n_one"]
            record[f"{group_name}_n_zero"] = summary["n_zero"]
            record[f"{group_name}_prop_one"] = summary["prop_one"]
            record[f"{group_name}_prop_zero"] = summary["prop_zero"]
    else:
        record["known_import_vs_known_export_p"] = compute_continuous_p_value(
            data_dict["known_import"], data_dict["known_export"]
        )
        if not positive_only:
            record["known_import_vs_shared_neg_p"] = compute_continuous_p_value(
                data_dict["known_import"], data_dict["shared_neg"]
            )
            record["new_import_vs_shared_neg_p"] = compute_continuous_p_value(
                data_dict["new_import"], data_dict["shared_neg"]
            )
            record["known_export_vs_shared_neg_p"] = compute_continuous_p_value(
                data_dict["known_export"], data_dict["shared_neg"]
            )
            record["new_export_vs_shared_neg_p"] = compute_continuous_p_value(
                data_dict["new_export"], data_dict["shared_neg"]
            )

    return record


def panel_output_base_name(panel_kind: str, positive_only: bool) -> str:
    suffix = "no_negative" if positive_only else "shared_negative"
    if panel_kind == "main":
        return f"selected_importexport_feature_one_row_mixed_panel_{suffix}"
    return f"supplementary_importexport_feature_two_by_two_panel_{suffix}"


def build_panel_legend_handles(positive_only: bool):
    box_legend_handles = [
        Patch(facecolor=KNOWN_IMPORT_COLOR, edgecolor="white", label="Known Import"),
        Patch(facecolor=KNOWN_EXPORT_COLOR, edgecolor="white", label="Known Export"),
        Patch(facecolor=NEW_IMPORT_COLOR, edgecolor="white", label="New Import"),
        Patch(facecolor=NEW_EXPORT_COLOR, edgecolor="white", label="New Export"),
    ]
    if not positive_only:
        box_legend_handles.insert(
            2, Patch(facecolor=SHARED_NEG_COLOR, edgecolor="white", label="Negative")
        )

    stacked_legend_handles = [
        Patch(facecolor=IMPORT_ZERO_COLOR, edgecolor="white", label="Import 0"),
        Patch(facecolor=IMPORT_ONE_COLOR, edgecolor="white", label="Import 1"),
        Patch(facecolor=EXPORT_ZERO_COLOR, edgecolor="white", label="Export 0"),
        Patch(facecolor=EXPORT_ONE_COLOR, edgecolor="white", label="Export 1"),
    ]
    if not positive_only:
        stacked_legend_handles[2:2] = [
            Patch(facecolor=NEG_ZERO_COLOR, edgecolor="white", label="Negative 0"),
            Patch(facecolor=NEG_ONE_COLOR, edgecolor="white", label="Negative 1"),
        ]
    return box_legend_handles, stacked_legend_handles


def draw_panel_single_row_mixed(selected_features: List[str],
    selected_feature_data: Dict[str, dict],
    output_root: Path,
    out_base_name: str,
    panel_title: str = "main",
    positive_only: bool = False,
):
    """Draw all panel features in one row; each column is boxplot or stacked bar."""
    plot_items = []
    for feature_name in selected_features:
        if feature_name not in selected_feature_data:
            continue
        feature_name, data_dict = normalize_feature_plot_item(
             feature_name, selected_feature_data[feature_name], positive_only=positive_only
        )
        plot_items.append((feature_name, data_dict))

    if len(plot_items) == 0:
        print(f"[WARNING] No selected features available for {panel_title} panel plotting.")
        return pd.DataFrame()

    apply_panel_font_style()

    draw_box = draw_single_boxplot_positive_only if positive_only else draw_single_boxplot
    draw_bar = draw_single_stackedbar_positive_only if positive_only else draw_single_stackedbar

    n_cols = len(plot_items)
    has_flag = any(item[1]["plot_type"] == "binary_stackedbar" for item in plot_items)
    fig_height = 5.85 if has_flag else 5.35

    fig, axes = plt.subplots(1, n_cols, figsize=(4.45 * n_cols, fig_height), squeeze=False)
    plot_axes = axes.reshape(-1)
    summary_records = []

    for col_idx, (feature_name, data_dict) in enumerate(plot_items):
        if data_dict["plot_type"] == "binary_stackedbar":
            draw_bar(plot_axes[col_idx], feature_name, data_dict)
            boost_stacked_axis_fonts(plot_axes[col_idx])
        else:
            draw_box(plot_axes[col_idx], feature_name, data_dict)
        summary_records.append(
            compute_feature_record( feature_name, data_dict, positive_only=positive_only)
        )

    box_legend_handles, stacked_legend_handles = build_panel_legend_handles( positive_only)

    if has_flag:
        fig.legend(
            handles=box_legend_handles,
            loc="lower center",
            ncol=4 if positive_only else 5,
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
            bbox_to_anchor=(0.5, PANEL_LEGEND_ANCHOR_UPPER),
            handlelength=1.35,
            columnspacing=1.05,
            handletextpad=0.40,
        )
        fig.legend(
            handles=stacked_legend_handles,
            loc="lower center",
            ncol=4 if positive_only else 6,
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
            bbox_to_anchor=(0.5, PANEL_LEGEND_ANCHOR_LOWER),
            handlelength=1.25,
            columnspacing=0.90,
            handletextpad=0.35,
        )
        fig.subplots_adjust(left=0.045, right=0.995, top=0.950, bottom=PANEL_BOTTOM_MARGIN_DUAL, wspace=0.24)
    else:
        fig.legend(
            handles=box_legend_handles,
            loc="lower center",
            ncol=4 if positive_only else 5,
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
            bbox_to_anchor=(0.5, PANEL_LEGEND_ANCHOR_SINGLE),
            handlelength=1.35,
            columnspacing=1.05,
            handletextpad=0.40,
        )
        fig.subplots_adjust(left=0.045, right=0.995, top=0.950, bottom=PANEL_BOTTOM_MARGIN_SINGLE, wspace=0.24)

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
    print(f"[INFO] {panel_title} panel layout (single row): {len(plot_items)} features")
    print([feature_name for feature_name, _ in plot_items])

    summary_df = pd.DataFrame(summary_records)
    summary_df["panel"] = panel_title
    summary_df["png_path"] = str(png_path)
    summary_df["pdf_path"] = str(pdf_path)
    summary_df["svg_path"] = str(svg_path)
    return summary_df


def normalize_feature_plot_item(feature_name: str, data_dict: dict, positive_only: bool = False):
    data_dict = dict(data_dict)
    plot_type = data_dict.get("plot_type", "continuous_boxplot")
    is_flag_name = is_binary_feature_name(feature_name)
    is_binary_value = values_are_binary_for_plot( data_dict, positive_only=positive_only)

    if plot_type == "binary_stackedbar" or is_binary_value:
        data_dict["plot_type"] = "binary_stackedbar"
    else:
        data_dict["plot_type"] = "continuous_boxplot"
    return feature_name, data_dict


def draw_supplementary_panel_custom_layout(row1_features: List[str],
    row2_features: List[str],
    selected_feature_data: Dict[str, dict],
    output_root: Path,
    out_base_name: str,
    panel_title: str = "supplementary",
    positive_only: bool = False,
):
    """Draw supplementary panel: row1 left-aligned box; row2 flags; leading columns align."""
    if len(selected_feature_data) == 0:
        print(f"[WARNING] No selected features available for {panel_title} panel plotting.")
        return pd.DataFrame()

    apply_panel_font_style()

    draw_box = draw_single_boxplot_positive_only if positive_only else draw_single_boxplot
    draw_bar = draw_single_stackedbar_positive_only if positive_only else draw_single_stackedbar

    n_cols = max(len(row1_features), len(row2_features), 1)
    fig, axes = plt.subplots(2, n_cols, figsize=(4.45 * n_cols, 9.25), squeeze=False)
    summary_records = []

    for col_idx in range(n_cols):
        if col_idx < len(row1_features):
            feature_name = row1_features[col_idx]
            if feature_name in selected_feature_data:
                feature_name, data_dict = normalize_feature_plot_item(
                     feature_name, selected_feature_data[feature_name], positive_only=positive_only
                )
                draw_box(axes[0, col_idx], feature_name, data_dict)
                summary_records.append(
                    compute_feature_record( feature_name, data_dict, positive_only=positive_only)
                )
            else:
                axes[0, col_idx].axis("off")
        else:
            axes[0, col_idx].axis("off")

    for col_idx in range(n_cols):
        if col_idx < len(row2_features):
            feature_name = row2_features[col_idx]
            if feature_name in selected_feature_data:
                feature_name, data_dict = normalize_feature_plot_item(
                     feature_name, selected_feature_data[feature_name], positive_only=positive_only
                )
                if data_dict["plot_type"] == "binary_stackedbar":
                    draw_bar(axes[1, col_idx], feature_name, data_dict)
                    boost_stacked_axis_fonts(axes[1, col_idx])
                else:
                    draw_box(axes[1, col_idx], feature_name, data_dict)
                summary_records.append(
                    compute_feature_record( feature_name, data_dict, positive_only=positive_only)
                )
            else:
                axes[1, col_idx].axis("off")
        else:
            axes[1, col_idx].axis("off")

    box_legend_handles = [
        Patch(facecolor=KNOWN_IMPORT_COLOR, edgecolor="white", label="Known Import"),
        Patch(facecolor=KNOWN_EXPORT_COLOR, edgecolor="white", label="Known Export"),
        Patch(facecolor=NEW_IMPORT_COLOR, edgecolor="white", label="New Import"),
        Patch(facecolor=NEW_EXPORT_COLOR, edgecolor="white", label="New Export"),
    ]
    if not positive_only:
        box_legend_handles.insert(
            2, Patch(facecolor=SHARED_NEG_COLOR, edgecolor="white", label="Negative")
        )

    stacked_legend_handles = [
        Patch(facecolor=IMPORT_ZERO_COLOR, edgecolor="white", label="Import 0"),
        Patch(facecolor=IMPORT_ONE_COLOR, edgecolor="white", label="Import 1"),
        Patch(facecolor=EXPORT_ZERO_COLOR, edgecolor="white", label="Export 0"),
        Patch(facecolor=EXPORT_ONE_COLOR, edgecolor="white", label="Export 1"),
    ]
    if not positive_only:
        stacked_legend_handles[2:2] = [
            Patch(facecolor=NEG_ZERO_COLOR, edgecolor="white", label="Negative 0"),
            Patch(facecolor=NEG_ONE_COLOR, edgecolor="white", label="Negative 1"),
        ]

    fig.legend(
        handles=box_legend_handles,
        loc="lower center",
        ncol=4 if positive_only else 5,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        bbox_to_anchor=(0.5, PANEL_LEGEND_ANCHOR_UPPER),
        handlelength=1.35,
        columnspacing=1.05,
        handletextpad=0.40,
    )
    fig.legend(
        handles=stacked_legend_handles,
        loc="lower center",
        ncol=4 if positive_only else 6,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        bbox_to_anchor=(0.5, PANEL_LEGEND_ANCHOR_LOWER),
        handlelength=1.25,
        columnspacing=0.90,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.045, right=0.995, top=0.950, bottom=PANEL_BOTTOM_MARGIN_DUAL, wspace=0.24, hspace=0.42)

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
    print(f"[INFO] {panel_title} panel layout")
    print(f"Row 1 features ({len(row1_features)} cols, left-aligned): {row1_features}")
    print(f"Row 2 features ({len(row2_features)} cols): {row2_features}")

    summary_df = pd.DataFrame(summary_records)
    summary_df["panel"] = panel_title
    summary_df["png_path"] = str(png_path)
    summary_df["pdf_path"] = str(pdf_path)
    summary_df["svg_path"] = str(svg_path)
    return summary_df




def draw_supplementary_panel_2x2(
    selected_feature_data: Dict[str, dict],
    output_root: Path,
    out_base_name: str,
    panel_title: str = "supplementary",
    positive_only: bool = False,
):
    """Draw supplementary panel as 2 rows x 2 cols."""
    grid_features = [
        (0, 0, SUPP_PANEL_ROW1[0]),
        (0, 1, SUPP_PANEL_ROW1[1]),
        (1, 0, SUPP_PANEL_ROW2[0]),
        (1, 1, SUPP_PANEL_ROW2[1]),
    ]
    available = [name for _, _, name in grid_features if name in selected_feature_data]
    if len(available) == 0:
        print(f"[WARNING] No selected features available for {panel_title} panel plotting.")
        return pd.DataFrame()

    apply_panel_font_style()
    draw_box = draw_single_boxplot_positive_only if positive_only else draw_single_boxplot
    draw_bar = draw_single_stackedbar_positive_only if positive_only else draw_single_stackedbar

    n_rows, n_cols = 2, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.45 * n_cols, 9.0), squeeze=False)
    summary_records = []
    has_flag = False

    for row_idx, col_idx, feature_name in grid_features:
        ax = axes[row_idx, col_idx]
        if feature_name not in selected_feature_data:
            ax.axis("off")
            continue
        feature_name, data_dict = normalize_feature_plot_item(
            feature_name, selected_feature_data[feature_name], positive_only=positive_only
        )
        if data_dict["plot_type"] == "binary_stackedbar":
            draw_bar(ax, feature_name, data_dict)
            boost_stacked_axis_fonts(ax)
            has_flag = True
        else:
            draw_box(ax, feature_name, data_dict)
        summary_records.append(
            compute_feature_record(feature_name, data_dict, positive_only=positive_only)
        )

    box_legend_handles, stacked_legend_handles = build_panel_legend_handles(positive_only)
    if has_flag:
        fig.legend(
            handles=box_legend_handles,
            loc="lower center",
            ncol=4 if positive_only else 5,
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
            bbox_to_anchor=(0.5, PANEL_LEGEND_ANCHOR_UPPER),
            handlelength=1.35,
            columnspacing=1.05,
            handletextpad=0.40,
        )
        fig.legend(
            handles=stacked_legend_handles,
            loc="lower center",
            ncol=4 if positive_only else 6,
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
            bbox_to_anchor=(0.5, PANEL_LEGEND_ANCHOR_LOWER),
            handlelength=1.25,
            columnspacing=0.90,
            handletextpad=0.35,
        )
        fig.subplots_adjust(left=0.045, right=0.995, top=0.950, bottom=PANEL_BOTTOM_MARGIN_DUAL, wspace=0.24, hspace=0.42)
    else:
        fig.legend(
            handles=box_legend_handles,
            loc="lower center",
            ncol=4 if positive_only else 5,
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
            bbox_to_anchor=(0.5, PANEL_LEGEND_ANCHOR_SINGLE),
            handlelength=1.35,
            columnspacing=1.05,
            handletextpad=0.40,
        )
        fig.subplots_adjust(left=0.045, right=0.995, top=0.950, bottom=PANEL_BOTTOM_MARGIN_SINGLE, wspace=0.24, hspace=0.42)

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
    print(f"[INFO] {panel_title} panel layout: 2 rows x 2 cols")
    print(f"Row 1 features: {SUPP_PANEL_ROW1}")
    print(f"Row 2 features: {SUPP_PANEL_ROW2}")

    summary_df = pd.DataFrame(summary_records)
    summary_df["panel"] = panel_title
    summary_df["png_path"] = str(png_path)
    summary_df["pdf_path"] = str(pdf_path)
    summary_df["svg_path"] = str(svg_path)
    return summary_df

def draw_selected_feature_two_row_panel(selected_features: List[str],
    selected_feature_data: Dict[str, dict],
    output_root: Path,
    out_base_name: str = "selected_importexport_feature_two_row_box_flag_panel_shared_negative",
    panel_title: str = "main",
    positive_only: bool = False,
):
    if len(selected_feature_data) == 0:
        print(f"[WARNING] No selected features available for {panel_title} panel plotting.")
        return pd.DataFrame()

    apply_panel_font_style()

    if positive_only and not out_base_name.endswith("_no_negative"):
        out_base_name = "selected_importexport_feature_two_row_box_flag_panel_no_negative"

    box_items, flag_items = split_features_for_two_row_panel(
         selected_features, selected_feature_data, positive_only=positive_only
    )
    n_cols = max(len(box_items), len(flag_items), 1)
    box_only = len(flag_items) == 0

    draw_box = draw_single_boxplot_positive_only if positive_only else draw_single_boxplot
    draw_bar = draw_single_stackedbar_positive_only if positive_only else draw_single_stackedbar

    box_legend_handles = [
        Patch(facecolor=KNOWN_IMPORT_COLOR, edgecolor="white", label="Known Import"),
        Patch(facecolor=KNOWN_EXPORT_COLOR, edgecolor="white", label="Known Export"),
        Patch(facecolor=NEW_IMPORT_COLOR, edgecolor="white", label="New Import"),
        Patch(facecolor=NEW_EXPORT_COLOR, edgecolor="white", label="New Export"),
    ]
    if not positive_only:
        box_legend_handles.insert(
            2, Patch(facecolor=SHARED_NEG_COLOR, edgecolor="white", label="Negative")
        )

    if box_only:
        fig, axes = plt.subplots(1, n_cols, figsize=(4.45 * n_cols, 5.35), squeeze=False)
        plot_axes = axes.reshape(1, n_cols)
        summary_records = []

        for col_idx, (feature_name, data_dict) in enumerate(box_items):
            draw_box(plot_axes[0, col_idx], feature_name, data_dict)
            summary_records.append(
                compute_feature_record( feature_name, data_dict, positive_only=positive_only)
            )

        for col_idx in range(len(box_items), n_cols):
            plot_axes[0, col_idx].axis("off")

        fig.legend(
            handles=box_legend_handles,
            loc="lower center",
            ncol=4 if positive_only else 5,
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
            bbox_to_anchor=(0.5, PANEL_LEGEND_ANCHOR_SINGLE),
            handlelength=1.35,
            columnspacing=1.05,
            handletextpad=0.40,
        )
        fig.subplots_adjust(left=0.045, right=0.995, top=0.950, bottom=PANEL_BOTTOM_MARGIN_SINGLE, wspace=0.24)
    else:
        fig, axes = plt.subplots(2, n_cols, figsize=(4.45 * n_cols, 9.25), squeeze=False)
        plot_axes = axes
        summary_records = []

        for col_idx, (feature_name, data_dict) in enumerate(box_items):
            draw_box(plot_axes[0, col_idx], feature_name, data_dict)
            summary_records.append(
                compute_feature_record( feature_name, data_dict, positive_only=positive_only)
            )

        for col_idx in range(len(box_items), n_cols):
            plot_axes[0, col_idx].axis("off")

        for col_idx, (feature_name, data_dict) in enumerate(flag_items):
            draw_bar(plot_axes[1, col_idx], feature_name, data_dict)
            boost_stacked_axis_fonts(plot_axes[1, col_idx])
            summary_records.append(
                compute_feature_record( feature_name, data_dict, positive_only=positive_only)
            )

        for col_idx in range(len(flag_items), n_cols):
            plot_axes[1, col_idx].axis("off")

        stacked_legend_handles = [
            Patch(facecolor=IMPORT_ZERO_COLOR, edgecolor="white", label="Import 0"),
            Patch(facecolor=IMPORT_ONE_COLOR, edgecolor="white", label="Import 1"),
            Patch(facecolor=EXPORT_ZERO_COLOR, edgecolor="white", label="Export 0"),
            Patch(facecolor=EXPORT_ONE_COLOR, edgecolor="white", label="Export 1"),
        ]
        if not positive_only:
            stacked_legend_handles[2:2] = [
                Patch(facecolor=NEG_ZERO_COLOR, edgecolor="white", label="Negative 0"),
                Patch(facecolor=NEG_ONE_COLOR, edgecolor="white", label="Negative 1"),
            ]

        fig.legend(
            handles=box_legend_handles,
            loc="lower center",
            ncol=4 if positive_only else 5,
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
            bbox_to_anchor=(0.5, PANEL_LEGEND_ANCHOR_UPPER),
            handlelength=1.35,
            columnspacing=1.05,
            handletextpad=0.40,
        )

        fig.legend(
            handles=stacked_legend_handles,
            loc="lower center",
            ncol=4 if positive_only else 6,
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
            bbox_to_anchor=(0.5, PANEL_LEGEND_ANCHOR_LOWER),
            handlelength=1.25,
            columnspacing=0.90,
            handletextpad=0.35,
        )

        fig.subplots_adjust(left=0.045, right=0.995, top=0.950, bottom=PANEL_BOTTOM_MARGIN_DUAL, wspace=0.24, hspace=0.42)

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
    print(f"[INFO] {panel_title} panel layout")
    print(f"Boxplot row features: {len(box_items)}")
    print([feature_name for feature_name, _ in box_items])
    if not box_only:
        print(f"Stacked barplot row features: {len(flag_items)}")
        print([feature_name for feature_name, _ in flag_items])

    summary_df = pd.DataFrame(summary_records)
    summary_df["panel"] = panel_title
    summary_df["png_path"] = str(png_path)
    summary_df["pdf_path"] = str(pdf_path)
    summary_df["svg_path"] = str(svg_path)
    return summary_df


def resolve_panel_feature_data(all_feature_data: Dict[str, dict],
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

    known_import_df, new_import_df, known_export_df, new_export_df = load_prediction_groups()

    if POSITIVE_ONLY_PANEL:
        sample_df = build_sample_group_df_positive_only(
            known_import_df=known_import_df,
            new_import_df=new_import_df,
            known_export_df=known_export_df,
            new_export_df=new_export_df,
        )
        print("[INFO] Group sizes (positive only, no negative)")
        print(f"New Import: {len(new_import_df)}")
        print(f"Known Import: {len(known_import_df)}")
        print(f"Known Export: {len(known_export_df)}")
        print(f"New Export: {len(new_export_df)}")
        sample_group_path = OUTPUT_ROOT / "selected_site_groups_positive_only.csv"
    else:
        negative_site_df = load_negative_site_table()
        excluded_indices = (
            set(known_import_df[INDEX_COL].astype(str).str.strip())
            | set(new_import_df[INDEX_COL].astype(str).str.strip())
            | set(known_export_df[INDEX_COL].astype(str).str.strip())
            | set(new_export_df[INDEX_COL].astype(str).str.strip())
        )
        shared_negative_index_set = build_shared_negative_index_set(
            negative_site_df=negative_site_df,
            excluded_indices=excluded_indices,
        )
        print("[INFO] Group sizes")
        print(f"New Import: {len(new_import_df)}")
        print(f"Known Import: {len(known_import_df)}")
        print(f"Shared Negative: {len(shared_negative_index_set)}")
        print(f"Known Export: {len(known_export_df)}")
        print(f"New Export: {len(new_export_df)}")
        sample_df = build_sample_group_df(
            known_import_df=known_import_df,
            new_import_df=new_import_df,
            known_export_df=known_export_df,
            new_export_df=new_export_df,
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
            positive_only=POSITIVE_ONLY_PANEL,
        )
    elif MAIN_PANEL_FEATURES or SUPP_PANEL_FEATURES:
        main_features, main_feature_data, main_missing_df = resolve_panel_feature_data(
            all_feature_data=all_feature_data,
            feature_names=MAIN_PANEL_FEATURES,
        )
        supp_features, supp_feature_data, supp_missing_df = resolve_panel_feature_data(
            all_feature_data=all_feature_data,
            feature_names=SUPP_PANEL_FEATURES,
        )
        missing_df = pd.concat([main_missing_df, supp_missing_df], ignore_index=True)
        print(f"[INFO] Main panel features: {len(main_features)}")
        print(f"[INFO] Supplementary panel features: {len(supp_features)}")

        main_summary_df = draw_panel_single_row_mixed(
            selected_features=main_features,
            selected_feature_data=main_feature_data,
            output_root=OUTPUT_ROOT,
            out_base_name=panel_output_base_name("main", POSITIVE_ONLY_PANEL),
            panel_title="main",
            positive_only=POSITIVE_ONLY_PANEL,
        )
        supp_summary_df = draw_supplementary_panel_2x2(
            selected_feature_data=supp_feature_data,
            output_root=OUTPUT_ROOT,
            out_base_name=panel_output_base_name("supplementary", POSITIVE_ONLY_PANEL),
            panel_title="supplementary",
            positive_only=POSITIVE_ONLY_PANEL,
        )
        summary_df = pd.concat([main_summary_df, supp_summary_df], ignore_index=True)

        with open(OUTPUT_ROOT / "main_panel_features.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(main_features) + ("\n" if main_features else ""))
        with open(OUTPUT_ROOT / "supplementary_panel_features.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(supp_features) + ("\n" if supp_features else ""))
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

    missing_path = OUTPUT_ROOT / "selected_feature_missing_or_skipped.csv"
    missing_df.to_csv(missing_path, index=False)
    print(f"[DONE] Saved missing or skipped feature table to: {missing_path}")

    filter_summary_path = OUTPUT_ROOT / "all_features_panel_filter_summary.csv"
    auto_names, auto_filter_df = auto_select_panel_features(
        all_feature_names=all_feature_names,
        all_feature_data=all_feature_data,
    )
    auto_filter_df.to_csv(filter_summary_path, index=False)
    print(f"[INFO] Auto panel filter summary (reference): {len(auto_names)} features would pass")
    print(f"[DONE] Saved all-features panel filter summary to: {filter_summary_path}")

    summary_path = OUTPUT_ROOT / "selected_feature_two_row_panel_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[DONE] Saved selected feature panel summary to: {summary_path}")
    print(f"[DONE] Saved all outputs to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
