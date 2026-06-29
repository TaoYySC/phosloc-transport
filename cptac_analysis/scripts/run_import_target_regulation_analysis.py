#!/usr/bin/env python3
"""
Simplified TEMPO target regulation boxplot pipeline.

For each cancer type and transport direction, this script:
1. Loads stable predicted transport phosphosites from 1_5 joint_score
   (mean_prob_functional >= 0.6, vote >= 4; five-fold functional ensemble).
2. Matches predicted sites to cancer phosphoproteomics data.
3. Maps each site to its TF gene symbol.
4. Selects ChIP-supported target genes for the TF.
5. Splits target genes into activate and repress groups using a signed regulon.
6. Computes the mean RNA expression of each target gene across matched tumor samples.
7. Draws one boxplot per cancer and direction with activate and repress target genes.

Optional TF abundance sensitivity (--confounder-analysis, default full):
- Runs after Figure A/B inside target_logfc_activity_random_analysis/tf_abundance_sensitivity/
- Merges diagnostic / ratio / adjusted metrics back into site_level_processing_summary.csv

Primary TF abundance integration (--abundance-primary-mode):
- unadjusted: original site-abundance split with raw target expression (legacy primary)
- residual: primary target expression / TF activity use residuals after regressing out TF abundance covariates (default: TF protein only)
- dual: run both unadjusted and TF-adjusted primaries; main outputs use adjusted; baseline saved under baseline_unadjusted/
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import logging

logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
import matplotlib
matplotlib.use("Agg")

PLOT_FONT_OFFSET = 6

EDITABLE_VECTOR_FONT_RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "text.usetex": False,
    "pdf.use14corefonts": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 10 + PLOT_FONT_OFFSET,
    "axes.titlesize": 10.5 + PLOT_FONT_OFFSET,
    "axes.labelsize": 9.5 + PLOT_FONT_OFFSET,
    "xtick.labelsize": 8.5 + PLOT_FONT_OFFSET,
    "ytick.labelsize": 8.5 + PLOT_FONT_OFFSET,
    "legend.fontsize": 8.0 + PLOT_FONT_OFFSET,
}
matplotlib.rcParams.update(EDITABLE_VECTOR_FONT_RC)

import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd
import seaborn as sns
from pyensembl import EnsemblRelease
from scipy.stats import mannwhitneyu, spearmanr, t as t_dist, wilcoxon

sns.set_style("white")
matplotlib.rcParams.update(EDITABLE_VECTOR_FONT_RC)

_CPTAC_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _CPTAC_ROOT.parent
_DEFAULT_PIPELINE_RUN = _CPTAC_ROOT / "results" / "import_target_regulation"

DEFAULT_CANCER_LIST = [
    "BRCA", "CCRCC", "COAD", "GBM", "HNSCC",
    "LSCC", "LUAD", "OV", "PDAC", "UCEC",
]
DEFAULT_DIRECTIONS = ["Nuclear Import"]
VALID_DIRECTIONS = ["Nuclear Import", "Nuclear Export"]
VALID_PHOSPHO_SPLIT_MODES = ["quantile_extreme", "observed_missing", "median_nonmissing", "both"]
VALID_CONFOUNDER_ANALYSES = ["none", "diagnostic", "ratio", "adjusted", "full"]
VALID_PHOSPHO_VALUE_MODES = ["site_abundance", "phospho_minus_protein"]
VALID_ABUNDANCE_PRIMARY_MODES = ["unadjusted", "residual", "dual"]
DEFAULT_ADJUSTMENT_COVARIATES = ["tf_protein"]
EXPRESSION_MODE_UNADJUSTED = "unadjusted"
EXPRESSION_MODE_TF_ADJUSTED = "tf_abundance_adjusted"


def _draw_x_axis_tick_marks(
    ax,
    tick_len_axes: float = 0.022 / 3.0,
    color: str = "#333333",
    linewidth: float = 1.0,
) -> List[plt.Line2D]:
    """Draw tick marks on the bottom x-axis spine (outward toward labels)."""
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    xmin, xmax = ax.get_xlim()
    tick_artists: List[plt.Line2D] = []
    for tick in ax.xaxis.get_major_ticks():
        x = float(tick.get_loc())
        if not tick.label1.get_visible():
            continue
        if not str(tick.label1.get_text()).strip():
            continue
        if x < xmin - 1e-9 or x > xmax + 1e-9:
            continue
        line, = ax.plot(
            [x, x],
            [0.0, -tick_len_axes],
            transform=trans,
            color=color,
            linewidth=linewidth,
            clip_on=False,
            solid_capstyle="butt",
            zorder=12,
        )
        tick_artists.append(line)
    return tick_artists


def _set_forest_effect_x_axis(
    ax,
    x_values: Sequence[float],
    left_padding_frac: float = 0.14,
    right_padding_frac: float = 0.14,
) -> None:
    """Set x-limits for forest plots, always including zero when in range."""
    if not x_values:
        return

    from matplotlib.ticker import MaxNLocator

    x_min = float(np.nanmin(x_values))
    x_max = float(np.nanmax(x_values))
    span = x_max - x_min if x_max > x_min else 1.0
    left = min(0.0, x_min - left_padding_frac * span)
    right = x_max + right_padding_frac * span
    ax.set_xlim(left, right)

    ticks = [float(t) for t in MaxNLocator(nbins=6, prune=None).tick_values(left, right)]
    if left <= 0.0 <= right:
        ticks.append(0.0)
    ticks = sorted({round(t, 12) for t in ticks if left - 1e-9 <= t <= right + 1e-9})
    ax.set_xticks(ticks)


def _draw_y_axis_tick_marks(
    ax,
    tick_len_axes: float = 0.022 / 3.0,
    color: str = "#333333",
    linewidth: float = 1.0,
) -> List[plt.Line2D]:
    """Draw tick marks on the left side of the y-axis spine (outward toward labels)."""
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ymin, ymax = ax.get_ylim()
    tick_artists: List[plt.Line2D] = []
    for tick in ax.yaxis.get_major_ticks():
        y = float(tick.get_loc())
        if not tick.label1.get_visible():
            continue
        if not str(tick.label1.get_text()).strip():
            continue
        if y < ymin - 1e-9 or y > ymax + 1e-9:
            continue
        line, = ax.plot(
            [-tick_len_axes, 0.0],
            [y, y],
            transform=trans,
            color=color,
            linewidth=linewidth,
            clip_on=False,
            solid_capstyle="butt",
            zorder=12,
        )
        tick_artists.append(line)
    return tick_artists


def _collect_numeric_axis_tick_bbox_extra(
    ax,
    numeric_x: bool = False,
    numeric_y: bool = True,
    x_pad: float = 10,
    y_pad: float = 10,
    tick_len_axes: float = 0.022 / 3.0,
    linewidth: float = 1.0,
) -> List:
    tick_artists: List[plt.Line2D] = []
    bbox_extra: List = []
    if numeric_y:
        ax.tick_params(axis="y", which="major", length=0, pad=y_pad, color="#333333")
        tick_artists.extend(
            _draw_y_axis_tick_marks(ax, tick_len_axes=tick_len_axes, linewidth=linewidth)
        )
        bbox_extra.extend(list(ax.get_yticklabels()))
        if ax.yaxis.get_label().get_text():
            bbox_extra.append(ax.yaxis.get_label())
    if numeric_x:
        ax.tick_params(axis="x", which="major", length=0, pad=x_pad, color="#333333")
        tick_artists.extend(
            _draw_x_axis_tick_marks(ax, tick_len_axes=tick_len_axes, linewidth=linewidth)
        )
        bbox_extra.extend(list(ax.get_xticklabels()))
        if ax.xaxis.get_label().get_text():
            bbox_extra.append(ax.xaxis.get_label())
    bbox_extra.extend(tick_artists)
    return bbox_extra


def _savefig_with_numeric_ticks(
    fig,
    path,
    ax,
    dpi: int,
    numeric_x: bool = False,
    numeric_y: bool = True,
    pad_inches: float = 0.08,
    **savefig_kwargs,
) -> None:
    bbox_extra = _collect_numeric_axis_tick_bbox_extra(ax, numeric_x=numeric_x, numeric_y=numeric_y)
    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        bbox_extra_artists=bbox_extra,
        pad_inches=pad_inches,
        **savefig_kwargs,
    )


def _zero_aligned_expression_ylim_from_values(
    values: Sequence[float],
    *,
    data_padding_frac: float = 0.06,
    fallback_ylim: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float, float]:
    """Return (y_lower, y_data_upper, y_range) for target-expression boxplots.

    Raw RNA values are non-negative, so the lower limit stays at 0.
    Residual/adjusted expression can be negative; include the full data range then.
    """
    clean = [float(v) for v in values if pd.notna(v)]
    if clean:
        data_min = float(min(clean))
        data_max = float(max(clean))
        value_span = data_max - data_min if data_max > data_min else max(abs(data_max), abs(data_min), 1.0)
        if data_min >= 0:
            y_lower = 0.0
        else:
            y_lower = data_min - data_padding_frac * value_span
        y_data_upper = data_max + data_padding_frac * value_span
    elif fallback_ylim is not None:
        y_lower, y_data_upper = fallback_ylim
        value_span = y_data_upper - y_lower if y_data_upper > y_lower else 1.0
        if y_lower >= 0:
            y_lower = 0.0
    else:
        y_lower, y_data_upper = 0.0, 1.0
        value_span = 1.0
    y_range = y_data_upper - y_lower
    return y_lower, y_data_upper, y_range


def _finalize_zero_aligned_expression_ylim(
    ax,
    y_lower: float,
    y_data_upper: float,
    y_range: float,
    *,
    final_top_padding_frac: float = 0.17,
    final_bottom_padding_frac: float = 0.08,
) -> None:
    """Apply final y limits and tick positions."""
    from matplotlib.ticker import MaxNLocator

    final_ymax = y_data_upper + y_range * final_top_padding_frac
    final_ymin = y_lower
    if y_lower < 0:
        final_ymin = y_lower - y_range * final_bottom_padding_frac
    ax.set_ylim(final_ymin, final_ymax)
    y_ticks = MaxNLocator(nbins=5, prune=None).tick_values(final_ymin, final_ymax)
    ax.set_yticks(y_ticks)


@dataclass
class TempoConfig:
    linkedomics_base: str = str(_CPTAC_ROOT / "data/source/1.cpatac/LinkedOmicsKB")
    chip_dir: str = str(_CPTAC_ROOT / "data/source/4.chipaltas/1.target_genes/targets_5kb")
    signed_regulon_path: str = str(_CPTAC_ROOT / "data/source/5.regulons/CollecTRI_regulons.csv")
    idmapping_path: str = str(_CPTAC_ROOT / "data/source/3.idmapping/HUMAN_9606_idmapping.dat")
    prediction_output_dir: str = str(
        _REPO_ROOT
        / "import_export/data/precomputed/1_transport_classifier_results/joint_score"
    )
    import_prediction_filename: str = "predicted_import_stable_gt0p6_vote4.csv"
    export_prediction_filename: str = "predicted_export_stable_gt0p6_vote4.csv"
    output_dir: str = str(_DEFAULT_PIPELINE_RUN)
    known_positive_path: str = str(
        _REPO_ROOT / "functional/data/dataset_phos_site/TF_positive_phos_site_0608.csv"
    )
    known_site_label_color: str = "#D55E00"
    new_site_label_color: str = "#0072B2"
    ensembl_release: int = 100
    species: str = "human"
    cancer_types: List[str] = field(default_factory=lambda: list(DEFAULT_CANCER_LIST))
    directions: List[str] = field(default_factory=lambda: list(DEFAULT_DIRECTIONS))
    max_missing_ratio: float = 0.8
    min_nonzero_ratio: float = 0.05
    max_nonzero_ratio: float = 0.95
    chip_threshold: float = 200.0
    min_chip_sample_frac: float = 0.1
    chip_top_n: Optional[int] = 500
    signed_target_mode: str = "chip_intersection"
    phospho_group_frac: float = 0.08
    phospho_split_mode: str = "median_nonmissing"
    min_group_samples: int = 3
    min_box_points: int = 3
    exclude_zero_phospho_for_split: bool = False
    dpi: int = 300
    random_iterations: int = 100
    random_seed: int = 42
    max_random_gene_rows_per_group: int = 50000
    bootstrap_iterations: int = 2000
    confounder_analysis: str = "full"
    abundance_primary_mode: str = "residual"
    phospho_value_mode: str = "site_abundance"
    adjustment_covariates: List[str] = field(default_factory=lambda: list(DEFAULT_ADJUSTMENT_COVARIATES))
    purity_column: str = "WES_purity"
    min_samples_for_adjustment: int = 10
    use_bh_pvalue_correction: bool = True

    def cancer_paths(self, cancer_type: str) -> Dict[str, str]:
        base = self.linkedomics_base
        return {
            "phospho": f"{base}/{cancer_type}/{cancer_type}_phospho_site_abundance_log2_reference_intensity_normalized_Tumor.txt",
            "rna": f"{base}/{cancer_type}/{cancer_type}_RNAseq_gene_RSEM_coding_UQ_1500_log2_Tumor.txt",
            "protein": f"{base}/{cancer_type}/{cancer_type}_proteomics_gene_abundance_log2_reference_intensity_normalized_Tumor.txt",
            "phenotype": f"{base}/{cancer_type}/{cancer_type}_phenotype.txt",
        }


class TargetRegulationBoxplotPipeline:
    def __init__(self, config: TempoConfig):
        self.config = config
        self._ensembl_db: Optional[EnsemblRelease] = None
        self._genes_cache: Optional[pd.DataFrame] = None
        self._idmapping_cache: Optional[pd.DataFrame] = None
        self._signed_regulon_cache: Optional[pd.DataFrame] = None
        self._known_positive_cache: Optional[pd.DataFrame] = None
        self._known_site_key_cache: Dict[str, set] = {}

    @staticmethod
    def _normalize_symbol(value: object) -> str:
        if pd.isna(value):
            return ""
        return str(value).strip().upper()

    @staticmethod
    def _direction_short(direction: str) -> str:
        if str(direction).endswith("Import"):
            return "Import"
        if str(direction).endswith("Export"):
            return "Export"
        raise ValueError(f"Unsupported direction: {direction}")

    @staticmethod
    def _direction_label(direction_short: str) -> str:
        direction_short = str(direction_short)
        if direction_short == "Import":
            return "Nuclear Import"
        if direction_short == "Export":
            return "Nuclear Export"
        return direction_short

    @staticmethod
    def _sanitize_filename(value: object) -> str:
        value = str(value)
        for old, new in [
            ("/", "_"), ("\\", "_"), (" ", "_"), ("|", "_"),
            (":", "_"), (";", "_"), (",", "_"), ("(", "_"), (")", "_"),
        ]:
            value = value.replace(old, new)
        while "__" in value:
            value = value.replace("__", "_")
        return value.strip("_")


    @staticmethod
    def _site_display_label_from_df(df: pd.DataFrame) -> pd.Series:
        tf = df["tf_name"].astype(str) if "tf_name" in df.columns else "TF"
        residue = df["RESIDUE"].astype(str) if "RESIDUE" in df.columns else ""
        position = df["POSITION"].astype(str) if "POSITION" in df.columns else ""
        return tf + "_" + residue + position

    @staticmethod
    def _site_identity_key(acc_id: object, residue: object, position: object) -> str:
        if pd.isna(acc_id) or pd.isna(residue) or pd.isna(position):
            return ""
        acc = str(acc_id).strip()
        res = str(residue).strip().upper()
        pos = pd.to_numeric(position, errors="coerce")
        if acc == "" or res == "" or pd.isna(pos):
            return ""
        return f"{acc}_{res}{int(pos)}"

    def load_known_positive_sites(self) -> pd.DataFrame:
        if self._known_positive_cache is not None:
            return self._known_positive_cache.copy()

        path = Path(self.config.known_positive_path)
        if not path.exists() or path.stat().st_size == 0:
            print(f"Warning: known positive site file is unavailable: {path}")
            self._known_positive_cache = pd.DataFrame(columns=["site_identity_key", "direction_short"])
            return self._known_positive_cache.copy()

        df = pd.read_csv(path)
        df = df.loc[:, ~df.columns.duplicated()].copy()

        if "INDEX" in df.columns:
            parsed = df["INDEX"].astype(str).str.extract(
                r"^(?P<idx_acc>.+)_(?P<idx_residue>[A-Za-z])(?P<idx_position>\d+)$"
            )
        else:
            parsed = pd.DataFrame(index=df.index, columns=["idx_acc", "idx_residue", "idx_position"])

        if "ACC_ID" not in df.columns:
            df["ACC_ID"] = parsed["idx_acc"]
        else:
            df["ACC_ID"] = df["ACC_ID"].where(df["ACC_ID"].notna(), parsed["idx_acc"])

        if "RESIDUE" not in df.columns:
            df["RESIDUE"] = parsed["idx_residue"]
        else:
            df["RESIDUE"] = df["RESIDUE"].where(df["RESIDUE"].notna(), parsed["idx_residue"])

        if "POSITION" not in df.columns:
            df["POSITION"] = parsed["idx_position"]
        else:
            df["POSITION"] = df["POSITION"].where(df["POSITION"].notna(), parsed["idx_position"])

        df["ACC_ID"] = df["ACC_ID"].astype(str).str.strip()
        df["RESIDUE"] = df["RESIDUE"].astype(str).str.strip().str.replace(r"\d+", "", regex=True).str.upper()
        df["POSITION"] = pd.to_numeric(df["POSITION"], errors="coerce")
        df = df.dropna(subset=["ACC_ID", "RESIDUE", "POSITION"]).copy()
        df["POSITION"] = df["POSITION"].astype(int)

        if "Transport_Direction" in df.columns:
            direction_text = df["Transport_Direction"].astype(str)
            df["direction_short"] = np.where(
                direction_text.str.contains("Import", case=False, na=False),
                "Import",
                np.where(direction_text.str.contains("Export", case=False, na=False), "Export", ""),
            )
        else:
            df["direction_short"] = ""

        df["site_identity_key"] = [
            self._site_identity_key(acc_id, residue, position)
            for acc_id, residue, position in zip(df["ACC_ID"], df["RESIDUE"], df["POSITION"])
        ]
        df = df[df["site_identity_key"].ne("")].copy()
        self._known_positive_cache = df.drop_duplicates(subset=["site_identity_key", "direction_short"]).reset_index(drop=True)
        return self._known_positive_cache.copy()

    def known_positive_site_keys(self, direction_short: str) -> set:
        direction_short = str(direction_short)
        if direction_short in self._known_site_key_cache:
            return self._known_site_key_cache[direction_short]

        df = self.load_known_positive_sites()
        if df.empty:
            keys = set()
        elif "direction_short" in df.columns and df["direction_short"].astype(str).str.len().gt(0).any():
            sub = df[df["direction_short"].astype(str).eq(direction_short)].copy()
            keys = set(sub["site_identity_key"].dropna().astype(str))
        else:
            keys = set(df["site_identity_key"].dropna().astype(str))

        self._known_site_key_cache[direction_short] = keys
        return keys

    def site_label_status(self, site_sub: pd.DataFrame, direction_short: str) -> str:
        if site_sub.empty:
            return "new_predicted"

        known_keys = self.known_positive_site_keys(direction_short)
        if not known_keys:
            return "new_predicted"

        site_meta = site_sub[["ACC_ID", "RESIDUE", "POSITION"]].drop_duplicates()
        for _, row in site_meta.iterrows():
            key = self._site_identity_key(row.get("ACC_ID"), row.get("RESIDUE"), row.get("POSITION"))
            if key in known_keys:
                return "known_positive"
        return "new_predicted"

    def get_ensembl_db(self) -> EnsemblRelease:
        if self._ensembl_db is None:
            self._ensembl_db = EnsemblRelease(self.config.ensembl_release, self.config.species)
        return self._ensembl_db

    def load_genes(self) -> pd.DataFrame:
        if self._genes_cache is not None:
            return self._genes_cache.copy()

        genes = []
        for gene in self.get_ensembl_db().genes():
            genes.append(
                {
                    "gene_id": gene.gene_id,
                    "gene_name": gene.gene_name,
                    "chrom": gene.contig,
                    "start": gene.start,
                    "end": gene.end,
                    "strand": gene.strand,
                }
            )
        df = pd.DataFrame(genes)
        valid_chroms = [str(i) for i in range(1, 23)] + ["X", "Y"]
        df = df[df["chrom"].isin(valid_chroms)].copy()
        df["gene_name_upper"] = df["gene_name"].map(self._normalize_symbol)
        self._genes_cache = df.reset_index(drop=True)
        return self._genes_cache.copy()

    def load_idmapping(self) -> pd.DataFrame:
        if self._idmapping_cache is not None:
            return self._idmapping_cache.copy()

        df = pd.read_csv(
            self.config.idmapping_path,
            sep="\t",
            names=["ACC_ID", "to", "ENSEMBL_GENE_ID"],
            dtype=str,
        )
        df = df.query("to == 'Ensembl'").copy()
        df["ENSEMBL_GENE_ID"] = df["ENSEMBL_GENE_ID"].str.split(".").str[0]
        df = df.dropna(subset=["ACC_ID", "ENSEMBL_GENE_ID"]).drop_duplicates()
        self._idmapping_cache = df[["ACC_ID", "ENSEMBL_GENE_ID"]].reset_index(drop=True)
        return self._idmapping_cache.copy()

    def load_phospho(self, cancer_type: str) -> pd.DataFrame:
        path = self.config.cancer_paths(cancer_type)["phospho"]
        df = pd.read_csv(path, sep="\t", index_col=0)
        df.index = [p.split(".")[0] + "|" + p.split("|")[2] for p in df.index]
        return df

    def load_rna(self, cancer_type: str) -> pd.DataFrame:
        path = self.config.cancer_paths(cancer_type)["rna"]
        df = pd.read_csv(path, sep="\t", index_col=0)
        df = df.loc[(df == 0).mean(axis=1) < 1].copy()
        df.index = [p.split(".")[0] for p in df.index]
        return df

    def load_protein(self, cancer_type: str) -> pd.DataFrame:
        path = self.config.cancer_paths(cancer_type)["protein"]
        if not Path(path).exists():
            raise FileNotFoundError(f"Protein matrix not found: {path}")
        df = pd.read_csv(path, sep="\t", index_col=0)
        df.index = [p.split(".")[0] for p in df.index]
        return df

    def load_phenotype(self, cancer_type: str) -> pd.DataFrame:
        path = self.config.cancer_paths(cancer_type)["phenotype"]
        if not Path(path).exists():
            return pd.DataFrame()
        return pd.read_csv(path, sep="\t", index_col=0)

    @staticmethod
    def _site_tf_gene_id(site: str) -> str:
        return str(site).split("|")[0]

    @staticmethod
    def _enabled_confounder_modules(confounder_analysis: str) -> set:
        mode = str(confounder_analysis)
        if mode == "none":
            return set()
        if mode == "diagnostic":
            return {"diagnostic"}
        if mode == "ratio":
            return {"ratio"}
        if mode == "adjusted":
            return {"adjusted"}
        if mode == "full":
            return {"diagnostic", "ratio", "adjusted"}
        raise ValueError(
            f"Unsupported confounder_analysis: {confounder_analysis}. "
            f"Choose from {VALID_CONFOUNDER_ANALYSES}"
        )

    @staticmethod
    def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
        aligned = pd.concat([x, y], axis=1).apply(pd.to_numeric, errors="coerce").dropna()
        if len(aligned) < 3:
            return np.nan
        rho, _ = spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
        return float(rho)

    @staticmethod
    def _safe_mannwhitney(a: pd.Series, b: pd.Series) -> float:
        a_clean = pd.to_numeric(a, errors="coerce").dropna()
        b_clean = pd.to_numeric(b, errors="coerce").dropna()
        if len(a_clean) < 3 or len(b_clean) < 3:
            return np.nan
        try:
            return float(mannwhitneyu(a_clean, b_clean, alternative="two-sided").pvalue)
        except ValueError:
            return np.nan

    @staticmethod
    def _fit_ols_coef_pvalue(y: np.ndarray, x_matrix: np.ndarray, coef_idx: int) -> Tuple[float, float]:
        n_samples, n_params = x_matrix.shape
        if n_samples <= n_params:
            return np.nan, np.nan
        if not np.all(np.isfinite(y)) or not np.all(np.isfinite(x_matrix)):
            return np.nan, np.nan
        if np.std(y) == 0:
            return np.nan, np.nan
        for col_idx in range(1, n_params):
            if np.std(x_matrix[:, col_idx]) == 0:
                return np.nan, np.nan
        try:
            beta, _, rank, _ = np.linalg.lstsq(x_matrix, y, rcond=None)
        except np.linalg.LinAlgError:
            return np.nan, np.nan
        if rank < n_params:
            return np.nan, np.nan
        fitted = x_matrix @ beta
        resid = y - fitted
        dof = n_samples - n_params
        if dof <= 0:
            return np.nan, np.nan
        mse = float(np.sum(resid ** 2) / dof)
        try:
            cov = mse * np.linalg.inv(x_matrix.T @ x_matrix)
        except np.linalg.LinAlgError:
            return np.nan, np.nan
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
        if se[coef_idx] <= 0 or not np.isfinite(se[coef_idx]):
            return float(beta[coef_idx]), np.nan
        t_stat = beta[coef_idx] / se[coef_idx]
        p_value = float(2.0 * (1.0 - t_dist.cdf(abs(t_stat), dof)))
        return float(beta[coef_idx]), p_value

    def _gene_abundance_series(
        self,
        gene_id: str,
        df_matrix: pd.DataFrame,
        samples: Sequence[str],
    ) -> pd.Series:
        out = pd.Series(index=list(samples), dtype=float)
        if gene_id not in df_matrix.index:
            return out
        available = [sample for sample in samples if sample in df_matrix.columns]
        if available:
            out.loc[available] = pd.to_numeric(df_matrix.loc[gene_id, available], errors="coerce")
        return out

    def _sample_covariate_series(
        self,
        df_phenotype: pd.DataFrame,
        samples: Sequence[str],
        column: str,
    ) -> pd.Series:
        if df_phenotype.empty or column not in df_phenotype.columns:
            return pd.Series(index=list(samples), dtype=float)
        available = [sample for sample in samples if sample in df_phenotype.index]
        out = pd.Series(index=list(samples), dtype=float)
        if available:
            out.loc[available] = pd.to_numeric(df_phenotype.loc[available, column], errors="coerce")
        return out

    def _transform_phospho_series_for_split(
        self,
        site: str,
        phospho_series: pd.Series,
        df_protein: Optional[pd.DataFrame],
        phospho_value_mode: Optional[str],
    ) -> pd.Series:
        mode = phospho_value_mode or self.config.phospho_value_mode
        if mode == "site_abundance":
            return phospho_series
        if mode != "phospho_minus_protein":
            raise ValueError(f"Unsupported phospho_value_mode: {mode}")
        if df_protein is None:
            raise ValueError("phospho_minus_protein requires a protein abundance matrix")
        tf_gene_id = self._site_tf_gene_id(site)
        protein_series = self._gene_abundance_series(tf_gene_id, df_protein, phospho_series.index)
        return phospho_series - protein_series

    def _needs_protein_matrix(self) -> bool:
        return (
            self.config.phospho_value_mode == "phospho_minus_protein"
            or self.config.abundance_primary_mode in ("residual", "dual")
            or self.config.confounder_analysis not in ("none",)
        )

    def _primary_expression_mode(self) -> str:
        if self.config.abundance_primary_mode == "unadjusted":
            return EXPRESSION_MODE_UNADJUSTED
        return EXPRESSION_MODE_TF_ADJUSTED

    def _expression_modes_to_build(self) -> List[str]:
        if self.config.abundance_primary_mode == "dual":
            return [EXPRESSION_MODE_UNADJUSTED, EXPRESSION_MODE_TF_ADJUSTED]
        if self.config.abundance_primary_mode == "residual":
            return [EXPRESSION_MODE_TF_ADJUSTED]
        return [EXPRESSION_MODE_UNADJUSTED]

    @staticmethod
    def _filter_points_by_expression_mode(
        df_points: pd.DataFrame,
        expression_mode: str,
    ) -> pd.DataFrame:
        if df_points.empty or "expression_mode" not in df_points.columns:
            return df_points
        return df_points.loc[df_points["expression_mode"].astype(str).eq(expression_mode)].copy()

    def _build_tf_adjustment_covariate_frame(
        self,
        tf_gene_id: str,
        df_rna: pd.DataFrame,
        df_protein: Optional[pd.DataFrame],
        samples: Sequence[str],
    ) -> pd.DataFrame:
        sample_list = list(samples)
        covariate_cols: Dict[str, pd.Series] = {}
        for covariate in self.config.adjustment_covariates:
            if covariate == "tf_mrna":
                covariate_cols["tf_mrna"] = pd.to_numeric(
                    self._gene_abundance_series(tf_gene_id, df_rna, sample_list).loc[sample_list],
                    errors="coerce",
                )
            elif covariate == "tf_protein":
                if df_protein is None:
                    return pd.DataFrame(index=sample_list)
                covariate_cols["tf_protein"] = pd.to_numeric(
                    self._gene_abundance_series(tf_gene_id, df_protein, sample_list).loc[sample_list],
                    errors="coerce",
                )
            else:
                raise ValueError(
                    f"Unsupported TF adjustment covariate for residualization: {covariate}"
                )
        if not covariate_cols:
            return pd.DataFrame(index=sample_list)
        return pd.DataFrame(covariate_cols, index=sample_list)

    def _residualize_target_expression(
        self,
        df_rna: pd.DataFrame,
        df_protein: pd.DataFrame,
        gene_ids: Sequence[str],
        tf_gene_id: str,
        samples: Sequence[str],
    ) -> pd.DataFrame:
        sample_list = list(samples)
        covariate_frame = self._build_tf_adjustment_covariate_frame(
            tf_gene_id,
            df_rna,
            df_protein,
            sample_list,
        )
        covariate_names = list(covariate_frame.columns)
        if not covariate_names:
            return pd.DataFrame(index=[], columns=sample_list)

        residual_rows: Dict[str, pd.Series] = {}
        min_samples = max(self.config.min_samples_for_adjustment, 3)

        for gene_id in gene_ids:
            if gene_id not in df_rna.index:
                residual_rows[gene_id] = pd.Series(index=sample_list, dtype=float)
                continue
            y = pd.to_numeric(df_rna.loc[gene_id, sample_list], errors="coerce")
            design = covariate_frame.copy()
            design["y"] = y
            design = design.dropna()
            if len(design) < min_samples:
                residual_rows[gene_id] = pd.Series(index=sample_list, dtype=float)
                continue
            x_matrix = np.column_stack(
                [np.ones(len(design), dtype=float)]
                + [design[col].to_numpy(dtype=float) for col in covariate_names]
            )
            try:
                beta, _, rank, _ = np.linalg.lstsq(x_matrix, design["y"].to_numpy(dtype=float), rcond=None)
            except np.linalg.LinAlgError:
                residual_rows[gene_id] = pd.Series(index=sample_list, dtype=float)
                continue
            if rank < x_matrix.shape[1]:
                residual_rows[gene_id] = pd.Series(index=sample_list, dtype=float)
                continue
            full_design = covariate_frame.copy()
            full_design["y"] = y
            full_design = full_design.dropna()
            if full_design.empty:
                residual_rows[gene_id] = pd.Series(index=sample_list, dtype=float)
                continue
            x_full = np.column_stack(
                [np.ones(len(full_design), dtype=float)]
                + [full_design[col].to_numpy(dtype=float) for col in covariate_names]
            )
            pred = x_full @ beta
            resid = pd.Series(index=sample_list, dtype=float)
            resid.loc[full_design.index.astype(str)] = full_design["y"].to_numpy(dtype=float) - pred
            residual_rows[gene_id] = resid

        if not residual_rows:
            return pd.DataFrame(index=[], columns=sample_list)
        return pd.DataFrame(residual_rows).T

    def prediction_path(self, direction: str) -> Path:
        base = Path(self.config.prediction_output_dir)
        direction_short = self._direction_short(direction)
        if direction_short == "Import":
            return base / self.config.import_prediction_filename
        return base / self.config.export_prediction_filename

    @staticmethod
    def _ensure_direction_score(df: pd.DataFrame) -> pd.DataFrame:
        if "direction_score" in df.columns:
            return df

        model_score_cols = [
            f"direction_score_model_{i}"
            for i in range(1, 6)
            if f"direction_score_model_{i}" in df.columns
        ]
        if model_score_cols:
            df = df.copy()
            df["direction_score"] = df[model_score_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            return df

        raise ValueError("Prediction file must contain direction_score or direction_score_model_1 to direction_score_model_5")

    def load_positive_sites(self, direction: str) -> pd.DataFrame:
        pred_path = self.prediction_path(direction)
        df = pd.read_csv(pred_path)
        if "INDEX" not in df.columns:
            raise ValueError(f"{pred_path} missing INDEX column")

        df = df.loc[:, ~df.columns.duplicated()].copy()
        df = self._ensure_direction_score(df)
        parsed_index = df["INDEX"].astype(str).str.extract(
            r"^(?P<idx_acc>.+)_(?P<idx_residue>[A-Z])(?P<idx_position>\d+)$"
        )

        if "ACC_ID" not in df.columns:
            if "functional_ACC_ID" in df.columns:
                df["ACC_ID"] = df["functional_ACC_ID"]
            elif "known_ACC_ID" in df.columns:
                df["ACC_ID"] = df["known_ACC_ID"]
            else:
                df["ACC_ID"] = parsed_index["idx_acc"]
        else:
            df["ACC_ID"] = df["ACC_ID"].where(df["ACC_ID"].notna(), parsed_index["idx_acc"])

        if "POSITION" not in df.columns:
            df["POSITION"] = parsed_index["idx_position"]
        else:
            df["POSITION"] = df["POSITION"].where(df["POSITION"].notna(), parsed_index["idx_position"])

        if "RESIDUE" not in df.columns:
            df["RESIDUE"] = parsed_index["idx_residue"]
        else:
            df["RESIDUE"] = df["RESIDUE"].where(df["RESIDUE"].notna(), parsed_index["idx_residue"])

        df = df.dropna(subset=["ACC_ID", "RESIDUE", "POSITION"]).copy()
        df["ACC_ID"] = df["ACC_ID"].astype(str).str.strip()
        df["RESIDUE"] = df["RESIDUE"].astype(str).str.strip().str.replace(r"\d+", "", regex=True)
        df["POSITION"] = pd.to_numeric(df["POSITION"], errors="coerce")
        df = df.dropna(subset=["POSITION"]).copy()
        df["POSITION"] = df["POSITION"].astype(int)

        idmap = self.load_idmapping().drop_duplicates("ACC_ID")
        df = df.merge(idmap, on="ACC_ID", how="left")
        df = df.dropna(subset=["ENSEMBL_GENE_ID"]).copy()
        df["ENSEMBL_GENE_ID"] = df["ENSEMBL_GENE_ID"].astype(str).str.split(".").str[0]

        available_genes = set(self.load_genes()["gene_id"])
        df = df[df["ENSEMBL_GENE_ID"].isin(available_genes)].copy()
        df["direction"] = direction
        df["direction_short"] = self._direction_short(direction)
        df["site"] = df["ENSEMBL_GENE_ID"] + "|" + df["RESIDUE"] + df["POSITION"].astype(str)
        df["score"] = pd.to_numeric(df["direction_score"], errors="coerce")

        keep_cols = [
            "INDEX", "ACC_ID", "RESIDUE", "POSITION", "ENSEMBL_GENE_ID",
            "direction", "direction_short", "site", "score",
        ]
        keep_cols.extend([c for c in ["mean_prob_functional", "vote_import", "vote_export"] if c in df.columns])
        return df[keep_cols].drop_duplicates(subset=["site"]).reset_index(drop=True)

    @staticmethod
    def filter_phospho_quality(df_phospho: pd.DataFrame, max_missing_ratio: float) -> pd.DataFrame:
        df_numeric = df_phospho.apply(pd.to_numeric, errors="coerce")
        missing_ratio = df_numeric.isna().mean(axis=1)
        return df_phospho[missing_ratio <= max_missing_ratio]

    @staticmethod
    def find_variable_sites(
        df_phospho: pd.DataFrame,
        sites: Sequence[str],
        min_nonzero_ratio: float,
        max_nonzero_ratio: float,
        split_mode: str,
        min_group_samples: int,
    ) -> pd.DataFrame:
        if len(sites) == 0:
            return pd.DataFrame(
                columns=[
                    "site", "nonzero_ratio", "observed_ratio", "missing_ratio",
                    "n_observed_samples", "n_missing_samples", "median_phospho",
                ]
            )

        split_mode = str(split_mode)
        df_numeric = df_phospho.loc[list(sites)].apply(pd.to_numeric, errors="coerce")
        observed_mask = df_numeric.notna()
        missing_mask = df_numeric.isna()
        nonzero_mask = observed_mask & df_numeric.gt(0)

        stats = pd.DataFrame(index=df_numeric.index)
        stats["site"] = stats.index.astype(str)
        stats["nonzero_ratio"] = nonzero_mask.mean(axis=1)
        stats["observed_ratio"] = observed_mask.mean(axis=1)
        stats["missing_ratio"] = missing_mask.mean(axis=1)
        stats["n_observed_samples"] = observed_mask.sum(axis=1).astype(int)
        stats["n_missing_samples"] = missing_mask.sum(axis=1).astype(int)
        stats["median_phospho"] = df_numeric.median(axis=1, skipna=True)

        if split_mode == "observed_missing":
            mask = (
                stats["n_observed_samples"].ge(min_group_samples)
                & stats["n_missing_samples"].ge(min_group_samples)
            )
        elif split_mode == "median_nonmissing":
            high_counts = []
            low_counts = []
            for _, values in df_numeric.iterrows():
                valid = values.dropna()
                if valid.empty:
                    high_counts.append(0)
                    low_counts.append(0)
                    continue
                median_value = valid.median()
                high_counts.append(int((valid >= median_value).sum()))
                low_counts.append(int((valid < median_value).sum()))
            stats["n_median_high_samples"] = high_counts
            stats["n_median_low_samples"] = low_counts
            mask = (
                stats["n_median_high_samples"].ge(min_group_samples)
                & stats["n_median_low_samples"].ge(min_group_samples)
            )
        elif split_mode == "quantile_extreme":
            mask = (
                stats["nonzero_ratio"].gt(min_nonzero_ratio)
                & stats["nonzero_ratio"].lt(max_nonzero_ratio)
                & stats["n_observed_samples"].ge(2 * min_group_samples)
            )
        else:
            raise ValueError(f"Unsupported phospho split mode: {split_mode}")

        return stats.loc[mask].reset_index(drop=True)

    def map_sites_to_tf(self, df_sites: pd.DataFrame) -> pd.DataFrame:
        genes = self.load_genes()[["gene_id", "gene_name"]].drop_duplicates("gene_id")
        out = df_sites.copy()
        out["tf_gene_id"] = out["site"].astype(str).str.split("|").str[0]
        out = out.merge(genes.rename(columns={"gene_id": "tf_gene_id", "gene_name": "tf_name"}), on="tf_gene_id", how="left")
        out = out.dropna(subset=["tf_name"]).copy()
        return out.reset_index(drop=True)

    def load_chip_targets(self, tf_name: str) -> pd.DataFrame:
        path = Path(self.config.chip_dir) / f"{tf_name}.5.tsv"
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()

        try:
            df = pd.read_csv(path, sep="\t", index_col=0)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
        except pd.errors.ParserError as exc:
            print(f"Warning: could not parse ChIP target file for {tf_name}: {path} ({exc})")
            return pd.DataFrame()
        except OSError as exc:
            print(f"Warning: could not read ChIP target file for {tf_name}: {path} ({exc})")
            return pd.DataFrame()

        df = df.filter(regex="SRX")
        if df.empty or df.shape[1] == 0:
            return pd.DataFrame()
        return df

    def check_chip_available(self, tf_name: str) -> bool:
        path = Path(self.config.chip_dir) / f"{tf_name}.5.tsv"
        if not path.exists() or path.stat().st_size == 0:
            return False

        try:
            df = pd.read_csv(path, sep="\t", index_col=0, nrows=1)
        except pd.errors.EmptyDataError:
            return False
        except pd.errors.ParserError:
            return False
        except OSError:
            return False

        return len(df.filter(regex="SRX").columns) > 0

    def classify_chip_targets(self, df_chip: pd.DataFrame) -> pd.DataFrame:
        if df_chip.empty:
            return pd.DataFrame(columns=["target", "chip_hit_count", "chip_hit_fraction", "chip_mean_score"])

        df_score = df_chip.apply(pd.to_numeric, errors="coerce")
        n_samples = df_score.shape[1]
        min_samples = max(1, int(np.ceil(n_samples * self.config.min_chip_sample_frac)))
        hit_mask = df_score > self.config.chip_threshold

        evidence = pd.DataFrame(index=df_score.index)
        evidence["target"] = evidence.index.astype(str)
        evidence["target_upper"] = evidence["target"].map(self._normalize_symbol)
        evidence["chip_hit_count"] = hit_mask.sum(axis=1).astype(int)
        evidence["chip_hit_fraction"] = evidence["chip_hit_count"] / max(n_samples, 1)
        evidence["chip_mean_score"] = df_score.mean(axis=1, skipna=True)
        evidence["chip_max_score"] = df_score.max(axis=1, skipna=True)
        evidence["chip_median_score"] = df_score.median(axis=1, skipna=True)

        out = evidence[evidence["chip_hit_count"] >= min_samples].copy()
        out = out.sort_values(
            ["chip_hit_count", "chip_hit_fraction", "chip_mean_score", "chip_max_score", "chip_median_score"],
            ascending=[False, False, False, False, False],
        )
        if self.config.chip_top_n is not None and self.config.chip_top_n > 0:
            out = out.head(self.config.chip_top_n)
        return out.reset_index(drop=True)

    @staticmethod
    def _join_unique(values: pd.Series) -> str:
        out: List[str] = []
        for value in values.dropna().astype(str):
            for part in value.split(";"):
                part = part.strip()
                if part and part not in out:
                    out.append(part)
        return ";".join(out)

    def load_signed_regulons(self) -> pd.DataFrame:
        if self._signed_regulon_cache is not None:
            return self._signed_regulon_cache.copy()

        path = Path(self.config.signed_regulon_path)
        if not path.exists():
            raise FileNotFoundError(f"Signed regulon file does not exist: {path}")

        df = pd.read_csv(path)
        required_cols = {"source", "target", "weight"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Signed regulon file missing columns: {sorted(missing_cols)}")

        for col in ["resources", "references", "sign_decision"]:
            if col not in df.columns:
                df[col] = ""

        df = df.copy()
        df["source"] = df["source"].astype(str).str.strip()
        df["target"] = df["target"].astype(str).str.strip()
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
        df = df.dropna(subset=["source", "target", "weight"])
        df = df[df["weight"].ne(0)].copy()
        df["source_upper"] = df["source"].map(self._normalize_symbol)
        df["target_upper"] = df["target"].map(self._normalize_symbol)
        df["weight_sign"] = np.sign(df["weight"]).astype(int)
        df = df[df["source_upper"].ne("") & df["target_upper"].ne("")].copy()
        self._signed_regulon_cache = df.reset_index(drop=True)
        return self._signed_regulon_cache.copy()

    def get_signed_targets_for_tf(self, tf_name: str, chip_targets: Optional[Sequence[str]]) -> pd.DataFrame:
        regulon = self.load_signed_regulons()
        tf_upper = self._normalize_symbol(tf_name)
        sub = regulon[regulon["source_upper"].eq(tf_upper)].copy()

        if chip_targets is not None:
            chip_target_upper = {self._normalize_symbol(g) for g in chip_targets}
            sub = sub[sub["target_upper"].isin(chip_target_upper)].copy()

        if sub.empty:
            return pd.DataFrame(
                columns=[
                    "target", "target_upper", "weight_mean", "target_mor", "target_regulation",
                    "n_regulon_edges", "has_conflicting_sign", "resources", "references", "sign_decision",
                ]
            )

        grouped = (
            sub.groupby(["source_upper", "target_upper"], as_index=False)
            .agg(
                source=("source", "first"),
                target=("target", "first"),
                weight_mean=("weight", "mean"),
                n_regulon_edges=("weight", "count"),
                n_unique_signs=("weight_sign", "nunique"),
                resources=("resources", self._join_unique),
                references=("references", self._join_unique),
                sign_decision=("sign_decision", self._join_unique),
            )
        )
        grouped["target_mor"] = np.sign(grouped["weight_mean"]).astype(int)
        grouped = grouped[grouped["target_mor"].ne(0)].copy()
        grouped["target_regulation"] = np.where(grouped["target_mor"].gt(0), "activate", "repress")
        grouped["has_conflicting_sign"] = grouped["n_unique_signs"] > 1
        return grouped.reset_index(drop=True)


    def _get_phospho_high_low_samples(
        self,
        site: str,
        df_phospho: pd.DataFrame,
        df_rna: pd.DataFrame,
        df_protein: Optional[pd.DataFrame] = None,
        phospho_value_mode: Optional[str] = None,
    ) -> Dict[str, object]:
        value_mode = phospho_value_mode or self.config.phospho_value_mode
        matched_samples = [s for s in df_rna.columns if s in df_phospho.columns]
        if len(matched_samples) == 0:
            matched_samples = list(df_rna.columns)

        base = {
            "low_samples": [],
            "high_samples": [],
            "n_matched_samples": len(matched_samples),
            "n_valid_phospho_samples": 0,
            "n_missing_phospho_samples": 0,
            "n_low_samples": 0,
            "n_high_samples": 0,
            "low_cutoff": np.nan,
            "high_cutoff": np.nan,
            "median_cutoff": np.nan,
            "low_mean_phospho": np.nan,
            "high_mean_phospho": np.nan,
            "phospho_split_mode": self.config.phospho_split_mode,
            "phospho_value_mode": value_mode,
        }

        if site not in df_phospho.index:
            out = dict(base)
            out["status"] = "site_not_in_phospho_matrix"
            return out

        phospho_series = pd.to_numeric(df_phospho.loc[site, matched_samples], errors="coerce")
        try:
            phospho_series = self._transform_phospho_series_for_split(
                site,
                phospho_series,
                df_protein,
                value_mode,
            )
        except ValueError as exc:
            out = dict(base)
            out["status"] = f"phospho_transform_error: {exc}"
            return out
        observed_values = phospho_series.dropna()
        missing_samples = phospho_series.index[phospho_series.isna()].tolist()

        if self.config.phospho_split_mode == "observed_missing":
            high_values = observed_values.copy()
            if self.config.exclude_zero_phospho_for_split:
                high_values = high_values[high_values > 0]
            low_samples = list(missing_samples)
            n_high = len(high_values)
            n_low = len(low_samples)
            if n_high < self.config.min_group_samples or n_low < self.config.min_group_samples:
                out = dict(base)
                out.update(
                    {
                        "status": "insufficient_observed_or_missing_samples",
                        "n_valid_phospho_samples": int(len(observed_values)),
                        "n_missing_phospho_samples": int(len(missing_samples)),
                        "n_high_samples": int(n_high),
                        "n_low_samples": int(n_low),
                    }
                )
                return out
            return {
                "status": "success",
                "low_samples": low_samples,
                "high_samples": high_values.index.tolist(),
                "n_matched_samples": len(matched_samples),
                "n_valid_phospho_samples": int(len(observed_values)),
                "n_missing_phospho_samples": int(len(missing_samples)),
                "n_low_samples": int(n_low),
                "n_high_samples": int(n_high),
                "low_cutoff": np.nan,
                "high_cutoff": np.nan,
                "median_cutoff": np.nan,
                "low_mean_phospho": np.nan,
                "high_mean_phospho": float(high_values.mean()) if len(high_values) else np.nan,
                "phospho_split_mode": self.config.phospho_split_mode,
            }

        phospho_values = observed_values.copy()
        if self.config.exclude_zero_phospho_for_split:
            phospho_values = phospho_values[phospho_values > 0]

        n_valid = len(phospho_values)
        if n_valid < 2 * self.config.min_group_samples:
            out = dict(base)
            out.update(
                {
                    "status": "insufficient_valid_phospho_samples",
                    "n_valid_phospho_samples": int(n_valid),
                    "n_missing_phospho_samples": int(len(missing_samples)),
                }
            )
            return out

        if self.config.phospho_split_mode == "median_nonmissing":
            median_value = float(phospho_values.median())
            low_values = phospho_values[phospho_values < median_value]
            high_values = phospho_values[phospho_values >= median_value]

            if len(low_values) < self.config.min_group_samples or len(high_values) < self.config.min_group_samples:
                out = dict(base)
                out.update(
                    {
                        "status": "insufficient_median_split_samples",
                        "n_valid_phospho_samples": int(n_valid),
                        "n_missing_phospho_samples": int(len(missing_samples)),
                        "n_low_samples": int(len(low_values)),
                        "n_high_samples": int(len(high_values)),
                        "median_cutoff": median_value,
                    }
                )
                return out

            return {
                "status": "success",
                "low_samples": low_values.index.tolist(),
                "high_samples": high_values.index.tolist(),
                "n_matched_samples": len(matched_samples),
                "n_valid_phospho_samples": int(n_valid),
                "n_missing_phospho_samples": int(len(missing_samples)),
                "n_low_samples": int(len(low_values)),
                "n_high_samples": int(len(high_values)),
                "low_cutoff": float(low_values.max()) if len(low_values) else np.nan,
                "high_cutoff": float(high_values.min()) if len(high_values) else np.nan,
                "median_cutoff": median_value,
                "low_mean_phospho": float(low_values.mean()) if len(low_values) else np.nan,
                "high_mean_phospho": float(high_values.mean()) if len(high_values) else np.nan,
                "phospho_split_mode": self.config.phospho_split_mode,
            }

        if self.config.phospho_split_mode != "quantile_extreme":
            raise ValueError(f"Unsupported phospho split mode: {self.config.phospho_split_mode}")

        group_size = int(np.floor(n_valid * self.config.phospho_group_frac))
        group_size = max(group_size, self.config.min_group_samples)
        group_size = min(group_size, n_valid // 2)

        if group_size < self.config.min_group_samples:
            out = dict(base)
            out.update(
                {
                    "status": "insufficient_group_size",
                    "n_valid_phospho_samples": int(n_valid),
                    "n_missing_phospho_samples": int(len(missing_samples)),
                }
            )
            return out

        sorted_values = phospho_values.sort_values(kind="mergesort")
        low_values = sorted_values.iloc[:group_size]
        high_values = sorted_values.iloc[-group_size:]

        return {
            "status": "success",
            "low_samples": low_values.index.tolist(),
            "high_samples": high_values.index.tolist(),
            "n_matched_samples": len(matched_samples),
            "n_valid_phospho_samples": int(n_valid),
            "n_missing_phospho_samples": int(len(missing_samples)),
            "n_low_samples": int(len(low_values)),
            "n_high_samples": int(len(high_values)),
            "low_cutoff": float(low_values.max()) if len(low_values) else np.nan,
            "high_cutoff": float(high_values.min()) if len(high_values) else np.nan,
            "median_cutoff": float(phospho_values.median()) if len(phospho_values) else np.nan,
            "low_mean_phospho": float(low_values.mean()) if len(low_values) else np.nan,
            "high_mean_phospho": float(high_values.mean()) if len(high_values) else np.nan,
            "phospho_split_mode": self.config.phospho_split_mode,
        }

    def build_site_target_expression_rows(
        self,
        cancer_type: str,
        direction: str,
        site_row: pd.Series,
        df_rna: pd.DataFrame,
        df_phospho: pd.DataFrame,
        chip_cache: Dict[str, pd.DataFrame],
        df_protein: Optional[pd.DataFrame] = None,
        phospho_value_mode: Optional[str] = None,
        expression_mode: str = EXPRESSION_MODE_UNADJUSTED,
    ) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
        site = str(site_row["site"])
        tf_name = str(site_row["tf_name"])
        direction_short = self._direction_short(direction)

        if tf_name in chip_cache:
            df_chip = chip_cache[tf_name]
        else:
            df_chip = self.load_chip_targets(tf_name)
            chip_cache[tf_name] = df_chip

        base_summary = {
            "cancer_type": cancer_type,
            "direction": direction,
            "direction_short": direction_short,
            "site": site,
            "tf_name": tf_name,
            "phospho_split_mode": self.config.phospho_split_mode,
            "nonzero_ratio": float(site_row.get("nonzero_ratio", np.nan)),
            "observed_ratio": float(site_row.get("observed_ratio", np.nan)),
            "missing_ratio": float(site_row.get("missing_ratio", np.nan)),
            "n_chip_targets": 0,
            "n_signed_targets": 0,
            "n_expression_targets": 0,
            "n_activate_expression_targets": 0,
            "n_repress_expression_targets": 0,
            "n_low_samples": 0,
            "n_high_samples": 0,
            "low_cutoff": np.nan,
            "high_cutoff": np.nan,
            "status": "pending",
        }

        split_info = self._get_phospho_high_low_samples(
            site,
            df_phospho,
            df_rna,
            df_protein=df_protein,
            phospho_value_mode=phospho_value_mode,
        )
        base_summary.update(
            {
                "n_matched_samples": split_info.get("n_matched_samples", 0),
                "n_valid_phospho_samples": split_info.get("n_valid_phospho_samples", 0),
                "n_low_samples": split_info.get("n_low_samples", 0),
                "n_high_samples": split_info.get("n_high_samples", 0),
                "low_cutoff": split_info.get("low_cutoff", np.nan),
                "high_cutoff": split_info.get("high_cutoff", np.nan),
                "median_cutoff": split_info.get("median_cutoff", np.nan),
                "n_missing_phospho_samples": split_info.get("n_missing_phospho_samples", 0),
                "phospho_split_mode": split_info.get("phospho_split_mode", self.config.phospho_split_mode),
                "low_mean_phospho": split_info.get("low_mean_phospho", np.nan),
                "high_mean_phospho": split_info.get("high_mean_phospho", np.nan),
            }
        )
        if split_info["status"] != "success":
            base_summary["status"] = split_info["status"]
            return [], base_summary

        if df_chip.empty:
            base_summary["status"] = "no_chip_data"
            return [], base_summary

        chip_targets = self.classify_chip_targets(df_chip)
        base_summary["n_chip_targets"] = int(len(chip_targets))
        if chip_targets.empty:
            base_summary["status"] = "no_chip_targets_after_filter"
            return [], base_summary

        chip_target_names = chip_targets["target"].dropna().astype(str).tolist()
        if self.config.signed_target_mode == "chip_intersection":
            signed_targets = self.get_signed_targets_for_tf(tf_name, chip_target_names)
        elif self.config.signed_target_mode == "regulon_only":
            signed_targets = self.get_signed_targets_for_tf(tf_name, None)
        else:
            raise ValueError(f"Unsupported signed_target_mode: {self.config.signed_target_mode}")

        base_summary["n_signed_targets"] = int(len(signed_targets))
        if signed_targets.empty:
            base_summary["status"] = "no_signed_targets"
            return [], base_summary

        genes = self.load_genes()[["gene_id", "gene_name", "gene_name_upper"]].copy()
        target_info = signed_targets.merge(
            genes,
            left_on="target_upper",
            right_on="gene_name_upper",
            how="left",
        )
        target_info = target_info.dropna(subset=["gene_id"]).copy()
        target_info = target_info[target_info["gene_id"].isin(df_rna.index)].copy()
        target_info = target_info.drop_duplicates(subset=["target_upper", "target_regulation", "gene_id"])

        if target_info.empty:
            base_summary["status"] = "no_targets_in_rna_matrix"
            return [], base_summary

        low_samples = split_info["low_samples"]
        high_samples = split_info["high_samples"]
        target_gene_ids = target_info["gene_id"].astype(str).tolist()
        valid_samples = list(dict.fromkeys(low_samples + high_samples))

        if expression_mode == EXPRESSION_MODE_TF_ADJUSTED:
            if df_protein is None:
                base_summary["status"] = "missing_protein_for_tf_adjusted_expression"
                return [], base_summary
            tf_gene_id = self._site_tf_gene_id(site)
            expr_matrix = self._residualize_target_expression(
                df_rna,
                df_protein,
                target_gene_ids,
                tf_gene_id,
                valid_samples,
            )
        else:
            expr_matrix = df_rna.loc[target_gene_ids, valid_samples].apply(pd.to_numeric, errors="coerce")

        low_expression = expr_matrix.loc[target_gene_ids, low_samples].mean(axis=1)
        high_expression = expr_matrix.loc[target_gene_ids, high_samples].mean(axis=1)

        rows: List[Dict[str, object]] = []
        chip_target_upper = set(chip_targets["target_upper"].dropna().astype(str))

        for _, target_row in target_info.iterrows():
            gene_id = str(target_row["gene_id"])
            target_upper = str(target_row["target_upper"])

            for phospho_group, samples, expression_values in [
                ("low", low_samples, low_expression),
                ("high", high_samples, high_expression),
            ]:
                value = expression_values.get(gene_id, np.nan)
                if pd.isna(value):
                    continue

                rows.append(
                    {
                        "cancer_type": cancer_type,
                        "direction": direction,
                        "direction_short": direction_short,
                        "site": site,
                        "tf_name": tf_name,
                        "phospho_split_mode": split_info.get("phospho_split_mode", self.config.phospho_split_mode),
                        "tf_gene_id": site_row.get("tf_gene_id", np.nan),
                        "ACC_ID": site_row.get("ACC_ID", np.nan),
                        "RESIDUE": site_row.get("RESIDUE", np.nan),
                        "POSITION": site_row.get("POSITION", np.nan),
                        "target_gene_id": gene_id,
                        "target_gene_name": target_row.get("gene_name", target_row.get("target", np.nan)),
                        "target_regulation": target_row["target_regulation"],
                        "target_mor": int(target_row["target_mor"]),
                        "phospho_group": phospho_group,
                        "expression_mode": expression_mode,
                        "group_mean_expression": float(value),
                        "n_samples_for_group_expression": int(len(samples)),
                        "n_low_samples": int(len(low_samples)),
                        "n_high_samples": int(len(high_samples)),
                        "low_cutoff": split_info["low_cutoff"],
                        "high_cutoff": split_info["high_cutoff"],
                        "median_cutoff": split_info.get("median_cutoff", np.nan),
                        "n_missing_phospho_samples": int(split_info.get("n_missing_phospho_samples", 0)),
                        "low_mean_phospho": split_info["low_mean_phospho"],
                        "high_mean_phospho": split_info["high_mean_phospho"],
                        "regulon_weight_mean": float(target_row["weight_mean"]),
                        "n_regulon_edges": int(target_row["n_regulon_edges"]),
                        "has_conflicting_sign": bool(target_row["has_conflicting_sign"]),
                        "in_chip_targets": target_upper in chip_target_upper,
                        "score": site_row.get("score", np.nan),
                    }
                )

        if not rows:
            base_summary["status"] = "no_valid_expression_values"
            return [], base_summary

        base_summary["n_expression_targets"] = target_info["gene_id"].nunique()
        base_summary["n_activate_expression_targets"] = target_info.loc[
            target_info["target_regulation"].eq("activate"), "gene_id"
        ].nunique()
        base_summary["n_repress_expression_targets"] = target_info.loc[
            target_info["target_regulation"].eq("repress"), "gene_id"
        ].nunique()
        base_summary["status"] = "success"
        return rows, base_summary

    def analyze_cancer_direction(self, cancer_type: str, direction: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print(
            f"Processing {cancer_type} | {direction} | split={self.config.phospho_split_mode} | "
            f"abundance_primary={self.config.abundance_primary_mode}"
        )
        df_phospho = self.load_phospho(cancer_type)
        df_phospho_filtered = self.filter_phospho_quality(df_phospho, self.config.max_missing_ratio)
        df_rna = self.load_rna(cancer_type)
        df_protein = self.load_protein(cancer_type) if self._needs_protein_matrix() else None
        df_positive = self.load_positive_sites(direction)

        matching_sites = sorted(set(df_positive["site"]) & set(df_phospho_filtered.index))
        df_variable = self.find_variable_sites(
            df_phospho=df_phospho,
            sites=matching_sites,
            min_nonzero_ratio=self.config.min_nonzero_ratio,
            max_nonzero_ratio=self.config.max_nonzero_ratio,
            split_mode=self.config.phospho_split_mode,
            min_group_samples=self.config.min_group_samples,
        )
        if df_variable.empty:
            return pd.DataFrame(), pd.DataFrame()

        df_sites = df_variable.merge(df_positive, on="site", how="left")
        df_sites = self.map_sites_to_tf(df_sites)

        if df_sites.empty:
            return pd.DataFrame(), pd.DataFrame()

        all_rows: List[Dict[str, object]] = []
        summary_rows: List[Dict[str, object]] = []
        chip_cache: Dict[str, pd.DataFrame] = {}

        expression_modes = self._expression_modes_to_build()
        for _, site_row in df_sites.iterrows():
            site_summary: Optional[Dict[str, object]] = None
            for expression_mode in expression_modes:
                try:
                    rows, summary = self.build_site_target_expression_rows(
                        cancer_type=cancer_type,
                        direction=direction,
                        site_row=site_row,
                        df_rna=df_rna,
                        df_phospho=df_phospho,
                        chip_cache=chip_cache,
                        df_protein=df_protein,
                        expression_mode=expression_mode,
                    )
                    if expression_mode == EXPRESSION_MODE_UNADJUSTED or site_summary is None:
                        site_summary = summary
                    all_rows.extend(rows)
                except Exception as exc:
                    site = str(site_row.get("site", ""))
                    tf_name = str(site_row.get("tf_name", ""))
                    direction_short = self._direction_short(direction)
                    if site_summary is None:
                        site_summary = {
                            "cancer_type": cancer_type,
                            "direction": direction,
                            "direction_short": direction_short,
                            "site": site,
                            "tf_name": tf_name,
                            "phospho_split_mode": self.config.phospho_split_mode,
                            "nonzero_ratio": float(site_row.get("nonzero_ratio", np.nan)),
                            "n_chip_targets": 0,
                            "n_signed_targets": 0,
                            "n_expression_targets": 0,
                            "n_activate_expression_targets": 0,
                            "n_repress_expression_targets": 0,
                            "status": f"error: {exc}",
                            "error_type": type(exc).__name__,
                        }
                    print(
                        f"Warning: skipped {cancer_type} | {direction_short} | {expression_mode} | "
                        f"{tf_name} | {site}: {exc}"
                    )
            if site_summary is not None:
                summary_rows.append(site_summary)

        return pd.DataFrame(all_rows), pd.DataFrame(summary_rows)




    @staticmethod
    def _format_p_value(p_value: float) -> str:
        if pd.isna(p_value):
            return "NA"
        if p_value < 1e-4:
            return "<1e-4"
        return f"{p_value:.4f}"

    @staticmethod
    def _p_to_stars(p_value: float) -> str:
        if pd.isna(p_value):
            return "ns"
        if p_value < 0.001:
            return "***"
        if p_value < 0.01:
            return "**"
        if p_value < 0.05:
            return "*"
        return "ns"

    def _draw_scatter_then_boxplot(
        self,
        ax,
        data_lists: List[List[float]],
        positions: List[float],
        box_colors: List[str],
        scatter_colors: Optional[List[str]] = None,
        widths: float = 0.28,
        box_alpha: float = 0.88,
        scatter_s: float = 3.5,
        scatter_alpha: float = 0.12,
        jitter_range: float = 0.055,
        rng: Optional[np.random.Generator] = None,
        medianprops: Optional[Dict[str, object]] = None,
        whiskerprops: Optional[Dict[str, object]] = None,
        capprops: Optional[Dict[str, object]] = None,
        box_edgecolor: str = "#3A3A3A",
        box_edgewidth: float = 0.85,
    ) -> dict:
        """Draw jittered scatter points first, then boxplots on top so boxes stay visible."""
        if scatter_colors is None:
            scatter_colors = box_colors
        if rng is None:
            rng = np.random.default_rng(42)
        if medianprops is None:
            medianprops = {"color": "#2F2F2F", "linewidth": 1.15}
        if whiskerprops is None:
            whiskerprops = {"color": "#4A4A4A", "linewidth": 0.85}
        if capprops is None:
            capprops = {"color": "#4A4A4A", "linewidth": 0.85}

        for pos, values, color in zip(positions, data_lists, scatter_colors):
            jitter = rng.uniform(-jitter_range, jitter_range, size=len(values))
            ax.scatter(
                np.full(len(values), pos) + jitter,
                values,
                s=scatter_s,
                alpha=scatter_alpha,
                edgecolors="none",
                zorder=1,
                color=color,
            )

        bp = ax.boxplot(
            data_lists,
            positions=positions,
            widths=widths,
            showfliers=False,
            patch_artist=True,
            medianprops=medianprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            zorder=5,
        )

        for box, color in zip(bp["boxes"], box_colors):
            box.set(
                facecolor=color,
                edgecolor=box_edgecolor,
                linewidth=box_edgewidth,
                alpha=box_alpha,
                zorder=5,
            )
        for element_key in ("whiskers", "caps", "medians"):
            for artist in bp.get(element_key, []):
                artist.set(zorder=6)

        return bp

    @staticmethod
    def _bh_adjust(p_values: pd.Series) -> pd.Series:
        p = pd.to_numeric(p_values, errors="coerce")
        q = pd.Series(np.nan, index=p.index, dtype=float)

        valid = p.notna()
        if valid.sum() == 0:
            return q

        p_valid = p.loc[valid].astype(float)
        order = np.argsort(p_valid.to_numpy())
        ranked_p = p_valid.to_numpy()[order]
        m = len(ranked_p)

        adjusted = ranked_p * m / np.arange(1, m + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        adjusted = np.clip(adjusted, 0, 1)

        q.loc[p_valid.index[order]] = adjusted
        return q

    def _add_p_value_correction(
        self,
        df_stats: pd.DataFrame,
        group_cols: Optional[List[str]] = None,
        p_col: str = "wilcoxon_p_expected",
        q_col: str = "wilcoxon_q_bh",
    ) -> pd.DataFrame:
        if df_stats.empty:
            return df_stats.copy()

        out = df_stats.copy()
        out[p_col] = pd.to_numeric(out[p_col], errors="coerce")
        out["wilcoxon_p_raw"] = out[p_col]

        if "significance" in out.columns:
            out["significance_raw"] = out["significance"]
        else:
            out["significance_raw"] = out[p_col].map(self._p_to_stars)

        out[q_col] = np.nan

        if group_cols is None or len(group_cols) == 0:
            out[q_col] = self._bh_adjust(out[p_col])
        else:
            for _, idx in out.groupby(group_cols, dropna=False).groups.items():
                out.loc[idx, q_col] = self._bh_adjust(out.loc[idx, p_col])

        out["significance_bh"] = out[q_col].map(self._p_to_stars)
        if self.config.use_bh_pvalue_correction:
            out["significance"] = out["significance_bh"]
            out["wilcoxon_p_for_plot"] = out[q_col]
        else:
            out["significance"] = out["significance_raw"]
            out["wilcoxon_p_for_plot"] = out[p_col]

        return out

    def _plot_significance_from_row(self, row: pd.Series) -> str:
        if self.config.use_bh_pvalue_correction:
            return str(row.get("significance_bh", row.get("significance", "ns")))
        return str(row.get("significance_raw", row.get("significance", "ns")))

    def _plot_significance_column(self, df: pd.DataFrame) -> str:
        if self.config.use_bh_pvalue_correction and "significance_bh" in df.columns:
            return "significance_bh"
        if "significance_raw" in df.columns:
            return "significance_raw"
        return "significance"

    def _finalize_mannwhitney_significance(self, out: pd.DataFrame, p_col: str = "mannwhitney_p") -> pd.DataFrame:
        if out.empty:
            return out
        out = out.copy()
        out["significance_raw"] = out[p_col].map(self._p_to_stars)
        out["mannwhitney_q_bh"] = np.nan
        for _, idx in out.groupby(["cancer_type", "direction_short"], dropna=False).groups.items():
            out.loc[idx, "mannwhitney_q_bh"] = self._bh_adjust(out.loc[idx, p_col])
        out["significance_bh"] = out["mannwhitney_q_bh"].map(self._p_to_stars)
        if self.config.use_bh_pvalue_correction:
            out["significance"] = out["significance_bh"]
        else:
            out["significance"] = out["significance_raw"]
        return out
    @staticmethod
    def _expected_high_low_alternative(direction_short: str, target_regulation: str) -> str:
        direction_short = str(direction_short).lower()
        target_regulation = str(target_regulation).lower()

        if direction_short == "import" and target_regulation == "activate":
            return "greater"
        if direction_short == "import" and target_regulation == "repress":
            return "less"
        if direction_short == "export" and target_regulation == "activate":
            return "less"
        if direction_short == "export" and target_regulation == "repress":
            return "greater"
        return "two-sided"

    @staticmethod
    def _expected_high_low_text(direction_short: str, target_regulation: str) -> str:
        alternative = TargetRegulationBoxplotPipeline._expected_high_low_alternative(
            direction_short,
            target_regulation,
        )
        if alternative == "greater":
            return "High > Low"
        if alternative == "less":
            return "High < Low"
        return "High vs Low"

    @staticmethod
    def _directional_effect(delta: float, alternative: str) -> float:
        if pd.isna(delta):
            return -np.inf
        if alternative == "greater":
            return float(delta)
        if alternative == "less":
            return float(-delta)
        return float(abs(delta))

    @staticmethod
    def _significance_rank(label: object) -> int:
        label = str(label)
        if label == "***":
            return 3
        if label == "**":
            return 2
        if label == "*":
            return 1
        return 0
    def _prepare_stats_for_plotting(self, df_stats: pd.DataFrame) -> pd.DataFrame:
        if df_stats.empty:
            return df_stats.copy()

        out = df_stats.copy()
        out["n_paired_target_genes"] = pd.to_numeric(
            out["n_paired_target_genes"],
            errors="coerce",
        ).fillna(0).astype(int)
        out = out[out["n_paired_target_genes"] >= self.config.min_box_points].copy()
        if out.empty:
            return out

        out["wilcoxon_p_expected"] = pd.to_numeric(out["wilcoxon_p_expected"], errors="coerce")
        out["delta_high_minus_low"] = pd.to_numeric(out["delta_high_minus_low"], errors="coerce")
        out["mean_high_phospho_expression"] = pd.to_numeric(
            out["mean_high_phospho_expression"],
            errors="coerce",
        )
        out["mean_low_phospho_expression"] = pd.to_numeric(
            out["mean_low_phospho_expression"],
            errors="coerce",
        )

        if self.config.use_bh_pvalue_correction and "wilcoxon_q_bh" in out.columns:
            out["wilcoxon_q_bh"] = pd.to_numeric(out["wilcoxon_q_bh"], errors="coerce")
            out["p_sort"] = out["wilcoxon_q_bh"].fillna(np.inf)
        else:
            out["p_sort"] = out["wilcoxon_p_expected"].fillna(np.inf)

        sig_col = self._plot_significance_column(out)
        out["significance"] = out[sig_col].fillna("ns").astype(str)
        out["significance_rank"] = out["significance"].map(self._significance_rank)
        out["is_significant"] = out["significance_rank"] > 0
        out["directional_effect"] = [
            self._directional_effect(delta, alternative)
            for delta, alternative in zip(out["delta_high_minus_low"], out["alternative"].astype(str))
        ]

        out["high_mean_sort"] = out["mean_high_phospho_expression"].fillna(-np.inf)

        return out

    def _ordered_sites_for_regulation(self, df_stats: pd.DataFrame, target_regulation: str) -> List[str]:
        if df_stats.empty:
            return []

        sub = df_stats[df_stats["target_regulation"].astype(str).eq(target_regulation)].copy()
        if sub.empty:
            return []

        if "high_mean_sort" not in sub.columns:
            sub["mean_high_phospho_expression"] = pd.to_numeric(
                sub["mean_high_phospho_expression"],
                errors="coerce",
            )
            sub["high_mean_sort"] = sub["mean_high_phospho_expression"].fillna(-np.inf)

        if "is_significant" not in sub.columns:
            sub["significance"] = sub["significance"].fillna("ns").astype(str)
            sub["significance_rank"] = sub["significance"].map(self._significance_rank)
            sub["is_significant"] = sub["significance_rank"] > 0

        if "p_sort" not in sub.columns:
            if self.config.use_bh_pvalue_correction and "wilcoxon_q_bh" in sub.columns:
                sub["p_sort"] = pd.to_numeric(sub["wilcoxon_q_bh"], errors="coerce").fillna(np.inf)
            else:
                sub["p_sort"] = pd.to_numeric(sub["wilcoxon_p_expected"], errors="coerce").fillna(np.inf)

        sub = sub.sort_values(
            [
                "is_significant",
                "high_mean_sort",
                "p_sort",
                "n_paired_target_genes",
                "site_label",
            ],
            ascending=[False, False, True, False, True],
        )

        return sub["site"].astype(str).drop_duplicates().tolist()

    def _prepare_activity_site_stats_for_plotting(self, df_stats: pd.DataFrame) -> pd.DataFrame:
        if df_stats.empty:
            return df_stats.copy()

        out = df_stats.copy()
        out["n_low_samples"] = pd.to_numeric(out["n_low_samples"], errors="coerce").fillna(0).astype(int)
        out["n_high_samples"] = pd.to_numeric(out["n_high_samples"], errors="coerce").fillna(0).astype(int)
        out = out[
            (out["n_low_samples"] >= self.config.min_group_samples)
            & (out["n_high_samples"] >= self.config.min_group_samples)
        ].copy()
        if out.empty:
            return out

        out["mannwhitney_p"] = pd.to_numeric(out["mannwhitney_p"], errors="coerce")
        out["delta_high_minus_low"] = pd.to_numeric(out["delta_high_minus_low"], errors="coerce")
        out["mean_high_activity"] = pd.to_numeric(out["mean_high_activity"], errors="coerce")
        out["mean_low_activity"] = pd.to_numeric(out["mean_low_activity"], errors="coerce")

        if self.config.use_bh_pvalue_correction and "mannwhitney_q_bh" in out.columns:
            out["mannwhitney_q_bh"] = pd.to_numeric(out["mannwhitney_q_bh"], errors="coerce")
            out["p_sort"] = out["mannwhitney_q_bh"].fillna(np.inf)
        else:
            out["p_sort"] = out["mannwhitney_p"].fillna(np.inf)

        sig_col = self._plot_significance_column(out)
        out["significance"] = out[sig_col].fillna("ns").astype(str)
        out["significance_rank"] = out["significance"].map(self._significance_rank)
        out["is_significant"] = out["significance_rank"] > 0
        out["high_mean_sort"] = out["mean_high_activity"].fillna(-np.inf)
        return out

    def _ordered_sites_for_activity(self, df_stats: pd.DataFrame) -> List[str]:
        if df_stats.empty:
            return []

        sub = self._prepare_activity_site_stats_for_plotting(df_stats)
        if sub.empty:
            return []

        sub = sub.sort_values(
            [
                "is_significant",
                "high_mean_sort",
                "p_sort",
                "n_low_samples",
                "n_high_samples",
                "site_label",
            ],
            ascending=[False, False, True, False, False, True],
        )
        return sub["site"].astype(str).drop_duplicates().tolist()

    def _ordered_cancers_by_activity_significance(
        self,
        df_stats: pd.DataFrame,
        direction_short: str,
        available_cancers: List[str],
    ) -> List[str]:
        if not available_cancers:
            return []

        if df_stats.empty or "direction_short" not in df_stats.columns:
            return list(available_cancers)

        stat_sub = df_stats[df_stats["direction_short"].astype(str).eq(str(direction_short))].copy()
        if stat_sub.empty:
            return list(available_cancers)

        stat_sub = stat_sub[stat_sub["cancer_type"].astype(str).isin([str(c) for c in available_cancers])].copy()
        if stat_sub.empty:
            return list(available_cancers)

        if self.config.use_bh_pvalue_correction and "mannwhitney_q_bh" in stat_sub.columns:
            stat_sub["p_sort"] = pd.to_numeric(stat_sub["mannwhitney_q_bh"], errors="coerce").fillna(np.inf)
        else:
            stat_sub["p_sort"] = pd.to_numeric(stat_sub["mannwhitney_p"], errors="coerce").fillna(np.inf)

        sig_col = self._plot_significance_column(stat_sub)
        stat_sub["significance"] = stat_sub[sig_col].fillna("ns").astype(str)
        stat_sub["significance_rank"] = stat_sub["significance"].map(self._significance_rank)
        stat_sub["is_significant"] = stat_sub["significance_rank"] > 0
        stat_sub["mean_high_activity"] = pd.to_numeric(stat_sub["mean_high_activity"], errors="coerce")
        stat_sub["high_mean_sort"] = stat_sub["mean_high_activity"].fillna(-np.inf)

        stat_sub = stat_sub.sort_values(
            ["significance_rank", "p_sort", "high_mean_sort", "cancer_type"],
            ascending=[False, True, False, True],
        )
        ordered = stat_sub["cancer_type"].astype(str).drop_duplicates().tolist()
        for cancer_type in available_cancers:
            cancer_type = str(cancer_type)
            if cancer_type not in ordered:
                ordered.append(cancer_type)
        return ordered

    @staticmethod
    def _paired_low_high_values(site_sub: pd.DataFrame) -> Tuple[List[float], List[float], int]:
        if site_sub.empty or not {"low", "high"}.issubset(set(site_sub["phospho_group"].astype(str))):
            return [], [], 0

        wide = (
            site_sub.pivot_table(
                index="target_gene_id",
                columns="phospho_group",
                values="group_mean_expression",
                aggfunc="mean",
            )
            .dropna(subset=["low", "high"], how="any")
        )
        if wide.empty:
            return [], [], 0

        low = pd.to_numeric(wide["low"], errors="coerce")
        high = pd.to_numeric(wide["high"], errors="coerce")
        valid = low.notna() & high.notna()
        low = low[valid]
        high = high[valid]
        return low.tolist(), high.tolist(), int(valid.sum())

    def _compare_high_low_by_site(self, df_plot: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        group_cols = ["cancer_type", "direction_short", "target_regulation", "site", "site_label", "tf_name"]

        for keys, sub in df_plot.groupby(group_cols):
            cancer_type, direction_short, target_regulation, site, site_label, tf_name = keys
            wide = (
                sub.pivot_table(
                    index="target_gene_id",
                    columns="phospho_group",
                    values="group_mean_expression",
                    aggfunc="mean",
                )
                .dropna(subset=["low", "high"], how="any")
                if {"low", "high"}.issubset(set(sub["phospho_group"]))
                else pd.DataFrame()
            )

            alternative = self._expected_high_low_alternative(direction_short, target_regulation)
            p_value = np.nan
            n_targets = int(len(wide))
            mean_low = np.nan
            mean_high = np.nan
            delta = np.nan

            if n_targets > 0:
                low = pd.to_numeric(wide["low"], errors="coerce")
                high = pd.to_numeric(wide["high"], errors="coerce")
                mean_low = float(low.mean())
                mean_high = float(high.mean())
                delta = float(mean_high - mean_low)

                if n_targets >= 3 and not np.allclose((high - low).fillna(0).to_numpy(), 0):
                    try:
                        p_value = float(wilcoxon(high, low, alternative=alternative).pvalue)
                    except ValueError:
                        p_value = np.nan

            rows.append(
                {
                    "cancer_type": cancer_type,
                    "direction_short": direction_short,
                    "target_regulation": target_regulation,
                    "site": site,
                    "site_label": site_label,
                    "tf_name": tf_name,
                    "n_paired_target_genes": n_targets,
                    "mean_low_phospho_expression": mean_low,
                    "mean_high_phospho_expression": mean_high,
                    "delta_high_minus_low": delta,
                    "expected_direction": self._expected_high_low_text(direction_short, target_regulation),
                    "alternative": alternative,
                    "wilcoxon_p_expected": p_value,
                    "significance": self._p_to_stars(p_value),
                }
            )

        return pd.DataFrame(rows)


    def plot_cancer_direction_boxplot(
        self,
        df_points: pd.DataFrame,
        df_stats: pd.DataFrame,
        cancer_type: str,
        direction_short: str,
        output_dir: Path,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        df_plot = df_points.copy()
        if df_plot.empty:
            return

        df_plot = df_plot[df_plot["target_regulation"].astype(str).eq("activate")].copy()
        if df_plot.empty:
            return

        if not df_stats.empty:
            df_stats = df_stats[df_stats["target_regulation"].astype(str).eq("activate")].copy()

        df_plot["site_label"] = self._site_display_label_from_df(df_plot)
        regulation_order = ["activate"]
        section_title_map = {
            "activate": "Activate targets",
        }
        group_color_map = {
            "low": "#8FAFBC",
            "high": "#E3A07A",
        }
        group_label_map = {
            "low": "Low phospho",
            "high": "High phospho",
        }
        site_label_color_map = {
            "known_positive": "#C97B49",
            "new_predicted": "#5E8FA1",
        }
        site_label_name_map = {
            "known_positive": "Known positive site",
            "new_predicted": "New predicted site",
        }

        df_stats_for_plot = self._prepare_stats_for_plotting(df_stats)

        positions = []
        data_lists = []
        box_colors = []
        scatter_colors = []
        x_centers = []
        ticklabels = []
        ticklabel_colors = []
        plotted_site_statuses = set()
        current_x = 1.0
        pair_offset = 0.16
        section_gap = 1.35
        section_meta = []
        site_stat_pos = {}

        for reg in regulation_order:
            sub = df_plot[df_plot["target_regulation"] == reg].copy()
            if sub.empty:
                continue

            site_order = self._ordered_sites_for_regulation(df_stats_for_plot, reg)
            if len(site_order) == 0:
                continue

            start_x = current_x
            plotted_site_count = 0
            plotted_target_genes = set()
            plotted_point_count = 0

            for site in site_order:
                site_sub = sub[sub["site"].astype(str).eq(str(site))].copy()
                if site_sub.empty:
                    continue

                site_label = str(site_sub["site_label"].dropna().astype(str).iloc[0])
                low_values, high_values, n_box_points = self._paired_low_high_values(site_sub)
                if n_box_points < self.config.min_box_points:
                    continue

                positions.extend([current_x - pair_offset, current_x + pair_offset])
                data_lists.extend([low_values, high_values])
                box_colors.extend([group_color_map["low"], group_color_map["high"]])
                scatter_colors.extend([group_color_map["low"], group_color_map["high"]])

                site_status = self.site_label_status(site_sub, direction_short)
                x_centers.append(current_x)
                ticklabels.append(site_label)
                ticklabel_colors.append(site_label_color_map.get(site_status, site_label_color_map["new_predicted"]))
                plotted_site_statuses.add(site_status)
                site_stat_pos[(reg, site)] = current_x

                plotted_target_genes.update(site_sub["target_gene_id"].dropna().astype(str).tolist())
                plotted_point_count += n_box_points
                current_x += 1.0
                plotted_site_count += 1

            end_x = current_x - 1.0
            if plotted_site_count > 0:
                section_meta.append(
                    {
                        "reg": reg,
                        "title": section_title_map[reg],
                        "start": start_x,
                        "end": end_x,
                        "n_sites": plotted_site_count,
                        "n_targets": len(plotted_target_genes),
                        "n_points": plotted_point_count,
                    }
                )
                current_x += section_gap

        if len(data_lists) == 0:
            return

        fig_width = max(10.0, 0.50 * len(x_centers) + 3.6)
        fig, ax = plt.subplots(figsize=(fig_width, 4.9))

        rng = np.random.default_rng(42)
        self._draw_scatter_then_boxplot(
            ax,
            data_lists,
            positions,
            box_colors,
            scatter_colors=scatter_colors,
            widths=0.23,
            box_alpha=0.88,
            scatter_s=3.5,
            scatter_alpha=0.35,
            jitter_range=0.045,
            rng=rng,
        )

        if len(section_meta) >= 2:
            for left, right in zip(section_meta[:-1], section_meta[1:]):
                x_div = (left["end"] + right["start"]) / 2.0
                ax.axvline(x=x_div, color="#6E6E6E", linestyle="--", linewidth=0.9, alpha=0.75)

        ax.set_xticks(x_centers)
        ax.set_xticklabels(ticklabels, rotation=55, ha="right", fontsize=12.6)
        for tick_label, tick_color in zip(ax.get_xticklabels(), ticklabel_colors):
            tick_label.set_color(tick_color)
            if tick_color == site_label_color_map["known_positive"]:
                tick_label.set_fontweight("bold")

        ax.set_ylabel("Mean target expression", fontsize=15.5)
        ax.set_xlabel("Phosphosite", fontsize=15.5)
        ax.tick_params(axis="y", labelsize=14.5)
        ax.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.30)
        ax.grid(axis="x", visible=False)
        sns.despine(ax=ax)

        all_values = [value for sublist in data_lists for value in sublist]
        y_lower, y_upper, y_range = _zero_aligned_expression_ylim_from_values(
            all_values,
            fallback_ylim=ax.get_ylim(),
        )
        ax.set_ylim(y_lower, y_upper)

        if not df_stats.empty:
            for _, stat_row in df_stats.iterrows():
                sig_label = stat_row.get("significance", "ns")
                if sig_label == "ns":
                    continue
                key = (stat_row["target_regulation"], stat_row["site"])
                if key not in site_stat_pos:
                    continue
                ax.text(
                    site_stat_pos[key],
                    y_upper + y_range * 0.095,
                    sig_label,
                    ha="center",
                    va="bottom",
                    fontsize=16.5,
                    fontweight="normal",
                    color="#2F2F2F",
                )

        _finalize_zero_aligned_expression_ylim(
            ax,
            y_lower,
            y_upper,
            y_range,
            final_top_padding_frac=0.17,
        )

        from matplotlib.patches import Patch

        handles = [
            Patch(
                facecolor=group_color_map["low"],
                edgecolor="#3A3A3A",
                linewidth=0.85,
                alpha=0.88,
                label=group_label_map["low"],
            ),
            Patch(
                facecolor=group_color_map["high"],
                edgecolor="#3A3A3A",
                linewidth=0.85,
                alpha=0.88,
                label=group_label_map["high"],
            ),
        ]

        site_legend_statuses = [status for status in ["known_positive", "new_predicted"] if status in plotted_site_statuses]
        for status in site_legend_statuses:
            handles.append(
                Patch(
                    facecolor="none",
                    edgecolor="none",
                    alpha=0.0,
                    label=site_label_name_map[status],
                )
            )

        legend = ax.legend(
            handles=handles,
            frameon=False,
            fontsize=14.2,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            handlelength=1.2,
            handletextpad=0.55,
            labelspacing=0.45,
        )

        legend_texts = legend.get_texts()
        for text in legend_texts:
            label = text.get_text()
            if label == site_label_name_map["known_positive"]:
                text.set_color(site_label_color_map["known_positive"])
            elif label == site_label_name_map["new_predicted"]:
                text.set_color(site_label_color_map["new_predicted"])

        ax.set_title(
            f"{cancer_type} {direction_short} activate targets",
            fontsize=16.5,
            pad=12,
        )

        plt.tight_layout(rect=[0, 0, 0.88, 1])

        prefix = self._sanitize_filename(
            f"{cancer_type}_{direction_short}_sitewise_high_low_phospho_target_expression_boxplot"
        )
        for ext in ["png", "pdf", "svg"]:
            _savefig_with_numeric_ticks(
                fig,
                output_dir / f"{prefix}.{ext}",
                ax,
                dpi=self.config.dpi,
                numeric_x=False,
                numeric_y=True,
            )
        plt.close(fig)


    def _merged_pair_index_columns(self) -> List[str]:
        return ["cancer_type", "direction_short", "target_regulation", "site", "target_gene_id"]

    def _build_merged_high_low_pairs(self, df_points: pd.DataFrame) -> pd.DataFrame:
        if df_points.empty:
            return pd.DataFrame()

        df = df_points.copy()
        if "site_label" not in df.columns:
            df["site_label"] = self._site_display_label_from_df(df)

        pair_cols = self._merged_pair_index_columns()
        required_cols = set(pair_cols + ["phospho_group", "group_mean_expression"])
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Merged analysis missing required columns: {sorted(missing_cols)}")

        wide = (
            df.pivot_table(
                index=pair_cols,
                columns="phospho_group",
                values="group_mean_expression",
                aggfunc="mean",
            )
            .reset_index()
        )

        if "low" not in wide.columns or "high" not in wide.columns:
            return pd.DataFrame()

        wide = wide.dropna(subset=["low", "high"], how="any").copy()
        if wide.empty:
            return pd.DataFrame()

        meta = (
            df.groupby(pair_cols, as_index=False)
            .agg(
                target_gene_name=("target_gene_name", "first"),
                n_source_sites=("site", "nunique"),
                n_source_tfs=("tf_name", "nunique"),
                source_sites=("site_label", self._join_unique),
                source_tfs=("tf_name", self._join_unique),
                mean_site_score=("score", "mean"),
            )
        )

        out = wide.merge(meta, on=pair_cols, how="left")
        out["low_phospho_mean_expression"] = pd.to_numeric(out["low"], errors="coerce")
        out["high_phospho_mean_expression"] = pd.to_numeric(out["high"], errors="coerce")
        out["delta_high_minus_low"] = out["high_phospho_mean_expression"] - out["low_phospho_mean_expression"]
        out["expected_direction"] = [
            self._expected_high_low_text(direction_short, target_regulation)
            for direction_short, target_regulation in zip(out["direction_short"], out["target_regulation"])
        ]
        out["alternative"] = [
            self._expected_high_low_alternative(direction_short, target_regulation)
            for direction_short, target_regulation in zip(out["direction_short"], out["target_regulation"])
        ]

        return out.reset_index(drop=True)

    def _compare_merged_high_low_by_cancer_regulation(self, df_pairs: pd.DataFrame) -> pd.DataFrame:
        if df_pairs.empty:
            return pd.DataFrame()

        rows: List[Dict[str, object]] = []
        group_cols = ["cancer_type", "direction_short", "target_regulation"]

        for keys, sub in df_pairs.groupby(group_cols):
            cancer_type, direction_short, target_regulation = keys
            low = pd.to_numeric(sub["low_phospho_mean_expression"], errors="coerce")
            high = pd.to_numeric(sub["high_phospho_mean_expression"], errors="coerce")
            valid = low.notna() & high.notna()
            low = low[valid]
            high = high[valid]

            alternative = self._expected_high_low_alternative(direction_short, target_regulation)
            p_value = np.nan
            n_pairs = int(valid.sum())
            mean_low = np.nan
            mean_high = np.nan
            delta = np.nan

            if n_pairs > 0:
                mean_low = float(low.mean())
                mean_high = float(high.mean())
                delta = float(mean_high - mean_low)

                if n_pairs >= self.config.min_box_points and not np.allclose((high - low).fillna(0).to_numpy(), 0):
                    try:
                        p_value = float(wilcoxon(high, low, alternative=alternative).pvalue)
                    except ValueError:
                        p_value = np.nan

            rows.append(
                {
                    "cancer_type": cancer_type,
                    "direction_short": direction_short,
                    "target_regulation": target_regulation,
                    "merged_pair_unit": "site_target",
                    "n_paired_points": n_pairs,
                    "n_unique_target_genes": int(sub.loc[valid, "target_gene_id"].nunique()) if "target_gene_id" in sub.columns else n_pairs,
                    "n_unique_sites": int(sub.loc[valid, "site"].nunique()) if "site" in sub.columns else int(sub.loc[valid, "n_source_sites"].sum()) if "n_source_sites" in sub.columns else np.nan,
                    "n_unique_tfs": int(sub.loc[valid, "n_source_tfs"].sum()) if "n_source_tfs" in sub.columns else np.nan,
                    "mean_low_phospho_expression": mean_low,
                    "mean_high_phospho_expression": mean_high,
                    "delta_high_minus_low": delta,
                    "expected_direction": self._expected_high_low_text(direction_short, target_regulation),
                    "alternative": alternative,
                    "wilcoxon_p_expected": p_value,
                    "significance": self._p_to_stars(p_value),
                }
            )

        return pd.DataFrame(rows)

    def plot_cancer_direction_merged_boxplot(
        self,
        df_pairs: pd.DataFrame,
        df_stats: pd.DataFrame,
        cancer_type: str,
        direction_short: str,
        output_dir: Path,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        if df_pairs.empty:
            return

        df_pairs = df_pairs[df_pairs["target_regulation"].astype(str).eq("activate")].copy()
        if df_pairs.empty:
            return

        if not df_stats.empty:
            df_stats = df_stats[df_stats["target_regulation"].astype(str).eq("activate")].copy()

        regulation_order = ["activate"]
        section_title_map = {
            "activate": "Activate targets",
        }
        group_color_map = {
            "low": "#95B0B5",
            "high": "#E7A983",
        }
        group_label_map = {
            "low": "Low phospho",
            "high": "High phospho",
        }

        positions = []
        data_lists = []
        box_colors = []
        scatter_colors = []
        x_centers = []
        ticklabels = []
        stat_pos = {}
        current_x = 1.0
        pair_offset = 0.18
        section_gap = 0.65

        for reg in regulation_order:
            sub = df_pairs[df_pairs["target_regulation"].astype(str).eq(reg)].copy()
            if sub.empty:
                continue

            low_values = pd.to_numeric(sub["low_phospho_mean_expression"], errors="coerce")
            high_values = pd.to_numeric(sub["high_phospho_mean_expression"], errors="coerce")
            valid = low_values.notna() & high_values.notna()
            low_values = low_values[valid].tolist()
            high_values = high_values[valid].tolist()
            n_points = int(valid.sum())

            if n_points < self.config.min_box_points:
                continue

            positions.extend([current_x - pair_offset, current_x + pair_offset])
            data_lists.extend([low_values, high_values])
            box_colors.extend([group_color_map["low"], group_color_map["high"]])
            scatter_colors.extend([group_color_map["low"], group_color_map["high"]])
            x_centers.append(current_x)
            label = section_title_map[reg]
            ticklabels.append(label)
            stat_pos[reg] = current_x
            current_x += 1.0 + section_gap

        if len(data_lists) == 0:
            return

        fig, ax = plt.subplots(figsize=(7.0, 4.8))

        rng = np.random.default_rng(42)
        self._draw_scatter_then_boxplot(
            ax,
            data_lists,
            positions,
            box_colors,
            scatter_colors=scatter_colors,
            widths=0.28,
            box_alpha=0.88,
            scatter_s=3.5,
            scatter_alpha=0.35,
            jitter_range=0.055,
            rng=rng,
        )

        if len(x_centers) >= 2:
            x_div = (x_centers[0] + x_centers[1]) / 2.0
            ax.axvline(x=x_div, color="#6E6E6E", linestyle="--", linewidth=0.9, alpha=0.75)

        ax.set_xticks(x_centers)
        ax.set_xticklabels(ticklabels, rotation=0, ha="center", fontsize=14.3)
        ax.set_ylabel("Mean target expression", fontsize=15.5)
        ax.set_xlabel("Merged site-target pairs across all sites", fontsize=15.5)
        ax.tick_params(axis="y", labelsize=14.5)
        ax.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.30)
        ax.grid(axis="x", visible=False)
        sns.despine(ax=ax)

        all_values = [value for sublist in data_lists for value in sublist]
        y_lower, y_upper, y_range = _zero_aligned_expression_ylim_from_values(
            all_values,
            fallback_ylim=ax.get_ylim(),
        )
        ax.set_ylim(y_lower, y_upper)

        if not df_stats.empty:
            for _, stat_row in df_stats.iterrows():
                reg = str(stat_row["target_regulation"])
                sig_label = stat_row.get("significance", "ns")
                if sig_label == "ns" or reg not in stat_pos:
                    continue
                ax.text(
                    stat_pos[reg],
                    y_upper + y_range * 0.055,
                    sig_label,
                    ha="center",
                    va="bottom",
                    fontsize=18,
                    fontweight="normal",
                    color="#2F2F2F",
                )

        _finalize_zero_aligned_expression_ylim(
            ax,
            y_lower,
            y_upper,
            y_range,
            final_top_padding_frac=0.14,
        )

        handles = [
            plt.Line2D([0], [0], marker="s", linestyle="", markersize=7, markerfacecolor=group_color_map["low"], markeredgecolor="#3A3A3A", label=group_label_map["low"]),
            plt.Line2D([0], [0], marker="s", linestyle="", markersize=7, markerfacecolor=group_color_map["high"], markeredgecolor="#3A3A3A", label=group_label_map["high"]),
        ]
        ax.legend(
            handles=handles,
            frameon=False,
            fontsize=14,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
        )

        plt.tight_layout(rect=[0, 0, 0.88, 1])

        prefix = self._sanitize_filename(
            f"{cancer_type}_{direction_short}_merged_all_sites_high_low_phospho_target_expression_boxplot"
        )
        for ext in ["png", "pdf", "svg"]:
            _savefig_with_numeric_ticks(
                fig,
                output_dir / f"{prefix}.{ext}",
                ax,
                dpi=self.config.dpi,
                numeric_x=False,
                numeric_y=True,
            )
        plt.close(fig)


    @staticmethod
    def _format_p_value_for_plot(p_value: float) -> str:
        if pd.isna(p_value):
            return "NA"
        if p_value < 1e-3:
            return f"{p_value:.1e}"
        return f"{p_value:.4f}"


    def plot_cancer_direction_merged_paired_boxplot(
        self,
        df_pairs: pd.DataFrame,
        df_stats: pd.DataFrame,
        cancer_type: str,
        direction_short: str,
        output_dir: Path,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        if df_pairs.empty:
            return

        df_pairs = df_pairs[df_pairs["target_regulation"].astype(str).eq("activate")].copy()
        if df_pairs.empty:
            return

        low = pd.to_numeric(df_pairs["low_phospho_mean_expression"], errors="coerce")
        high = pd.to_numeric(df_pairs["high_phospho_mean_expression"], errors="coerce")
        valid = low.notna() & high.notna()

        plot_df = df_pairs.loc[valid].copy()
        plot_df["low_value"] = low.loc[valid].to_numpy()
        plot_df["high_value"] = high.loc[valid].to_numpy()
        plot_df["delta_value"] = plot_df["high_value"] - plot_df["low_value"]

        if len(plot_df) < self.config.min_box_points:
            return

        plot_df = plot_df.sort_values("delta_value", ascending=True).reset_index(drop=True)

        low_values = plot_df["low_value"].to_numpy()
        high_values = plot_df["high_value"].to_numpy()
        n_pairs = len(plot_df)

        color_low = "#95B0B5"
        color_high = "#E7A983"
        edge_color = "#333333"
        line_color = "#B7B7B7"

        fig, ax = plt.subplots(figsize=(4.2, 4.6))

        x_low = 1.0
        x_high = 2.0
        box_width = 0.36

        rng = np.random.default_rng(42)
        jitter = rng.uniform(-0.055, 0.055, size=n_pairs)

        line_alpha = max(0.5, min(0.18, 35.0 / max(n_pairs, 1)))
        point_alpha = max(0.35, min(0.70, 90.0 / max(n_pairs, 1)))

        low_x = np.full(n_pairs, x_low) + jitter
        high_x = np.full(n_pairs, x_high) + jitter

        for lx, hx, lv, hv in zip(low_x, high_x, low_values, high_values):
            ax.plot(
                [lx, hx],
                [lv, hv],
                color=line_color,
                linewidth=0.6,
                alpha=line_alpha,
                zorder=1,
            )

        ax.scatter(
            low_x,
            low_values,
            s=7,
            color=color_low,
            edgecolors=edge_color,
            linewidths=0.25,
            alpha=point_alpha,
            zorder=2,
        )
        ax.scatter(
            high_x,
            high_values,
            s=7,
            color=color_high,
            edgecolors=edge_color,
            linewidths=0.25,
            alpha=point_alpha,
            zorder=2,
        )

        bp = ax.boxplot(
            [low_values, high_values],
            positions=[x_low, x_high],
            widths=box_width,
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "#222222", "linewidth": 1.3},
            whiskerprops={"color": "#555555", "linewidth": 0.9},
            capprops={"color": "#555555", "linewidth": 0.9},
            boxprops={"linewidth": 0.9},
            zorder=5,
        )

        for box, color in zip(bp["boxes"], [color_low, color_high]):
            box.set(facecolor=color, edgecolor=edge_color, alpha=0.55, zorder=5)
        for element_key in ("whiskers", "caps", "medians"):
            for artist in bp.get(element_key, []):
                artist.set(zorder=6)


        stat_sub = pd.DataFrame()
        if not df_stats.empty:
            stat_sub = df_stats[
                df_stats["cancer_type"].astype(str).eq(str(cancer_type))
                & df_stats["direction_short"].astype(str).eq(str(direction_short))
                & df_stats["target_regulation"].astype(str).eq("activate")
            ].copy()

        p_value = np.nan
        sig_label = "ns"
        if not stat_sub.empty:
            if "wilcoxon_p_for_plot" in stat_sub.columns:
                p_col = "wilcoxon_p_for_plot"
            elif "wilcoxon_q_bh" in stat_sub.columns:
                p_col = "wilcoxon_q_bh"
            else:
                p_col = "wilcoxon_p_expected"

            p_value_series = pd.to_numeric(stat_sub[p_col], errors="coerce").dropna()
            if len(p_value_series) > 0:
                p_value = float(p_value_series.iloc[0])
                if "significance" in stat_sub.columns:
                    sig_series = stat_sub["significance"].dropna().astype(str)
                    sig_label = sig_series.iloc[0] if len(sig_series) > 0 else self._p_to_stars(p_value)
                else:
                    sig_label = self._p_to_stars(p_value)
            else:
                p_value = np.nan

        ymin = float(np.nanmin([np.nanmin(low_values), np.nanmin(high_values)]))
        ymax = float(np.nanmax([np.nanmax(low_values), np.nanmax(high_values)]))
        y_range = ymax - ymin
        if y_range <= 0:
            y_range = 1.0

        bracket_y = ymax + y_range * 0.08
        text_y = ymax + y_range * 0.115

        ax.plot(
            [x_low, x_low, x_high, x_high],
            [bracket_y - y_range * 0.015, bracket_y, bracket_y, bracket_y - y_range * 0.015],
            color="#222222",
            linewidth=0.85,
            clip_on=False,
        )

        if pd.isna(p_value):
            p_text = sig_label
        elif p_value < 1e-4:
            p_text = f"{sig_label}"
        else:
            p_text = f"{sig_label}"

        ax.text(
            1.5,
            text_y,
            p_text,
            ha="center",
            va="bottom",
            fontsize=15.2,
            color="#222222",
        )

        ax.set_xlim(0.55, 2.45)
        ax.set_ylim(ymin - y_range * 0.08, ymax + y_range * 0.20)

        ax.set_xticks([x_low, x_high])
        ax.set_xticklabels(["Low phospho", "High phospho"], fontsize=15)
        ax.set_ylabel("Mean target expression", fontsize=15.8)
        ax.set_xlabel("")

        ax.set_title(
            f"{cancer_type} {direction_short} activate targets",
            fontsize=16.5,
            pad=12,
        )

        ax.tick_params(axis="y", labelsize=14.8)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.22)
        ax.grid(axis="x", visible=False)

        sns.despine(ax=ax)
        plt.tight_layout()

        prefix = self._sanitize_filename(
            f"{cancer_type}_{direction_short}_merged_all_sites_high_low_phospho_target_expression_paired_line_boxplot"
        )
        for ext in ["png", "pdf", "svg"]:
            _savefig_with_numeric_ticks(
                fig,
                output_dir / f"{prefix}.{ext}",
                ax,
                dpi=self.config.dpi,
                numeric_x=False,
                numeric_y=True,
            )
        plt.close(fig)

    def plot_all_merged_boxplots(self, df_points: pd.DataFrame, output_dir: Path) -> None:
        if df_points.empty:
            return

        plot_dir = output_dir / "merged_all_sites_boxplots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        df_pairs = self._build_merged_high_low_pairs(df_points)
        if df_pairs.empty:
            print("No merged high low pairs were available for plotting.")
            return

        df_pairs.to_csv(plot_dir / "merged_high_low_expression_pairs.csv", index=False)

        df_stats = self._compare_merged_high_low_by_cancer_regulation(df_pairs)
        df_stats = self._add_p_value_correction(
            df_stats,
            group_cols=["direction_short", "target_regulation"],
            p_col="wilcoxon_p_expected",
            q_col="wilcoxon_q_bh",
        )
        df_stats.to_csv(plot_dir / "merged_high_low_comparison_by_cancer_direction_regulation.csv", index=False)

        for (cancer_type, direction_short), df_sub in df_pairs.groupby(["cancer_type", "direction_short"]):
            stat_sub = df_stats[
                df_stats["cancer_type"].eq(cancer_type)
                & df_stats["direction_short"].eq(direction_short)
            ].copy()
            self.plot_cancer_direction_merged_boxplot(
                df_sub,
                stat_sub,
                str(cancer_type),
                str(direction_short),
                plot_dir,
            )
            self.plot_cancer_direction_merged_paired_boxplot(
                df_sub,
                stat_sub,
                str(cancer_type),
                str(direction_short),
                plot_dir,
            )

        self.plot_activate_merged_across_cancers(df_pairs, df_stats, plot_dir)



    def plot_activate_merged_across_cancers(
        self,
        df_pairs: pd.DataFrame,
        df_stats: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        if df_pairs.empty:
            return

        df_activate_all = df_pairs[df_pairs["target_regulation"].astype(str).eq("activate")].copy()
        if df_activate_all.empty:
            print("No activate merged pairs were available for across-cancer plotting.")
            return

        group_color_map = {
            "low": "#95B0B5",
            "high": "#E7A983",
        }
        group_label_map = {
            "low": "Low phospho",
            "high": "High phospho",
        }

        available_directions = set(df_activate_all["direction_short"].dropna().astype(str))
        configured_directions = [self._direction_short(d) for d in self.config.directions]
        direction_order = [d for d in configured_directions if d in available_directions]
        direction_order.extend(sorted(available_directions - set(direction_order)))

        for direction_short in direction_order:
            df_sub = df_activate_all[df_activate_all["direction_short"].astype(str).eq(str(direction_short))].copy()
            if df_sub.empty:
                continue

            available_cancers = set(df_sub["cancer_type"].dropna().astype(str))
            stat_sub_direction = (
                df_stats[
                    df_stats["direction_short"].astype(str).eq(str(direction_short))
                    & df_stats["target_regulation"].astype(str).eq("activate")
                ].copy()
                if not df_stats.empty
                else pd.DataFrame()
            )
            if not stat_sub_direction.empty:
                order_stats = stat_sub_direction.rename(
                    columns={
                        "wilcoxon_p_expected": "mannwhitney_p",
                        "wilcoxon_q_bh": "mannwhitney_q_bh",
                        "mean_high_phospho_expression": "mean_high_activity",
                    }
                )
                cancer_order = self._ordered_cancers_by_activity_significance(
                    order_stats,
                    str(direction_short),
                    [str(c) for c in self.config.cancer_types if str(c) in available_cancers]
                    + sorted(available_cancers - {str(c) for c in self.config.cancer_types}),
                )
            else:
                cancer_order = [c for c in self.config.cancer_types if str(c) in available_cancers]
                cancer_order.extend(sorted(available_cancers - set(cancer_order)))

            positions = []
            data_lists = []
            box_colors = []
            scatter_colors = []
            x_centers = []
            ticklabels = []
            stat_pos = {}
            summary_rows: List[Dict[str, object]] = []

            current_x = 1.0
            pair_offset = 0.20

            for cancer_type in cancer_order:
                csub = df_sub[df_sub["cancer_type"].astype(str).eq(str(cancer_type))].copy()
                if csub.empty:
                    continue

                low_values = pd.to_numeric(csub["low_phospho_mean_expression"], errors="coerce")
                high_values = pd.to_numeric(csub["high_phospho_mean_expression"], errors="coerce")
                valid = low_values.notna() & high_values.notna()
                low_list = low_values[valid].tolist()
                high_list = high_values[valid].tolist()
                n_points = int(valid.sum())

                if n_points < self.config.min_box_points:
                    continue

                positions.extend([current_x - pair_offset, current_x + pair_offset])
                data_lists.extend([low_list, high_list])
                box_colors.extend([group_color_map["low"], group_color_map["high"]])
                scatter_colors.extend([group_color_map["low"], group_color_map["high"]])
                x_centers.append(current_x)
                ticklabels.append(str(cancer_type))
                stat_pos[str(cancer_type)] = current_x

                summary_rows.append(
                    {
                        "cancer_type": str(cancer_type),
                        "direction_short": str(direction_short),
                        "target_regulation": "activate",
                        "n_paired_points": n_points,
                        "n_unique_target_genes": int(csub.loc[valid, "target_gene_id"].nunique()) if "target_gene_id" in csub.columns else n_points,
                        "n_unique_sites": int(csub.loc[valid, "site"].nunique()) if "site" in csub.columns else np.nan,
                        "mean_low_phospho_expression": float(np.mean(low_list)) if low_list else np.nan,
                        "mean_high_phospho_expression": float(np.mean(high_list)) if high_list else np.nan,
                        "delta_high_minus_low": float(np.mean(high_list) - np.mean(low_list)) if low_list and high_list else np.nan,
                    }
                )
                current_x += 1.0

            if len(data_lists) == 0:
                print(f"No {direction_short} activate cancer group passed the minimum box point cutoff.")
                continue

            direction_file_prefix = str(direction_short).lower()
            pd.DataFrame(summary_rows).to_csv(
                output_dir / f"{direction_file_prefix}_activate_merged_across_cancers_plot_summary.csv",
                index=False,
            )

            fig_width = max(11.0, 0.85 * len(x_centers) + 3.0)
            fig_height = 5.0
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            rng = np.random.default_rng(self.config.random_seed)
            self._draw_scatter_then_boxplot(
                ax,
                data_lists,
                positions,
                box_colors,
                scatter_colors=scatter_colors,
                widths=0.34,
                box_alpha=0.88,
                scatter_s=3.5,
                scatter_alpha=0.12,
                jitter_range=0.06,
                rng=rng,
            )

            ax.set_xticks(x_centers)
            ax.set_xticklabels(ticklabels, rotation=35, ha="right", fontsize=14.5)
            ax.set_ylabel("Mean target expression", fontsize=15.5)
            ax.set_xlabel("Cancer type", fontsize=15.5)
            ax.tick_params(axis="y", labelsize=14.5)
            ax.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.30)
            ax.grid(axis="x", visible=False)
            sns.despine(ax=ax)
            ax.spines["left"].set_linewidth(0.85)
            ax.spines["left"].set_color("#4A4A4A")

            y_cap = 1.0
            ax.set_ylim(-y_cap, y_cap)
            ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
            star_y = y_cap * 0.90

            stat_sub = (
                df_stats[
                    df_stats["direction_short"].astype(str).eq(str(direction_short))
                    & df_stats["target_regulation"].astype(str).eq("activate")
                ].copy()
                if not df_stats.empty
                else pd.DataFrame()
            )
            if not stat_sub.empty:
                for _, stat_row in stat_sub.iterrows():
                    cancer_type = str(stat_row.get("cancer_type", ""))
                    sig_label = self._plot_significance_from_row(stat_row)
                    if sig_label == "ns" or cancer_type not in stat_pos:
                        continue
                    ax.text(
                        stat_pos[cancer_type],
                        star_y,
                        sig_label,
                        ha="center",
                        va="center",
                        fontsize=17,
                        color="#2F2F2F",
                    )

            handles = [
                plt.Line2D([0], [0], marker="s", linestyle="", markersize=7, markerfacecolor=group_color_map["low"], markeredgecolor="#3A3A3A", label=group_label_map["low"]),
                plt.Line2D([0], [0], marker="s", linestyle="", markersize=7, markerfacecolor=group_color_map["high"], markeredgecolor="#3A3A3A", label=group_label_map["high"]),
            ]
            ax.legend(
                handles=handles,
                frameon=False,
                fontsize=14,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0.0,
            )
            plt.tight_layout(rect=[0, 0, 0.88, 1])

            prefix = f"{direction_short}_activate_merged_all_cancers_high_low_phospho_target_expression_boxplot"
            prefix = self._sanitize_filename(prefix)
            for ext in ["png", "pdf", "svg"]:
                _savefig_with_numeric_ticks(
                fig,
                output_dir / f"{prefix}.{ext}",
                ax,
                dpi=self.config.dpi,
                numeric_x=False,
                numeric_y=True,
            )
            plt.close(fig)


    @staticmethod
    def _expected_effect_sign(direction_short: str, target_regulation: str) -> int:
        direction_short = str(direction_short).lower()
        target_regulation = str(target_regulation).lower()

        if direction_short == "import" and target_regulation == "activate":
            return 1
        if direction_short == "import" and target_regulation == "repress":
            return -1
        if direction_short == "export" and target_regulation == "activate":
            return -1
        if direction_short == "export" and target_regulation == "repress":
            return 1
        return 1

    @staticmethod
    def _zscore_by_gene(df_rna: pd.DataFrame) -> pd.DataFrame:
        df = df_rna.apply(pd.to_numeric, errors="coerce")
        mean = df.mean(axis=1)
        std = df.std(axis=1, ddof=0).replace(0, np.nan)
        return df.sub(mean, axis=0).div(std, axis=0)

    @staticmethod
    def _safe_wilcoxon_greater(values: pd.Series) -> float:
        values = pd.to_numeric(values, errors="coerce").dropna()
        if len(values) < 3:
            return np.nan
        if np.allclose(values.to_numpy(), 0):
            return np.nan
        try:
            return float(wilcoxon(values, alternative="greater").pvalue)
        except ValueError:
            return np.nan

    def build_target_logfc_table(self, df_points: pd.DataFrame) -> pd.DataFrame:
        if df_points.empty:
            return pd.DataFrame()

        df_pairs = self._build_merged_high_low_pairs(df_points)
        if df_pairs.empty:
            return pd.DataFrame()

        out = df_pairs.copy()
        out["target_logFC"] = (
            pd.to_numeric(out["high_phospho_mean_expression"], errors="coerce")
            - pd.to_numeric(out["low_phospho_mean_expression"], errors="coerce")
        )
        out["expected_sign"] = [
            self._expected_effect_sign(direction_short, target_regulation)
            for direction_short, target_regulation in zip(out["direction_short"], out["target_regulation"])
        ]
        out["directional_logFC"] = out["target_logFC"] * out["expected_sign"]
        out["site_label"] = out.get("source_sites", out["site"])
        out["tf_name"] = out.get("source_tfs", np.nan)
        return out.reset_index(drop=True)

    def summarize_target_logfc(self, df_logfc: pd.DataFrame) -> pd.DataFrame:
        if df_logfc.empty:
            return pd.DataFrame()

        rows: List[Dict[str, object]] = []
        group_cols = ["cancer_type", "direction_short", "target_regulation", "site"]
        for keys, sub in df_logfc.groupby(group_cols, dropna=False):
            cancer_type, direction_short, target_regulation, site = keys
            directional = pd.to_numeric(sub["directional_logFC"], errors="coerce").dropna()
            raw = pd.to_numeric(sub["target_logFC"], errors="coerce").dropna()
            p_value = self._safe_wilcoxon_greater(directional)
            rows.append(
                {
                    "cancer_type": cancer_type,
                    "direction_short": direction_short,
                    "target_regulation": target_regulation,
                    "site": site,
                    "site_label": sub["site_label"].dropna().astype(str).iloc[0] if "site_label" in sub.columns and sub["site_label"].notna().any() else site,
                    "tf_name": sub["tf_name"].dropna().astype(str).iloc[0] if "tf_name" in sub.columns and sub["tf_name"].notna().any() else np.nan,
                    "n_target_genes": int(len(directional)),
                    "mean_target_logFC": float(raw.mean()) if len(raw) else np.nan,
                    "median_target_logFC": float(raw.median()) if len(raw) else np.nan,
                    "mean_directional_logFC": float(directional.mean()) if len(directional) else np.nan,
                    "median_directional_logFC": float(directional.median()) if len(directional) else np.nan,
                    "wilcoxon_p_directional_gt0": p_value,
                    "significance": self._p_to_stars(p_value),
                }
            )

        out = pd.DataFrame(rows)
        if not out.empty:
            out["wilcoxon_q_bh"] = self._bh_adjust(out["wilcoxon_p_directional_gt0"])
            out["significance_bh"] = out["wilcoxon_q_bh"].map(self._p_to_stars)
        return out

    def plot_target_logfc_distributions(self, df_logfc: pd.DataFrame, df_summary: pd.DataFrame, output_dir: Path) -> None:
        if df_logfc.empty:
            return

        plot_dir = output_dir / "target_logfc_distribution"
        plot_dir.mkdir(parents=True, exist_ok=True)

        df = df_logfc.copy()
        df["target_logFC"] = pd.to_numeric(df["target_logFC"], errors="coerce")
        df["directional_logFC"] = pd.to_numeric(df["directional_logFC"], errors="coerce")
        df = df.dropna(subset=["target_logFC", "directional_logFC"])
        if df.empty:
            return

        for (direction_short, target_regulation), sub in df.groupby(["direction_short", "target_regulation"], dropna=False):
            sub = sub.copy()
            if len(sub) < self.config.min_box_points:
                continue

            for value_col, x_label, suffix in [
                ("target_logFC", "Target gene logFC (high phospho minus low phospho)", "raw"),
                ("directional_logFC", "Directional target gene logFC", "directional"),
            ]:
                values = pd.to_numeric(sub[value_col], errors="coerce").dropna()
                if len(values) < self.config.min_box_points:
                    continue

                fig, ax = plt.subplots(figsize=(5.2, 4.2))
                if len(values) >= 8 and float(values.std(ddof=0)) > 0:
                    sns.kdeplot(values, ax=ax, fill=True, linewidth=1.3, color="#5E8FA1", alpha=0.32)
                ax.hist(values, bins=min(35, max(8, int(np.sqrt(len(values))))), density=True, alpha=0.26, color="#5E8FA1", edgecolor="#FFFFFF", linewidth=0.4)
                ax.axvline(0, color="#4A4A4A", linestyle=":", linewidth=1.0)
                ax.axvline(float(values.mean()), color="#C97B49", linewidth=1.3, label=f"Mean = {values.mean():.3f}")
                ax.set_xlabel(x_label, fontsize=15.5)
                ax.set_ylabel("Density", fontsize=15.5)
                ax.tick_params(axis="both", labelsize=14.5)
                ax.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.30)
                ax.grid(axis="x", visible=False)
                ax.legend(frameon=False, fontsize=14, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
                sns.despine(ax=ax)
                plt.tight_layout(rect=[0, 0, 0.84, 1])
                prefix = self._sanitize_filename(f"{direction_short}_{target_regulation}_target_gene_logfc_distribution_{suffix}")
                for ext in ["png", "pdf", "svg"]:
                    _savefig_with_numeric_ticks(
                        fig,
                        plot_dir / f"{prefix}.{ext}",
                        ax,
                        dpi=self.config.dpi,
                        numeric_x=True,
                        numeric_y=True,
                    )
                plt.close(fig)

            cancers = [c for c in self.config.cancer_types if c in set(sub["cancer_type"].astype(str))]
            if len(cancers) > 0:
                positions = []
                data_lists = []
                ticklabels = []
                current_x = 1.0
                for cancer_type in cancers:
                    values = pd.to_numeric(sub.loc[sub["cancer_type"].astype(str).eq(cancer_type), "directional_logFC"], errors="coerce").dropna()
                    if len(values) < self.config.min_box_points:
                        continue
                    positions.append(current_x)
                    data_lists.append(values.tolist())
                    ticklabels.append(cancer_type)
                    current_x += 1.0

                if data_lists:
                    fig_width = max(6.5, 0.55 * len(data_lists) + 2.6)
                    fig, ax = plt.subplots(figsize=(fig_width, 4.3))
                    rng = np.random.default_rng(self.config.random_seed)
                    self._draw_scatter_then_boxplot(
                        ax,
                        data_lists,
                        positions,
                        ["#8FAFBC"] * len(positions),
                        scatter_colors=["#5E8FA1"] * len(positions),
                        widths=0.46,
                        box_alpha=0.86,
                        scatter_s=5.0,
                        scatter_alpha=0.32,
                        jitter_range=0.08,
                        rng=rng,
                        medianprops={"color": "#222222", "linewidth": 1.2},
                    )

                    ax.axhline(0, color="#4A4A4A", linestyle=":", linewidth=1.0)
                    ax.set_xticks(positions)
                    ax.set_xticklabels(ticklabels, rotation=35, ha="right", fontsize=14.5)
                    ax.set_ylabel("Directional target gene logFC", fontsize=15.5)
                    ax.set_xlabel("Cancer type", fontsize=15.5)
                    ax.tick_params(axis="y", labelsize=14.5)
                    ax.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.30)
                    ax.grid(axis="x", visible=False)
                    sns.despine(ax=ax)
                    plt.tight_layout()
                    prefix = self._sanitize_filename(f"{direction_short}_{target_regulation}_target_gene_directional_logfc_by_cancer_boxplot")
                    for ext in ["png", "pdf", "svg"]:
                        _savefig_with_numeric_ticks(
                            fig,
                            plot_dir / f"{prefix}.{ext}",
                            ax,
                            dpi=self.config.dpi,
                            numeric_x=False,
                            numeric_y=True,
                        )
                    plt.close(fig)

    def build_tf_activity_table(
        self,
        df_points: pd.DataFrame,
        expression_mode: Optional[str] = None,
    ) -> pd.DataFrame:
        if df_points.empty:
            return pd.DataFrame()

        if expression_mode is None:
            expression_mode = self._primary_expression_mode()
        df_points = self._filter_points_by_expression_mode(df_points, expression_mode)
        if df_points.empty:
            return pd.DataFrame()

        activity_rows: List[Dict[str, object]] = []
        rna_cache: Dict[str, pd.DataFrame] = {}
        z_cache: Dict[str, pd.DataFrame] = {}
        protein_cache: Dict[str, pd.DataFrame] = {}
        phospho_cache: Dict[str, pd.DataFrame] = {}

        group_cols = ["cancer_type", "direction_short", "site", "tf_name"]
        for keys, sub in df_points.groupby(group_cols, dropna=False):
            cancer_type, direction_short, site, tf_name = keys
            cancer_type = str(cancer_type)
            direction_short = str(direction_short)
            site = str(site)
            tf_name = str(tf_name)

            if cancer_type not in rna_cache:
                rna_cache[cancer_type] = self.load_rna(cancer_type)
            if expression_mode == EXPRESSION_MODE_UNADJUSTED and cancer_type not in z_cache:
                z_cache[cancer_type] = self._zscore_by_gene(rna_cache[cancer_type])
            needs_protein = (
                expression_mode == EXPRESSION_MODE_TF_ADJUSTED
                or self.config.phospho_value_mode == "phospho_minus_protein"
            )
            if needs_protein and cancer_type not in protein_cache:
                protein_cache[cancer_type] = self.load_protein(cancer_type)
            if cancer_type not in phospho_cache:
                phospho_cache[cancer_type] = self.load_phospho(cancer_type)

            df_rna = rna_cache[cancer_type]
            df_protein = protein_cache.get(cancer_type)
            df_phospho = phospho_cache[cancer_type]

            target_mor = (
                pd.to_numeric(sub["target_mor"], errors="coerce")
                if "target_mor" in sub.columns
                else pd.Series(np.nan, index=sub.index)
            )
            pos_genes = sub.loc[target_mor.gt(0), "target_gene_id"].dropna().astype(str).drop_duplicates().tolist()
            neg_genes = sub.loc[target_mor.lt(0), "target_gene_id"].dropna().astype(str).drop_duplicates().tolist()
            activity_genes = sorted(set(pos_genes) | set(neg_genes))
            activity_genes = [g for g in activity_genes if g in df_rna.index]

            if expression_mode == EXPRESSION_MODE_TF_ADJUSTED:
                if df_protein is None:
                    continue
                tf_gene_id = self._site_tf_gene_id(site)
                residual_matrix = self._residualize_target_expression(
                    df_rna,
                    df_protein,
                    activity_genes,
                    tf_gene_id,
                    list(df_rna.columns),
                )
                if residual_matrix.empty:
                    continue
                df_z = self._zscore_by_gene(residual_matrix)
            else:
                df_z = z_cache[cancer_type]

            split_info = self._get_phospho_high_low_samples(
                site,
                df_phospho,
                df_rna,
                df_protein=df_protein,
            )
            if split_info.get("status") != "success":
                continue

            low_samples = [s for s in split_info["low_samples"] if s in df_z.columns]
            high_samples = [s for s in split_info["high_samples"] if s in df_z.columns]
            if len(low_samples) == 0 or len(high_samples) == 0:
                continue

            pos_genes = [g for g in pos_genes if g in df_z.index]
            neg_genes = [g for g in neg_genes if g in df_z.index]

            if len(pos_genes) == 0 and len(neg_genes) == 0:
                continue

            site_label = (
                str(sub["site_label"].dropna().astype(str).iloc[0])
                if "site_label" in sub.columns and sub["site_label"].notna().any()
                else str(self._site_display_label_from_df(sub).iloc[0])
            )
            site_meta = sub.iloc[0]

            for phospho_group, samples in [("low", low_samples), ("high", high_samples)]:
                for sample in samples:
                    pos_score = float(df_z.loc[pos_genes, sample].mean()) if len(pos_genes) > 0 else np.nan
                    neg_score = float(df_z.loc[neg_genes, sample].mean()) if len(neg_genes) > 0 else np.nan

                    if len(pos_genes) > 0 and len(neg_genes) > 0:
                        activity_score = pos_score - neg_score
                    elif len(pos_genes) > 0:
                        activity_score = pos_score
                    else:
                        activity_score = -neg_score

                    activity_rows.append(
                        {
                            "cancer_type": cancer_type,
                            "direction_short": direction_short,
                            "site": site,
                            "site_label": site_label,
                            "tf_name": tf_name,
                            "ACC_ID": site_meta.get("ACC_ID", np.nan),
                            "RESIDUE": site_meta.get("RESIDUE", np.nan),
                            "POSITION": site_meta.get("POSITION", np.nan),
                            "sample": sample,
                            "phospho_group": phospho_group,
                            "expression_mode": expression_mode,
                            "tf_activity_score": activity_score,
                            "positive_target_score": pos_score,
                            "negative_target_score": neg_score,
                            "n_positive_targets": len(pos_genes),
                            "n_negative_targets": len(neg_genes),
                            "n_total_targets": len(set(pos_genes) | set(neg_genes)),
                            "n_low_samples": len(low_samples),
                            "n_high_samples": len(high_samples),
                            "low_cutoff": split_info.get("low_cutoff", np.nan),
                            "high_cutoff": split_info.get("high_cutoff", np.nan),
                        }
                    )

        return pd.DataFrame(activity_rows)

    def build_abundance_impact_target_comparison(self, df_all_points: pd.DataFrame) -> pd.DataFrame:
        if df_all_points.empty or "expression_mode" not in df_all_points.columns:
            return pd.DataFrame()

        rows: List[Dict[str, object]] = []
        group_cols = [
            "cancer_type",
            "direction_short",
            "site",
            "tf_name",
            "target_gene_id",
            "target_gene_name",
            "target_regulation",
        ]
        for keys, sub in df_all_points.groupby(group_cols, dropna=False):
            raw_sub = sub.loc[sub["expression_mode"].astype(str).eq(EXPRESSION_MODE_UNADJUSTED)]
            adj_sub = sub.loc[sub["expression_mode"].astype(str).eq(EXPRESSION_MODE_TF_ADJUSTED)]
            if raw_sub.empty or adj_sub.empty:
                continue

            cancer_type, direction_short, site, tf_name, target_gene_id, target_gene_name, target_regulation = keys
            expected_sign = self._expected_effect_sign(str(direction_short), str(target_regulation))

            def _group_logfc(mode_sub: pd.DataFrame) -> float:
                low = pd.to_numeric(
                    mode_sub.loc[mode_sub["phospho_group"].eq("low"), "group_mean_expression"],
                    errors="coerce",
                )
                high = pd.to_numeric(
                    mode_sub.loc[mode_sub["phospho_group"].eq("high"), "group_mean_expression"],
                    errors="coerce",
                )
                if low.empty or high.empty:
                    return np.nan
                return float(high.mean() - low.mean())

            raw_logfc = _group_logfc(raw_sub)
            adj_logfc = _group_logfc(adj_sub)
            if pd.isna(raw_logfc) and pd.isna(adj_logfc):
                continue

            raw_directional = raw_logfc * expected_sign if pd.notna(raw_logfc) else np.nan
            adj_directional = adj_logfc * expected_sign if pd.notna(adj_logfc) else np.nan
            rows.append(
                {
                    "cancer_type": cancer_type,
                    "direction_short": direction_short,
                    "site": site,
                    "tf_name": tf_name,
                    "target_gene_id": target_gene_id,
                    "target_gene_name": target_gene_name,
                    "target_regulation": target_regulation,
                    "expected_sign": expected_sign,
                    "unadjusted_target_logFC": raw_logfc,
                    "tf_adjusted_target_logFC": adj_logfc,
                    "delta_logFC_adjusted_minus_unadjusted": adj_logfc - raw_logfc
                    if pd.notna(raw_logfc) and pd.notna(adj_logfc)
                    else np.nan,
                    "unadjusted_directional_logFC": raw_directional,
                    "tf_adjusted_directional_logFC": adj_directional,
                    "delta_directional_logFC": adj_directional - raw_directional
                    if pd.notna(raw_directional) and pd.notna(adj_directional)
                    else np.nan,
                    "directional_sign_flip": bool(np.sign(raw_directional) != np.sign(adj_directional))
                    if pd.notna(raw_directional) and pd.notna(adj_directional) and raw_directional != 0 and adj_directional != 0
                    else False,
                }
            )

        return pd.DataFrame(rows)

    def build_abundance_impact_site_summary(self, df_target_compare: pd.DataFrame) -> pd.DataFrame:
        if df_target_compare.empty:
            return pd.DataFrame()

        rows: List[Dict[str, object]] = []
        group_cols = ["cancer_type", "direction_short", "site", "tf_name"]
        for keys, sub in df_target_compare.groupby(group_cols, dropna=False):
            cancer_type, direction_short, site, tf_name = keys
            raw_dir = pd.to_numeric(sub["unadjusted_directional_logFC"], errors="coerce")
            adj_dir = pd.to_numeric(sub["tf_adjusted_directional_logFC"], errors="coerce")
            rows.append(
                {
                    "cancer_type": cancer_type,
                    "direction_short": direction_short,
                    "site": site,
                    "tf_name": tf_name,
                    "n_targets_compared": int(len(sub)),
                    "mean_unadjusted_directional_logFC": float(raw_dir.mean()) if len(raw_dir) else np.nan,
                    "mean_tf_adjusted_directional_logFC": float(adj_dir.mean()) if len(adj_dir) else np.nan,
                    "mean_delta_directional_logFC": float(
                        pd.to_numeric(sub["delta_directional_logFC"], errors="coerce").mean()
                    ),
                    "fraction_directional_sign_flip": float(sub["directional_sign_flip"].mean()),
                    "fraction_unadjusted_directional_gt0": float((raw_dir > 0).mean()) if len(raw_dir) else np.nan,
                    "fraction_tf_adjusted_directional_gt0": float((adj_dir > 0).mean()) if len(adj_dir) else np.nan,
                }
            )
        return pd.DataFrame(rows)

    def save_abundance_impact_reports(
        self,
        df_all_points: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        if self.config.abundance_primary_mode != "dual":
            return
        if df_all_points.empty or "expression_mode" not in df_all_points.columns:
            return

        impact_dir = output_dir / "target_logfc_activity_random_analysis" / "tf_abundance_sensitivity"
        impact_dir.mkdir(parents=True, exist_ok=True)
        df_target_compare = self.build_abundance_impact_target_comparison(df_all_points)
        df_site_compare = self.build_abundance_impact_site_summary(df_target_compare)
        df_target_compare.to_csv(impact_dir / "abundance_impact_target_logfc_comparison.csv", index=False)
        df_site_compare.to_csv(impact_dir / "abundance_impact_site_summary.csv", index=False)
        print(f"Saved TF abundance impact comparison: {impact_dir / 'abundance_impact_target_logfc_comparison.csv'}")

    def compare_tf_activity_high_low(self, df_activity: pd.DataFrame) -> pd.DataFrame:
        if df_activity.empty:
            return pd.DataFrame()

        rows: List[Dict[str, object]] = []
        group_cols = ["cancer_type", "direction_short", "site", "tf_name"]
        for keys, sub in df_activity.groupby(group_cols, dropna=False):
            cancer_type, direction_short, site, tf_name = keys
            low = pd.to_numeric(sub.loc[sub["phospho_group"].eq("low"), "tf_activity_score"], errors="coerce").dropna()
            high = pd.to_numeric(sub.loc[sub["phospho_group"].eq("high"), "tf_activity_score"], errors="coerce").dropna()

            if str(direction_short) == "Import":
                alternative = "greater"
                expected_direction = "High activity > Low activity"
            elif str(direction_short) == "Export":
                alternative = "less"
                expected_direction = "High activity < Low activity"
            else:
                alternative = "two-sided"
                expected_direction = "High activity vs Low activity"

            p_value = np.nan
            if len(low) >= self.config.min_group_samples and len(high) >= self.config.min_group_samples:
                try:
                    p_value = float(mannwhitneyu(high, low, alternative=alternative).pvalue)
                except ValueError:
                    p_value = np.nan

            rows.append(
                {
                    "cancer_type": cancer_type,
                    "direction_short": direction_short,
                    "site": site,
                    "site_label": (
                        str(sub["site_label"].dropna().astype(str).iloc[0])
                        if "site_label" in sub.columns and sub["site_label"].notna().any()
                        else str(site)
                    ),
                    "tf_name": tf_name,
                    "n_low_samples": len(low),
                    "n_high_samples": len(high),
                    "mean_low_activity": float(low.mean()) if len(low) else np.nan,
                    "mean_high_activity": float(high.mean()) if len(high) else np.nan,
                    "delta_high_minus_low": float(high.mean() - low.mean()) if len(low) and len(high) else np.nan,
                    "expected_direction": expected_direction,
                    "alternative": alternative,
                    "mannwhitney_p": p_value,
                    "significance": self._p_to_stars(p_value),
                }
            )

        out = pd.DataFrame(rows)
        if not out.empty:
            out = self._finalize_mannwhitney_significance(out, p_col="mannwhitney_p")
        return out

    def compare_tf_activity_by_cancer_direction(self, df_activity: pd.DataFrame) -> pd.DataFrame:
        if df_activity.empty:
            return pd.DataFrame()

        rows: List[Dict[str, object]] = []
        group_cols = ["cancer_type", "direction_short"]
        for keys, sub in df_activity.groupby(group_cols, dropna=False):
            cancer_type, direction_short = keys
            low = pd.to_numeric(sub.loc[sub["phospho_group"].eq("low"), "tf_activity_score"], errors="coerce").dropna()
            high = pd.to_numeric(sub.loc[sub["phospho_group"].eq("high"), "tf_activity_score"], errors="coerce").dropna()

            if str(direction_short) == "Import":
                alternative = "greater"
            elif str(direction_short) == "Export":
                alternative = "less"
            else:
                alternative = "two-sided"

            p_value = np.nan
            if len(low) >= self.config.min_group_samples and len(high) >= self.config.min_group_samples:
                try:
                    p_value = float(mannwhitneyu(high, low, alternative=alternative).pvalue)
                except ValueError:
                    p_value = np.nan

            rows.append(
                {
                    "cancer_type": cancer_type,
                    "direction_short": direction_short,
                    "n_low_points": len(low),
                    "n_high_points": len(high),
                    "mean_low_activity": float(low.mean()) if len(low) else np.nan,
                    "mean_high_activity": float(high.mean()) if len(high) else np.nan,
                    "delta_high_minus_low": float(high.mean() - low.mean()) if len(low) and len(high) else np.nan,
                    "alternative": alternative,
                    "mannwhitney_p": p_value,
                    "significance": self._p_to_stars(p_value),
                }
            )

        out = pd.DataFrame(rows)
        if not out.empty:
            out["significance_raw"] = out["mannwhitney_p"].map(self._p_to_stars)
            out["mannwhitney_q_bh"] = np.nan
            for _, idx in out.groupby(["direction_short"], dropna=False).groups.items():
                out.loc[idx, "mannwhitney_q_bh"] = self._bh_adjust(out.loc[idx, "mannwhitney_p"])
            out["significance_bh"] = out["mannwhitney_q_bh"].map(self._p_to_stars)
            if self.config.use_bh_pvalue_correction:
                out["significance"] = out["significance_bh"]
            else:
                out["significance"] = out["significance_raw"]
        return out

    def plot_cancer_direction_activity_boxplot(
        self,
        df_activity: pd.DataFrame,
        df_stats: pd.DataFrame,
        cancer_type: str,
        direction_short: str,
        output_dir: Path,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        df_plot = df_activity.copy()
        if df_plot.empty:
            return

        df_plot = df_plot[
            df_plot["cancer_type"].astype(str).eq(str(cancer_type))
            & df_plot["direction_short"].astype(str).eq(str(direction_short))
        ].copy()
        if df_plot.empty:
            return

        if "site_label" not in df_plot.columns or df_plot["site_label"].isna().all():
            df_plot["site_label"] = df_plot["site"].astype(str)

        group_color_map = {
            "low": "#8FAFBC",
            "high": "#E3A07A",
        }
        group_label_map = {
            "low": "Low phospho",
            "high": "High phospho",
        }
        site_label_color_map = {
            "known_positive": "#C97B49",
            "new_predicted": "#5E8FA1",
        }
        site_label_name_map = {
            "known_positive": "Known positive site",
            "new_predicted": "New predicted site",
        }

        df_stats_for_plot = self._prepare_activity_site_stats_for_plotting(df_stats)
        site_order = self._ordered_sites_for_activity(df_stats_for_plot)

        positions = []
        data_lists = []
        box_colors = []
        scatter_colors = []
        x_centers = []
        ticklabels = []
        ticklabel_colors = []
        plotted_site_statuses = set()
        site_stat_pos = {}
        current_x = 1.0
        pair_offset = 0.16

        for site in site_order:
            site_sub = df_plot[df_plot["site"].astype(str).eq(str(site))].copy()
            if site_sub.empty:
                continue

            low_values = pd.to_numeric(
                site_sub.loc[site_sub["phospho_group"].eq("low"), "tf_activity_score"],
                errors="coerce",
            ).dropna().tolist()
            high_values = pd.to_numeric(
                site_sub.loc[site_sub["phospho_group"].eq("high"), "tf_activity_score"],
                errors="coerce",
            ).dropna().tolist()

            if (
                len(low_values) < self.config.min_group_samples
                or len(high_values) < self.config.min_group_samples
            ):
                continue

            site_label = str(site_sub["site_label"].dropna().astype(str).iloc[0])
            positions.extend([current_x - pair_offset, current_x + pair_offset])
            data_lists.extend([low_values, high_values])
            box_colors.extend([group_color_map["low"], group_color_map["high"]])
            scatter_colors.extend([group_color_map["low"], group_color_map["high"]])

            site_status = self.site_label_status(site_sub, direction_short)
            x_centers.append(current_x)
            ticklabels.append(site_label)
            ticklabel_colors.append(site_label_color_map.get(site_status, site_label_color_map["new_predicted"]))
            plotted_site_statuses.add(site_status)
            site_stat_pos[site] = current_x
            current_x += 1.0

        if not data_lists:
            return

        fig_width = max(10.0, 0.50 * len(x_centers) + 3.6)
        fig, ax = plt.subplots(figsize=(fig_width, 4.9))

        rng = np.random.default_rng(42)
        self._draw_scatter_then_boxplot(
            ax,
            data_lists,
            positions,
            box_colors,
            scatter_colors=scatter_colors,
            widths=0.23,
            box_alpha=0.88,
            scatter_s=3.5,
            scatter_alpha=0.35,
            jitter_range=0.045,
            rng=rng,
        )

        ax.axhline(0, color="#4A4A4A", linestyle=":", linewidth=1.0)
        ax.set_xticks(x_centers)
        ax.set_xticklabels(ticklabels, rotation=55, ha="right", fontsize=12.6)
        for tick_label, tick_color in zip(ax.get_xticklabels(), ticklabel_colors):
            tick_label.set_color(tick_color)
            if tick_color == site_label_color_map["known_positive"]:
                tick_label.set_fontweight("bold")

        ax.set_ylabel("Signed regulon activity score", fontsize=15.5)
        ax.set_xlabel("Phosphosite", fontsize=15.5)
        ax.tick_params(axis="y", labelsize=14.5)
        ax.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.30)
        ax.grid(axis="x", visible=False)
        sns.despine(ax=ax)

        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin

        if not df_stats_for_plot.empty:
            for _, stat_row in df_stats_for_plot.iterrows():
                sig_label = self._plot_significance_from_row(stat_row)
                site_key = str(stat_row.get("site", ""))
                if sig_label == "ns" or site_key not in site_stat_pos:
                    continue
                ax.text(
                    site_stat_pos[site_key],
                    ymax + y_range * 0.095,
                    sig_label,
                    ha="center",
                    va="bottom",
                    fontsize=16.5,
                    fontweight="normal",
                    color="#2F2F2F",
                )

        ax.set_ylim(ymin, ymax + y_range * 0.17)
        ax.set_title(
            f"{cancer_type} {direction_short} signed TF activity score",
            fontsize=16.5,
            pad=12,
        )

        from matplotlib.patches import Patch

        handles = [
            Patch(
                facecolor=group_color_map["low"],
                edgecolor="#3A3A3A",
                linewidth=0.85,
                alpha=0.88,
                label=group_label_map["low"],
            ),
            Patch(
                facecolor=group_color_map["high"],
                edgecolor="#3A3A3A",
                linewidth=0.85,
                alpha=0.88,
                label=group_label_map["high"],
            ),
        ]
        site_legend_statuses = [status for status in ["known_positive", "new_predicted"] if status in plotted_site_statuses]
        for status in site_legend_statuses:
            handles.append(
                Patch(
                    facecolor="none",
                    edgecolor="none",
                    alpha=0.0,
                    label=site_label_name_map[status],
                )
            )

        legend = ax.legend(
            handles=handles,
            frameon=False,
            fontsize=14.2,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            handlelength=1.2,
            handletextpad=0.55,
            labelspacing=0.45,
        )
        for text in legend.get_texts():
            label = text.get_text()
            if label == site_label_name_map["known_positive"]:
                text.set_color(site_label_color_map["known_positive"])
            elif label == site_label_name_map["new_predicted"]:
                text.set_color(site_label_color_map["new_predicted"])

        plt.tight_layout(rect=[0, 0, 0.88, 1])

        prefix = self._sanitize_filename(
            f"{cancer_type}_{direction_short}_sitewise_high_low_phospho_tf_activity_score_boxplot"
        )
        for ext in ["png", "pdf", "svg"]:
            _savefig_with_numeric_ticks(
                fig,
                output_dir / f"{prefix}.{ext}",
                ax,
                dpi=self.config.dpi,
                numeric_x=False,
                numeric_y=True,
            )
        plt.close(fig)

    def plot_all_sitewise_activity_boxplots(
        self,
        df_activity: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        if df_activity.empty:
            return

        plot_dir = output_dir / "figureB_sitewise_tf_activity_score"
        plot_dir.mkdir(parents=True, exist_ok=True)

        df = df_activity.copy()
        df["tf_activity_score"] = pd.to_numeric(df["tf_activity_score"], errors="coerce")
        df = df.dropna(subset=["tf_activity_score"])
        if df.empty:
            return

        if "site_label" not in df.columns or df["site_label"].isna().all():
            df["site_label"] = df["site"].astype(str)

        df.to_csv(plot_dir / "FigureB_signed_tf_activity_score_points_all.csv", index=False)

        stats = self.compare_tf_activity_high_low(df)
        stats.to_csv(plot_dir / "high_low_phospho_activity_comparison_by_site_all.csv", index=False)
        stats_for_plot = self._prepare_activity_site_stats_for_plotting(stats)
        stats_for_plot.to_csv(plot_dir / "high_low_phospho_activity_comparison_by_site_plotted.csv", index=False)

        if stats_for_plot.empty:
            print(
                f"No activity sites passed min_group_samples={self.config.min_group_samples}; "
                "no sitewise activity boxplots will be drawn."
            )
            return

        keep_cols = ["cancer_type", "direction_short", "site"]
        keep_keys = stats_for_plot[keep_cols].drop_duplicates()
        df_for_plot = df.merge(keep_keys, on=keep_cols, how="inner")
        df_for_plot.to_csv(plot_dir / "FigureB_signed_tf_activity_score_points_plotted.csv", index=False)

        for (cancer_type, direction_short), df_sub in df_for_plot.groupby(["cancer_type", "direction_short"]):
            stat_sub = stats_for_plot[
                stats_for_plot["cancer_type"].astype(str).eq(str(cancer_type))
                & stats_for_plot["direction_short"].astype(str).eq(str(direction_short))
            ].copy()
            self.plot_cancer_direction_activity_boxplot(
                df_sub,
                stat_sub,
                str(cancer_type),
                str(direction_short),
                plot_dir,
            )

    def plot_tf_activity_score_boxplots(self, df_activity: pd.DataFrame, df_stats: pd.DataFrame, output_dir: Path) -> None:
        if df_activity.empty:
            return

        plot_dir = output_dir / "tf_activity_score_boxplots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        df = df_activity.copy()
        df["tf_activity_score"] = pd.to_numeric(df["tf_activity_score"], errors="coerce")
        df = df.dropna(subset=["tf_activity_score"])
        if df.empty:
            return

        group_color_map = {"low": "#95B0B5", "high": "#E7A983"}
        group_label_map = {"low": "Low phospho", "high": "High phospho"}

        for direction_short, sub in df.groupby("direction_short", dropna=False):
            sub = sub.copy()
            cancers = [c for c in self.config.cancer_types if c in set(sub["cancer_type"].astype(str))]
            if not cancers:
                cancers = sorted(sub["cancer_type"].astype(str).unique())

            positions = []
            data_lists = []
            box_colors = []
            x_centers = []
            ticklabels = []
            stat_pos = {}
            current_x = 1.0
            pair_offset = 0.17

            for cancer_type in cancers:
                cancer_sub = sub[sub["cancer_type"].astype(str).eq(cancer_type)].copy()
                low_values = pd.to_numeric(cancer_sub.loc[cancer_sub["phospho_group"].eq("low"), "tf_activity_score"], errors="coerce").dropna().tolist()
                high_values = pd.to_numeric(cancer_sub.loc[cancer_sub["phospho_group"].eq("high"), "tf_activity_score"], errors="coerce").dropna().tolist()
                if len(low_values) < self.config.min_group_samples or len(high_values) < self.config.min_group_samples:
                    continue
                positions.extend([current_x - pair_offset, current_x + pair_offset])
                data_lists.extend([low_values, high_values])
                box_colors.extend([group_color_map["low"], group_color_map["high"]])
                x_centers.append(current_x)
                ticklabels.append(cancer_type)
                stat_pos[cancer_type] = current_x
                current_x += 1.0

            if not data_lists:
                continue

            fig_width = max(7.0, 0.62 * len(x_centers) + 2.8)
            fig, ax = plt.subplots(figsize=(fig_width, 4.6))
            rng = np.random.default_rng(self.config.random_seed)
            self._draw_scatter_then_boxplot(
                ax,
                data_lists,
                positions,
                box_colors,
                scatter_colors=box_colors,
                widths=0.26,
                box_alpha=0.88,
                scatter_s=4.0,
                scatter_alpha=0.26,
                jitter_range=0.05,
                rng=rng,
                medianprops={"color": "#222222", "linewidth": 1.15},
            )

            ax.axhline(0, color="#4A4A4A", linestyle=":", linewidth=1.0)
            ax.set_xticks(x_centers)
            ax.set_xticklabels(ticklabels, rotation=35, ha="right", fontsize=14.5)
            ax.set_ylabel("Signed regulon activity score", fontsize=15.5)
            ax.set_xlabel("Cancer type", fontsize=15.5)
            ax.tick_params(axis="y", labelsize=14.5)
            ax.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.30)
            ax.grid(axis="x", visible=False)
            sns.despine(ax=ax)

            ymin, ymax = ax.get_ylim()
            y_range = ymax - ymin
            stat_sub = df_stats[df_stats["direction_short"].astype(str).eq(str(direction_short))].copy() if not df_stats.empty else pd.DataFrame()
            for _, stat_row in stat_sub.iterrows():
                cancer_type = str(stat_row.get("cancer_type", ""))
                sig_label = self._plot_significance_from_row(stat_row)
                if sig_label == "ns" or cancer_type not in stat_pos:
                    continue
                ax.text(stat_pos[cancer_type], ymax + y_range * 0.055, sig_label, ha="center", va="bottom", fontsize=17, color="#2F2F2F")
            ax.set_ylim(ymin, ymax + y_range * 0.14)

            handles = [
                plt.Line2D([0], [0], marker="s", linestyle="", markersize=7, markerfacecolor=group_color_map["low"], markeredgecolor="#3A3A3A", label=group_label_map["low"]),
                plt.Line2D([0], [0], marker="s", linestyle="", markersize=7, markerfacecolor=group_color_map["high"], markeredgecolor="#3A3A3A", label=group_label_map["high"]),
            ]
            ax.legend(handles=handles, frameon=False, fontsize=14, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
            plt.tight_layout(rect=[0, 0, 0.88, 1])
            prefix = self._sanitize_filename(f"{direction_short}_tf_activity_score_by_cancer_boxplot")
            for ext in ["png", "pdf", "svg"]:
                _savefig_with_numeric_ticks(
                    fig,
                    plot_dir / f"{prefix}.{ext}",
                    ax,
                    dpi=self.config.dpi,
                    numeric_x=False,
                    numeric_y=True,
                )
            plt.close(fig)

    def build_random_logfc_table(
        self,
        df_logfc: pd.DataFrame,
        n_random_sets: int = 10,
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if df_logfc.empty or n_random_sets <= 0:
            return pd.DataFrame(), pd.DataFrame()

        rng = np.random.default_rng(seed)
        random_rows: List[Dict[str, object]] = []
        summary_rows: List[Dict[str, object]] = []
        rna_cache: Dict[str, pd.DataFrame] = {}
        phospho_cache: Dict[str, pd.DataFrame] = {}
        protein_cache: Dict[str, pd.DataFrame] = {}

        group_cols = ["cancer_type", "direction_short", "target_regulation", "site"]
        for keys, sub in df_logfc.groupby(group_cols, dropna=False):
            cancer_type, direction_short, target_regulation, site = keys
            cancer_type = str(cancer_type)
            direction_short = str(direction_short)
            target_regulation = str(target_regulation)
            site = str(site)

            if cancer_type not in rna_cache:
                rna_cache[cancer_type] = self.load_rna(cancer_type)
            if cancer_type not in phospho_cache:
                phospho_cache[cancer_type] = self.load_phospho(cancer_type)
            df_protein = None
            if self._needs_protein_matrix():
                if cancer_type not in protein_cache:
                    protein_cache[cancer_type] = self.load_protein(cancer_type)
                df_protein = protein_cache[cancer_type]

            df_rna = rna_cache[cancer_type]
            df_phospho = phospho_cache[cancer_type]
            split_info = self._get_phospho_high_low_samples(
                site,
                df_phospho,
                df_rna,
                df_protein=df_protein,
            )
            if split_info.get("status") != "success":
                continue

            low_samples = [s for s in split_info["low_samples"] if s in df_rna.columns]
            high_samples = [s for s in split_info["high_samples"] if s in df_rna.columns]
            if len(low_samples) == 0 or len(high_samples) == 0:
                continue

            target_genes = sub["target_gene_id"].dropna().astype(str).drop_duplicates().tolist()
            target_genes = [g for g in target_genes if g in df_rna.index]
            n_targets = len(target_genes)
            if n_targets < self.config.min_box_points:
                continue

            target_gene_set = set(target_genes)
            background_genes = [g for g in df_rna.index.astype(str) if g not in target_gene_set]
            if len(background_genes) < n_targets:
                continue

            observed_logfc = pd.to_numeric(sub["target_logFC"], errors="coerce").dropna()
            observed_directional = pd.to_numeric(sub["directional_logFC"], errors="coerce").dropna()
            if len(observed_directional) == 0:
                continue

            expected_sign = self._expected_effect_sign(direction_short, target_regulation)
            observed_mean_logfc = float(observed_logfc.mean()) if len(observed_logfc) else np.nan
            observed_directional_mean = float(observed_directional.mean())
            random_directional_values = []

            for iteration in range(n_random_sets):
                sampled_genes = rng.choice(background_genes, size=n_targets, replace=False)
                random_logfc_by_gene = (
                    df_rna.loc[sampled_genes, high_samples].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                    - df_rna.loc[sampled_genes, low_samples].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                )
                random_mean_logfc = float(random_logfc_by_gene.mean())
                random_directional_mean_logfc = random_mean_logfc * expected_sign
                random_directional_values.append(random_directional_mean_logfc)
                random_rows.append(
                    {
                        "cancer_type": cancer_type,
                        "direction_short": direction_short,
                        "target_regulation": target_regulation,
                        "site": site,
                        "site_label": sub["site_label"].dropna().astype(str).iloc[0] if "site_label" in sub.columns and sub["site_label"].notna().any() else site,
                        "tf_name": sub["tf_name"].dropna().astype(str).iloc[0] if "tf_name" in sub.columns and sub["tf_name"].notna().any() else np.nan,
                        "iteration": iteration,
                        "n_targets": n_targets,
                        "random_mean_logFC": random_mean_logfc,
                        "random_directional_mean_logFC": random_directional_mean_logfc,
                    }
                )

            random_array = np.asarray(random_directional_values, dtype=float)
            empirical_p = float((np.sum(random_array >= observed_directional_mean) + 1) / (len(random_array) + 1)) if len(random_array) else np.nan
            z_score = float((observed_directional_mean - np.nanmean(random_array)) / np.nanstd(random_array, ddof=1)) if len(random_array) > 1 and np.nanstd(random_array, ddof=1) > 0 else np.nan
            summary_rows.append(
                {
                    "cancer_type": cancer_type,
                    "direction_short": direction_short,
                    "target_regulation": target_regulation,
                    "site": site,
                    "site_label": sub["site_label"].dropna().astype(str).iloc[0] if "site_label" in sub.columns and sub["site_label"].notna().any() else site,
                    "tf_name": sub["tf_name"].dropna().astype(str).iloc[0] if "tf_name" in sub.columns and sub["tf_name"].notna().any() else np.nan,
                    "n_targets": n_targets,
                    "observed_mean_logFC": observed_mean_logfc,
                    "observed_directional_mean_logFC": observed_directional_mean,
                    "random_mean_directional_logFC": float(np.nanmean(random_array)) if len(random_array) else np.nan,
                    "random_median_directional_logFC": float(np.nanmedian(random_array)) if len(random_array) else np.nan,
                    "random_sd_directional_logFC": float(np.nanstd(random_array, ddof=1)) if len(random_array) > 1 else np.nan,
                    "observed_vs_random_z": z_score,
                    "empirical_p_observed_gt_random": empirical_p,
                }
            )

        df_random = pd.DataFrame(random_rows)
        df_summary = pd.DataFrame(summary_rows)
        if not df_summary.empty:
            df_summary["empirical_q_bh"] = self._bh_adjust(df_summary["empirical_p_observed_gt_random"])
            df_summary["significance_bh"] = df_summary["empirical_q_bh"].map(self._p_to_stars)
            df_summary["significance_raw"] = df_summary["empirical_p_observed_gt_random"].map(self._p_to_stars)
            if self.config.use_bh_pvalue_correction:
                df_summary["significance"] = df_summary["significance_bh"]
            else:
                df_summary["significance"] = df_summary["significance_raw"]
        return df_random, df_summary

    def plot_random_logfc_distributions(self, df_random: pd.DataFrame, df_summary: pd.DataFrame, output_dir: Path) -> None:
        if df_random.empty or df_summary.empty:
            return

        plot_dir = output_dir / "observed_vs_random_target_logfc"
        plot_dir.mkdir(parents=True, exist_ok=True)

        df_random = df_random.copy()
        df_summary = df_summary.copy()
        df_random["random_directional_mean_logFC"] = pd.to_numeric(df_random["random_directional_mean_logFC"], errors="coerce")
        df_summary["observed_directional_mean_logFC"] = pd.to_numeric(df_summary["observed_directional_mean_logFC"], errors="coerce")

        for (direction_short, target_regulation), random_sub in df_random.groupby(["direction_short", "target_regulation"], dropna=False):
            summary_sub = df_summary[
                df_summary["direction_short"].astype(str).eq(str(direction_short))
                & df_summary["target_regulation"].astype(str).eq(str(target_regulation))
            ].copy()
            random_values = pd.to_numeric(random_sub["random_directional_mean_logFC"], errors="coerce").dropna()
            observed_values = pd.to_numeric(summary_sub["observed_directional_mean_logFC"], errors="coerce").dropna()
            if len(random_values) < self.config.min_box_points or len(observed_values) == 0:
                continue

            fig, ax = plt.subplots(figsize=(5.5, 4.2))
            if len(random_values) >= 8 and float(random_values.std(ddof=0)) > 0:
                sns.kdeplot(random_values, ax=ax, fill=True, linewidth=1.2, color="#8FAFBC", alpha=0.34, label="Random targets")
            ax.hist(random_values, bins=min(40, max(10, int(np.sqrt(len(random_values))))), density=True, alpha=0.22, color="#8FAFBC", edgecolor="#FFFFFF", linewidth=0.35)
            ax.axvline(float(observed_values.median()), color="#C97B49", linewidth=1.4, label=f"Observed median = {observed_values.median():.3f}")
            ax.axvline(0, color="#4A4A4A", linestyle=":", linewidth=1.0)
            ylim = ax.get_ylim()
            rug_y = ylim[1] * 0.018
            ax.scatter(observed_values, np.full(len(observed_values), rug_y), marker="|", s=55, color="#C97B49", alpha=0.75, label="Observed sites")
            ax.set_ylim(ylim)
            ax.set_xlabel("Mean directional logFC", fontsize=15.5)
            ax.set_ylabel("Density", fontsize=15.5)
            ax.tick_params(axis="both", labelsize=14.5)
            ax.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.30)
            ax.grid(axis="x", visible=False)
            ax.legend(frameon=False, fontsize=14, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
            sns.despine(ax=ax)
            plt.tight_layout(rect=[0, 0, 0.82, 1])
            prefix = self._sanitize_filename(f"{direction_short}_{target_regulation}_observed_vs_random_directional_logfc_distribution")
            for ext in ["png", "pdf", "svg"]:
                _savefig_with_numeric_ticks(
                    fig,
                    plot_dir / f"{prefix}.{ext}",
                    ax,
                    dpi=self.config.dpi,
                    numeric_x=True,
                    numeric_y=True,
                )
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(4.4, 4.3))
            data_lists = [random_values.tolist(), observed_values.tolist()]
            positions = [1.0, 2.0]
            box_colors = ["#8FAFBC", "#C97B49"]
            scatter_specs = [
                (1.0, random_values.sample(min(len(random_values), 1200), random_state=self.config.random_seed).tolist(), "#8FAFBC", 3.5, 0.18),
                (2.0, observed_values.tolist(), "#C97B49", 13.0, 0.78),
            ]
            rng = np.random.default_rng(self.config.random_seed)
            for pos, values, color, size, alpha in scatter_specs:
                jitter = rng.uniform(-0.06, 0.06, size=len(values))
                ax.scatter(
                    np.full(len(values), pos) + jitter,
                    values,
                    s=size,
                    alpha=alpha,
                    edgecolors="none",
                    color=color,
                    zorder=1,
                )
            bp = ax.boxplot(
                data_lists,
                positions=positions,
                widths=0.45,
                showfliers=False,
                patch_artist=True,
                medianprops={"color": "#222222", "linewidth": 1.2},
                whiskerprops={"color": "#4A4A4A", "linewidth": 0.85},
                capprops={"color": "#4A4A4A", "linewidth": 0.85},
                zorder=5,
            )
            for box, color in zip(bp["boxes"], box_colors):
                box.set(facecolor=color, edgecolor="#3A3A3A", linewidth=0.85, alpha=0.86, zorder=5)
            for element_key in ("whiskers", "caps", "medians"):
                for artist in bp.get(element_key, []):
                    artist.set(zorder=6)
            ax.axhline(0, color="#4A4A4A", linestyle=":", linewidth=1.0)
            ax.set_xticks(positions)
            ax.set_xticklabels(["Random", "Observed"], fontsize=14.8)
            ax.set_ylabel("Mean directional logFC", fontsize=15.5)
            ax.tick_params(axis="y", labelsize=14.5)
            ax.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.30)
            ax.grid(axis="x", visible=False)
            sns.despine(ax=ax)
            plt.tight_layout()
            prefix = self._sanitize_filename(f"{direction_short}_{target_regulation}_observed_vs_random_directional_logfc_boxplot")
            for ext in ["png", "pdf", "svg"]:
                _savefig_with_numeric_ticks(
                    fig,
                    plot_dir / f"{prefix}.{ext}",
                    ax,
                    dpi=self.config.dpi,
                    numeric_x=False,
                    numeric_y=True,
                )
            plt.close(fig)

    def plot_all_boxplots(self, df_points: pd.DataFrame, output_dir: Path) -> None:
        if df_points.empty:
            return

        plot_dir = output_dir / "high_low_phospho_boxplots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        df_points = df_points.copy()
        df_points["site_label"] = self._site_display_label_from_df(df_points)
        df_points.to_csv(plot_dir / "target_gene_high_low_expression_points_all.csv", index=False)

        summary = (
            df_points.groupby(
                ["cancer_type", "direction_short", "target_regulation", "site", "site_label", "tf_name", "phospho_group"],
                as_index=False,
            )
            .agg(
                n_points=("group_mean_expression", "size"),
                n_target_genes=("target_gene_id", "nunique"),
                mean_expression=("group_mean_expression", "mean"),
                median_expression=("group_mean_expression", "median"),
            )
        )
        summary.to_csv(plot_dir / "target_gene_expression_summary_by_site_and_phospho_group_all.csv", index=False)

        stats = self._compare_high_low_by_site(df_points)
        stats = self._add_p_value_correction(
            stats,
            group_cols=["cancer_type", "direction_short", "target_regulation"],
            p_col="wilcoxon_p_expected",
            q_col="wilcoxon_q_bh",
        )
        stats.to_csv(plot_dir / "high_low_phospho_comparison_by_site_all.csv", index=False)
        stats_for_plot = self._prepare_stats_for_plotting(stats)
        stats_for_plot.to_csv(plot_dir / "high_low_phospho_comparison_by_site_plotted.csv", index=False)

        if stats_for_plot.empty:
            print(f"No sites passed min_box_points={self.config.min_box_points}; no boxplots will be drawn.")
            return

        keep_cols = ["cancer_type", "direction_short", "target_regulation", "site"]
        keep_keys = stats_for_plot[keep_cols].drop_duplicates()
        df_points_for_plot = df_points.merge(keep_keys, on=keep_cols, how="inner")
        df_points_for_plot.to_csv(plot_dir / "target_gene_high_low_expression_points_plotted.csv", index=False)

        plotted_summary = (
            df_points_for_plot.groupby(
                ["cancer_type", "direction_short", "target_regulation", "site", "site_label", "tf_name", "phospho_group"],
                as_index=False,
            )
            .agg(
                n_points=("group_mean_expression", "size"),
                n_target_genes=("target_gene_id", "nunique"),
                mean_expression=("group_mean_expression", "mean"),
                median_expression=("group_mean_expression", "median"),
            )
        )
        plotted_summary.to_csv(plot_dir / "target_gene_expression_summary_by_site_and_phospho_group_plotted.csv", index=False)

        for (cancer_type, direction_short), df_sub in df_points_for_plot.groupby(["cancer_type", "direction_short"]):
            stat_sub = stats_for_plot[
                stats_for_plot["cancer_type"].eq(cancer_type)
                & stats_for_plot["direction_short"].eq(direction_short)
            ].copy()
            self.plot_cancer_direction_boxplot(
                df_sub,
                stat_sub,
                str(cancer_type),
                str(direction_short),
                plot_dir,
            )




    def build_pooled_matched_random_control(
        self,
        df_logfc: pd.DataFrame,
        n_random_sets: int = 10,
        seed: int = 42,
        max_random_gene_rows_per_group: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if df_logfc.empty or n_random_sets <= 0:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        rng = np.random.default_rng(seed)
        if max_random_gene_rows_per_group is None:
            max_random_gene_rows_per_group = self.config.max_random_gene_rows_per_group

        observed_rows: List[Dict[str, object]] = []
        random_gene_rows: List[Dict[str, object]] = []
        random_global_rows: List[Dict[str, object]] = []
        summary_rows: List[Dict[str, object]] = []

        rna_cache: Dict[str, pd.DataFrame] = {}
        phospho_cache: Dict[str, pd.DataFrame] = {}
        protein_cache: Dict[str, pd.DataFrame] = {}

        df = df_logfc.copy()
        df["directional_logFC"] = pd.to_numeric(df["directional_logFC"], errors="coerce")
        df["target_logFC"] = pd.to_numeric(df["target_logFC"], errors="coerce")
        df = df.dropna(subset=["directional_logFC"])
        if df.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        group_cols = ["direction_short", "target_regulation"]
        site_cols = ["cancer_type", "direction_short", "target_regulation", "site"]
        grouped_regulations = list(df.groupby(group_cols, dropna=False))
        total_regulation_groups = len(grouped_regulations)

        for regulation_idx, (group_keys, group_sub) in enumerate(grouped_regulations, start=1):
            direction_short, target_regulation = group_keys
            direction_short = str(direction_short)
            target_regulation = str(target_regulation)
            direction_label = self._direction_label(direction_short)
            print(
                f"Processing pooled random control [{regulation_idx}/{total_regulation_groups}] "
                f"{direction_label} | {target_regulation}"
            )

            observed_values_for_group: List[float] = []
            random_iteration_sum = np.zeros(n_random_sets, dtype=float)
            random_iteration_count = np.zeros(n_random_sets, dtype=float)
            random_gene_rows_kept_by_cancer: Dict[str, int] = {}
            n_sites_used = 0
            n_target_gene_values = 0

            for _, obs_row in group_sub.iterrows():
                value = obs_row.get("directional_logFC", np.nan)
                if pd.isna(value):
                    continue
                observed_value = float(value)
                observed_values_for_group.append(observed_value)
                observed_rows.append(
                    {
                        "source": "Observed targets",
                        "direction_short": direction_short,
                        "target_regulation": target_regulation,
                        "cancer_type": obs_row.get("cancer_type", np.nan),
                        "site": obs_row.get("site", np.nan),
                        "site_label": obs_row.get("site_label", obs_row.get("site", np.nan)),
                        "tf_name": obs_row.get("tf_name", obs_row.get("source_tfs", np.nan)),
                        "target_gene_id": obs_row.get("target_gene_id", np.nan),
                        "target_gene_name": obs_row.get("target_gene_name", np.nan),
                        "directional_logFC": observed_value,
                        "target_logFC": obs_row.get("target_logFC", np.nan),
                        "iteration": np.nan,
                    }
                )

            site_groups = list(group_sub.groupby(site_cols, dropna=False))
            total_sites = len(site_groups)
            for site_idx, (site_keys, site_sub) in enumerate(site_groups, start=1):
                cancer_type, _, _, site = site_keys
                cancer_type = str(cancer_type)
                site = str(site)
                site_label = (
                    site_sub["site_label"].dropna().astype(str).iloc[0]
                    if "site_label" in site_sub.columns and site_sub["site_label"].notna().any()
                    else site
                )
                tf_name = (
                    site_sub["tf_name"].dropna().astype(str).iloc[0]
                    if "tf_name" in site_sub.columns and site_sub["tf_name"].notna().any()
                    else "TF"
                )
                print(
                    f"Processing pooled random control site [{site_idx}/{total_sites}] "
                    f"{cancer_type} | {direction_label} | {tf_name} | {site_label}"
                )

                if cancer_type not in rna_cache:
                    rna_cache[cancer_type] = self.load_rna(cancer_type)
                if cancer_type not in phospho_cache:
                    phospho_cache[cancer_type] = self.load_phospho(cancer_type)
                df_protein = None
                if self._needs_protein_matrix():
                    if cancer_type not in protein_cache:
                        protein_cache[cancer_type] = self.load_protein(cancer_type)
                    df_protein = protein_cache[cancer_type]

                df_rna = rna_cache[cancer_type]
                df_phospho = phospho_cache[cancer_type]
                split_info = self._get_phospho_high_low_samples(
                    site,
                    df_phospho,
                    df_rna,
                    df_protein=df_protein,
                )
                if split_info.get("status") != "success":
                    print(f"  Skipped pooled random control: phospho split status={split_info.get('status')}")
                    continue

                low_samples = [sample for sample in split_info["low_samples"] if sample in df_rna.columns]
                high_samples = [sample for sample in split_info["high_samples"] if sample in df_rna.columns]
                if len(low_samples) == 0 or len(high_samples) == 0:
                    print("  Skipped pooled random control: no matched high/low RNA samples")
                    continue

                target_genes = (
                    site_sub["target_gene_id"]
                    .dropna()
                    .astype(str)
                    .drop_duplicates()
                    .tolist()
                )
                target_genes = [gene_id for gene_id in target_genes if gene_id in df_rna.index]
                n_targets = len(target_genes)
                if n_targets < self.config.min_box_points:
                    print(
                        f"  Skipped pooled random control: n_targets={n_targets} "
                        f"< min_box_points={self.config.min_box_points}"
                    )
                    continue

                target_gene_set = set(target_genes)
                background_genes = [gene_id for gene_id in df_rna.index.astype(str) if gene_id not in target_gene_set]
                if len(background_genes) < n_targets:
                    print(
                        f"  Skipped pooled random control: background genes={len(background_genes)} "
                        f"< n_targets={n_targets}"
                    )
                    continue

                expected_sign = self._expected_effect_sign(direction_short, target_regulation)
                background_high = (
                    df_rna.loc[background_genes, high_samples]
                    .apply(pd.to_numeric, errors="coerce")
                    .mean(axis=1)
                )
                background_low = (
                    df_rna.loc[background_genes, low_samples]
                    .apply(pd.to_numeric, errors="coerce")
                    .mean(axis=1)
                )
                background_directional = ((background_high - background_low) * expected_sign).dropna()
                if len(background_directional) < n_targets:
                    print(
                        f"  Skipped pooled random control: valid background directional values="
                        f"{len(background_directional)} < n_targets={n_targets}"
                    )
                    continue

                background_values = background_directional.to_numpy(dtype=float)
                n_background = len(background_values)

                n_sites_used += 1
                n_target_gene_values += n_targets
                print(
                    f"  Running {n_random_sets} pooled random iterations "
                    f"({n_targets} genes/site, {n_background} background genes)"
                )

                random_gene_rows_kept_by_cancer.setdefault(cancer_type, 0)
                for iteration in range(n_random_sets):
                    sampled_idx = rng.choice(n_background, size=n_targets, replace=False)
                    sampled_values = background_values[sampled_idx]
                    random_iteration_sum[iteration] += float(np.sum(sampled_values))
                    random_iteration_count[iteration] += float(len(sampled_values))

                    kept_for_cancer = random_gene_rows_kept_by_cancer.get(cancer_type, 0)
                    if kept_for_cancer < max_random_gene_rows_per_group:
                        remaining = max_random_gene_rows_per_group - kept_for_cancer
                        values_to_keep = sampled_values[:remaining]
                        for value in values_to_keep:
                            random_gene_rows.append(
                                {
                                    "source": "Matched random genes",
                                    "direction_short": direction_short,
                                    "target_regulation": target_regulation,
                                    "cancer_type": cancer_type,
                                    "site": site,
                                    "site_label": site_label,
                                    "tf_name": tf_name,
                                    "target_gene_id": np.nan,
                                    "target_gene_name": np.nan,
                                    "directional_logFC": float(value),
                                    "target_logFC": np.nan,
                                    "iteration": iteration,
                                }
                            )
                        random_gene_rows_kept_by_cancer[cancer_type] = kept_for_cancer + len(values_to_keep)

            valid_iterations = random_iteration_count > 0
            if not np.any(valid_iterations) or len(observed_values_for_group) == 0:
                print(
                    f"  Skipped pooled random control summary for {direction_label} | {target_regulation}: "
                    "no valid random iterations or observed values"
                )
                continue

            random_global = random_iteration_sum[valid_iterations] / random_iteration_count[valid_iterations]
            observed_array = np.asarray(observed_values_for_group, dtype=float)
            observed_global_mean = float(np.nanmean(observed_array))
            observed_global_median = float(np.nanmedian(observed_array))
            random_mean = float(np.nanmean(random_global))
            random_median = float(np.nanmedian(random_global))
            empirical_p = float((np.sum(random_global >= observed_global_mean) + 1) / (len(random_global) + 1))

            valid_counts = random_iteration_count[valid_iterations]
            for iteration_index, value in enumerate(random_global):
                random_global_rows.append(
                    {
                        "direction_short": direction_short,
                        "target_regulation": target_regulation,
                        "iteration": iteration_index,
                        "random_global_mean_directional_logFC": float(value),
                        "n_random_gene_values": int(valid_counts[iteration_index]),
                    }
                )

            summary_rows.append(
                {
                    "direction_short": direction_short,
                    "target_regulation": target_regulation,
                    "n_sites_used": int(n_sites_used),
                    "n_observed_gene_values": int(len(observed_array)),
                    "n_target_gene_values_used_for_random": int(n_target_gene_values),
                    "n_random_iterations": int(len(random_global)),
                    "observed_global_mean_directional_logFC": observed_global_mean,
                    "observed_global_median_directional_logFC": observed_global_median,
                    "random_global_mean_directional_logFC": random_mean,
                    "random_global_median_directional_logFC": random_median,
                    "random_global_sd_directional_logFC": float(np.nanstd(random_global, ddof=1)) if len(random_global) > 1 else np.nan,
                    "empirical_p_observed_mean_gt_random": empirical_p,
                }
            )
            print(
                f"  Finished pooled random control {direction_label} | {target_regulation}: "
                f"sites={n_sites_used}, observed_mean={observed_global_mean:.4f}, "
                f"empirical_p={empirical_p:.4f}"
            )

        print(
            f"Pooled matched random control complete: {len(summary_rows)} regulation groups, "
            f"{n_random_sets} iterations per group"
        )
        df_gene = pd.DataFrame(observed_rows + random_gene_rows)
        df_global = pd.DataFrame(random_global_rows)
        df_summary = pd.DataFrame(summary_rows)
        if not df_summary.empty:
            df_summary["empirical_q_bh"] = self._bh_adjust(df_summary["empirical_p_observed_mean_gt_random"])
            df_summary["significance_bh"] = df_summary["empirical_q_bh"].map(self._p_to_stars)
            df_summary["significance_raw"] = df_summary["empirical_p_observed_mean_gt_random"].map(self._p_to_stars)
            if self.config.use_bh_pvalue_correction:
                df_summary["significance"] = df_summary["significance_bh"]
            else:
                df_summary["significance"] = df_summary["significance_raw"]
        return df_gene, df_global, df_summary

    def _compute_kde_curve(
        self,
        values: pd.Series,
        x_grid: np.ndarray,
    ) -> Optional[np.ndarray]:
        values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) < self.config.min_box_points:
            return None
        values_array = values.to_numpy(dtype=float)
        if len(values_array) < 3 or float(np.nanstd(values_array)) <= 0:
            return None
        try:
            from scipy.stats import gaussian_kde
            density = gaussian_kde(values_array)(x_grid)
        except Exception:
            return None
        if not np.all(np.isfinite(density)) or float(np.nanmax(density)) <= 0:
            return None
        return density

    def _bootstrap_mean_difference_ci(
        self,
        observed: np.ndarray,
        random_values: np.ndarray,
        n_bootstrap: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Tuple[float, float, float, float]:
        observed_array = np.asarray(observed, dtype=float)
        random_array = np.asarray(random_values, dtype=float)
        observed_array = observed_array[np.isfinite(observed_array)]
        random_array = random_array[np.isfinite(random_array)]

        if len(observed_array) == 0 or len(random_array) == 0:
            return np.nan, np.nan, np.nan, np.nan

        effect = float(observed_array.mean() - random_array.mean())
        n_boot = int(n_bootstrap if n_bootstrap is not None else self.config.bootstrap_iterations)
        rng = np.random.default_rng(seed if seed is not None else self.config.random_seed)

        n_obs = len(observed_array)
        n_rand = len(random_array)
        boot_diffs = np.empty(n_boot, dtype=float)
        for boot_idx in range(n_boot):
            obs_sample = observed_array[rng.integers(0, n_obs, size=n_obs)]
            rand_sample = random_array[rng.integers(0, n_rand, size=n_rand)]
            boot_diffs[boot_idx] = float(obs_sample.mean() - rand_sample.mean())

        ci_lower = float(np.percentile(boot_diffs, 2.5))
        ci_upper = float(np.percentile(boot_diffs, 97.5))
        se = float(np.std(boot_diffs, ddof=1)) if n_boot > 1 else np.nan
        return effect, ci_lower, ci_upper, se

    @staticmethod
    def _fixed_effect_meta_analysis(
        effects: Sequence[float],
        ses: Sequence[float],
    ) -> Tuple[float, float, float, float]:
        effects_array = np.asarray(effects, dtype=float)
        ses_array = np.asarray(ses, dtype=float)
        valid = np.isfinite(effects_array) & np.isfinite(ses_array) & (ses_array > 0)
        if not np.any(valid):
            return np.nan, np.nan, np.nan, np.nan

        effects_valid = effects_array[valid]
        ses_valid = ses_array[valid]
        weights = 1.0 / (ses_valid ** 2)
        pooled_effect = float(np.sum(weights * effects_valid) / np.sum(weights))
        pooled_se = float(np.sqrt(1.0 / np.sum(weights)))
        ci_lower = pooled_effect - 1.96 * pooled_se
        ci_upper = pooled_effect + 1.96 * pooled_se
        return pooled_effect, ci_lower, ci_upper, pooled_se

    @staticmethod
    def _format_fdr_value(q_value: float) -> str:
        if pd.isna(q_value):
            return "NA"
        if q_value < 1e-4:
            return f"{q_value:.1e}"
        return f"{q_value:.4f}"

    def _plot_p_value_header(self) -> str:
        return "FDR" if self.config.use_bh_pvalue_correction else "P"

    def _plot_p_value_from_mapping(self, row: object) -> float:
        if isinstance(row, pd.Series):
            if self.config.use_bh_pvalue_correction:
                return pd.to_numeric(row.get("mannwhitney_q_bh"), errors="coerce")
            return pd.to_numeric(row.get("mannwhitney_p_observed_gt_random"), errors="coerce")
        if self.config.use_bh_pvalue_correction:
            return pd.to_numeric(getattr(row, "mannwhitney_q_bh", np.nan), errors="coerce")
        return pd.to_numeric(getattr(row, "mannwhitney_p_observed_gt_random", np.nan), errors="coerce")

    def _summarize_gene_level_observed_vs_random_by_cancer(self, df: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        if df.empty:
            return pd.DataFrame()

        group_cols = ["direction_short", "target_regulation", "cancer_type"]
        for keys, sub in df.groupby(group_cols, dropna=False):
            direction_short, target_regulation, cancer_type = keys
            observed = pd.to_numeric(
                sub.loc[sub["source"].eq("Observed targets"), "directional_logFC"],
                errors="coerce",
            ).replace([np.inf, -np.inf], np.nan).dropna()
            random_values = pd.to_numeric(
                sub.loc[sub["source"].eq("Matched random genes"), "directional_logFC"],
                errors="coerce",
            ).replace([np.inf, -np.inf], np.nan).dropna()

            p_value = np.nan
            effect_size = np.nan
            effect_ci_lower = np.nan
            effect_ci_upper = np.nan
            effect_se = np.nan
            if len(observed) >= self.config.min_box_points and len(random_values) >= self.config.min_box_points:
                try:
                    p_value = float(mannwhitneyu(observed, random_values, alternative="greater").pvalue)
                except ValueError:
                    p_value = np.nan
                effect_size, effect_ci_lower, effect_ci_upper, effect_se = self._bootstrap_mean_difference_ci(
                    observed.to_numpy(dtype=float),
                    random_values.to_numpy(dtype=float),
                )

            rows.append(
                {
                    "direction_short": direction_short,
                    "target_regulation": target_regulation,
                    "cancer_type": cancer_type,
                    "n_observed_gene_values": int(len(observed)),
                    "n_random_gene_values": int(len(random_values)),
                    "observed_median_directional_logFC": float(observed.median()) if len(observed) else np.nan,
                    "random_median_directional_logFC": float(random_values.median()) if len(random_values) else np.nan,
                    "delta_median_observed_minus_random": (
                        float(observed.median() - random_values.median())
                        if len(observed) and len(random_values)
                        else np.nan
                    ),
                    "observed_mean_directional_logFC": float(observed.mean()) if len(observed) else np.nan,
                    "random_mean_directional_logFC": float(random_values.mean()) if len(random_values) else np.nan,
                    "effect_size_mean_diff": effect_size,
                    "effect_size_ci_lower": effect_ci_lower,
                    "effect_size_ci_upper": effect_ci_upper,
                    "effect_size_se": effect_se,
                    "mannwhitney_p_observed_gt_random": p_value,
                    "mannwhitney_p_raw": p_value,
                    "significance_raw": self._p_to_stars(p_value),
                }
            )

        out = pd.DataFrame(rows)
        if not out.empty:
            out["mannwhitney_q_bh"] = np.nan
            for _, idx in out.groupby(["direction_short", "target_regulation"], dropna=False).groups.items():
                out.loc[idx, "mannwhitney_q_bh"] = self._bh_adjust(out.loc[idx, "mannwhitney_p_observed_gt_random"])
            out["significance_bh"] = out["mannwhitney_q_bh"].map(self._p_to_stars)
            if self.config.use_bh_pvalue_correction:
                out["significance"] = out["significance_bh"]
            else:
                out["significance"] = out["significance_raw"]
        return out

    def plot_figureA_gene_level_logfc_observed_vs_random(self, df_gene: pd.DataFrame, output_dir: Path) -> None:
        if df_gene.empty:
            return

        plot_dir = output_dir / "figureA_gene_level_observed_vs_random_logfc_by_cancer"
        plot_dir.mkdir(parents=True, exist_ok=True)

        df = df_gene.copy()
        df["directional_logFC"] = pd.to_numeric(df["directional_logFC"], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["directional_logFC"])
        if df.empty:
            return

        stats = self._summarize_gene_level_observed_vs_random_by_cancer(df)
        if not stats.empty:
            stats.to_csv(plot_dir / "FigureA_gene_level_observed_vs_random_by_cancer_stats.csv", index=False)

        color_observed = "#C97B49"
        color_random = "#8FAFBC"
        edge_random = "#5E8FA1"
        zero_color = "#4A4A4A"
        ridge_height = 0.78
        min_points = self.config.min_box_points

        for (direction_short, target_regulation), sub in df.groupby(["direction_short", "target_regulation"], dropna=False):
            sub = sub.copy()
            available_cancers = set(sub["cancer_type"].dropna().astype(str))
            ordered_cancers = [c for c in self.config.cancer_types if c in available_cancers]
            ordered_cancers.extend(sorted(available_cancers - set(ordered_cancers)))

            plot_cancers: List[str] = []
            cancer_values: Dict[str, Dict[str, pd.Series]] = {}
            all_values_for_xlim: List[float] = []

            for cancer_type in ordered_cancers:
                cancer_sub = sub[sub["cancer_type"].astype(str).eq(cancer_type)].copy()
                observed = pd.to_numeric(
                    cancer_sub.loc[cancer_sub["source"].eq("Observed targets"), "directional_logFC"],
                    errors="coerce",
                ).replace([np.inf, -np.inf], np.nan).dropna()
                random_values = pd.to_numeric(
                    cancer_sub.loc[cancer_sub["source"].eq("Matched random genes"), "directional_logFC"],
                    errors="coerce",
                ).replace([np.inf, -np.inf], np.nan).dropna()

                if len(observed) < min_points or len(random_values) < min_points:
                    continue

                plot_cancers.append(cancer_type)
                cancer_values[cancer_type] = {
                    "observed": observed,
                    "random": random_values,
                }
                all_values_for_xlim.extend(observed.tolist())
                all_values_for_xlim.extend(random_values.tolist())

            if len(plot_cancers) == 0:
                continue

            stat_sub = stats[
                stats["direction_short"].astype(str).eq(str(direction_short))
                & stats["target_regulation"].astype(str).eq(str(target_regulation))
            ].copy() if not stats.empty else pd.DataFrame()
            stat_lookup: Dict[str, str] = {}
            if not stat_sub.empty:
                for _, stat_row in stat_sub.iterrows():
                    cancer_key = str(stat_row.get("cancer_type", ""))
                    stat_lookup[cancer_key] = self._plot_significance_from_row(stat_row)

            all_array = np.asarray(all_values_for_xlim, dtype=float)
            x_min = float(np.nanquantile(all_array, 0.005))
            x_max = float(np.nanquantile(all_array, 0.995))
            if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min >= x_max:
                x_min = float(np.nanmin(all_array)) - 0.5
                x_max = float(np.nanmax(all_array)) + 0.5
            x_min = min(x_min, -0.05)
            x_max = max(x_max, 0.05)
            x_padding = 0.05 * (x_max - x_min)
            x_grid = np.linspace(x_min - x_padding, x_max + x_padding, 500)

            fig_height = max(4.6, 0.62 * len(plot_cancers) + 1.8)
            fig_width = 7.0
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            row_positions: List[float] = []
            row_labels: List[str] = []
            for row_idx, cancer_type in enumerate(plot_cancers):
                y_base = float(len(plot_cancers) - row_idx)
                row_positions.append(y_base)
                row_labels.append(cancer_type)

                observed = cancer_values[cancer_type]["observed"]
                random_values = cancer_values[cancer_type]["random"]
                random_density = self._compute_kde_curve(random_values, x_grid)
                observed_density = self._compute_kde_curve(observed, x_grid)

                max_density = 0.0
                if random_density is not None:
                    max_density = max(max_density, float(np.nanmax(random_density)))
                if observed_density is not None:
                    max_density = max(max_density, float(np.nanmax(observed_density)))
                if max_density <= 0:
                    continue

                if random_density is not None:
                    y_random = y_base + random_density / max_density * ridge_height
                    ax.fill_between(
                        x_grid,
                        y_base,
                        y_random,
                        facecolor=color_random,
                        edgecolor=edge_random,
                        linewidth=0.7,
                        alpha=0.36,
                        zorder=1,
                    )
                    ax.plot(x_grid, y_random, color=edge_random, linewidth=0.75, alpha=0.85, zorder=2)

                if observed_density is not None:
                    y_observed = y_base + observed_density / max_density * ridge_height
                    ax.fill_between(
                        x_grid,
                        y_base,
                        y_observed,
                        facecolor=color_observed,
                        edgecolor="none",
                        alpha=0.20,
                        zorder=3,
                    )
                    ax.plot(x_grid, y_observed, color=color_observed, linewidth=1.25, alpha=0.96, zorder=4)

                observed_median = float(observed.median())
                random_median = float(random_values.median())
                ax.vlines(
                    random_median,
                    y_base,
                    y_base + ridge_height * 0.92,
                    color=edge_random,
                    linewidth=0.75,
                    linestyle="--",
                    alpha=0.92,
                    zorder=5,
                )
                ax.vlines(
                    observed_median,
                    y_base,
                    y_base + ridge_height * 0.92,
                    color=color_observed,
                    linewidth=1.0,
                    alpha=0.95,
                    zorder=6,
                )
                sig_label = stat_lookup.get(cancer_type, "ns")
                label_y = y_base + ridge_height * 0.36
                ax.text(
                    x_grid[-1],
                    label_y,
                    f"n = {len(observed)}",
                    ha="right",
                    va="top",
                    fontsize=8.5,
                    color="#2F2F2F",
                )
                if sig_label != "ns":
                    ax.text(
                        x_grid[-1],
                        label_y + 0.045,
                        sig_label,
                        ha="right",
                        va="bottom",
                        fontsize=8.5,
                        color="#2F2F2F",
                    )

            ax.axvline(0, color=zero_color, linestyle=":", linewidth=0.9, alpha=0.86, zorder=0)
            ax.set_yticks(row_positions)
            ax.set_yticklabels(row_labels, fontsize=9.5)
            ax.set_ylim(0.55, len(plot_cancers) + ridge_height + 0.35)
            ax.set_xlim(float(x_grid[0]), float(x_grid[-1]))
            ax.set_xlabel("Directional gene log2FC", fontsize=11)
            ax.set_ylabel("Cancer type", fontsize=11)
            ax.tick_params(axis="x", labelsize=9.5)
            ax.tick_params(axis="y", length=0)
            ax.grid(axis="x", linestyle=":", linewidth=0.52, alpha=0.25)
            ax.grid(axis="y", visible=False)

            from matplotlib.patches import Patch
            from matplotlib.lines import Line2D
            handles = [
                Patch(facecolor=color_random, edgecolor=edge_random, alpha=0.36, label="Matched random genes"),
                Line2D([0], [0], color=color_observed, linewidth=1.5, label="Observed targets"),
                Line2D([0], [0], color=zero_color, linewidth=0.9, linestyle=":", label="Zero log2FC"),
            ]
            ax.legend(
                handles=handles,
                frameon=False,
                fontsize=9,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0.0,
                handlelength=1.6,
                handletextpad=0.55,
            )
            sns.despine(ax=ax, left=True)
            plt.tight_layout(rect=[0, 0, 0.84, 1])

            prefix = self._sanitize_filename(
                f"FigureA_{direction_short}_{target_regulation}_gene_level_observed_vs_random_directional_log2fc_by_cancer_ridge"
            )
            for ext in ["png", "pdf", "svg"]:
                _savefig_with_numeric_ticks(
                    fig,
                    plot_dir / f"{prefix}.{ext}",
                    ax,
                    dpi=self.config.dpi,
                    numeric_x=True,
                    numeric_y=False,
                )
            plt.close(fig)

    def plot_figureA_gene_level_forest(self, df_gene: pd.DataFrame, output_dir: Path) -> None:
        if df_gene.empty:
            return

        plot_dir = output_dir / "figureA_gene_level_observed_vs_random_logfc_by_cancer"
        plot_dir.mkdir(parents=True, exist_ok=True)

        df = df_gene.copy()
        df["directional_logFC"] = pd.to_numeric(df["directional_logFC"], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["directional_logFC"])
        if df.empty:
            return

        stats = self._summarize_gene_level_observed_vs_random_by_cancer(df)
        if stats.empty:
            return

        stats.to_csv(plot_dir / "FigureA_gene_level_observed_vs_random_by_cancer_stats.csv", index=False)

        color_main = "#C97B49"
        color_summary = "#8B4513"
        zero_color = "#4A4A4A"
        min_points = self.config.min_box_points
        meta_rows: List[Dict[str, object]] = []

        for (direction_short, target_regulation), stat_sub in stats.groupby(
            ["direction_short", "target_regulation"], dropna=False
        ):
            stat_sub = stat_sub.copy()
            available_cancers = set(stat_sub["cancer_type"].dropna().astype(str))
            ordered_cancers = [c for c in self.config.cancer_types if c in available_cancers]
            ordered_cancers.extend(sorted(available_cancers - set(ordered_cancers)))

            plot_rows: List[Dict[str, object]] = []
            for cancer_type in ordered_cancers:
                cancer_stats = stat_sub[stat_sub["cancer_type"].astype(str).eq(cancer_type)]
                if cancer_stats.empty:
                    continue
                row = cancer_stats.iloc[0].to_dict()
                if int(row.get("n_observed_gene_values", 0)) < min_points:
                    continue
                if int(row.get("n_random_gene_values", 0)) < min_points:
                    continue
                if not np.isfinite(row.get("effect_size_mean_diff", np.nan)):
                    continue
                row["row_label"] = str(cancer_type)
                row["is_summary"] = False
                plot_rows.append(row)

            if len(plot_rows) == 0:
                continue

            effects = [float(r["effect_size_mean_diff"]) for r in plot_rows]
            ses = [float(r["effect_size_se"]) for r in plot_rows]
            pooled_effect, pooled_ci_lower, pooled_ci_upper, pooled_se = self._fixed_effect_meta_analysis(effects, ses)
            meta_rows.append(
                {
                    "direction_short": direction_short,
                    "target_regulation": target_regulation,
                    "n_cancers": int(len(plot_rows)),
                    "pooled_effect_size_mean_diff": pooled_effect,
                    "pooled_effect_size_ci_lower": pooled_ci_lower,
                    "pooled_effect_size_ci_upper": pooled_ci_upper,
                    "pooled_effect_size_se": pooled_se,
                }
            )
            plot_rows.append(
                {
                    "row_label": "Summary",
                    "is_summary": True,
                    "effect_size_mean_diff": pooled_effect,
                    "effect_size_ci_lower": pooled_ci_lower,
                    "effect_size_ci_upper": pooled_ci_upper,
                    "effect_size_se": pooled_se,
                    "mannwhitney_q_bh": np.nan,
                    "n_observed_gene_values": np.nan,
                }
            )

            df_plot = pd.DataFrame(plot_rows)
            y_positions = np.arange(len(df_plot))[::-1]

            fig_height = max(5.2, 0.58 * len(df_plot) + 2.0)
            fig = plt.figure(figsize=(9.2, fig_height))
            gs = fig.add_gridspec(
                nrows=1,
                ncols=2,
                width_ratios=[0.66, 0.34],
                wspace=0.03,
            )
            ax = fig.add_subplot(gs[0, 0])
            ax_table = fig.add_subplot(gs[0, 1], sharey=ax)

            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")
            ax_table.set_facecolor("white")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax_table.axis("off")

            x_values: List[float] = []
            for y_pos, row in zip(y_positions, df_plot.itertuples(index=False)):
                effect = float(getattr(row, "effect_size_mean_diff"))
                ci_low = float(getattr(row, "effect_size_ci_lower"))
                ci_high = float(getattr(row, "effect_size_ci_upper"))
                is_summary = bool(getattr(row, "is_summary"))
                x_values.extend([effect, ci_low, ci_high])

                if is_summary:
                    color = color_summary
                    marker = "D"
                    markersize = 7.4
                    linewidth = 2.0
                else:
                    color = color_main
                    marker = "o"
                    markersize = 5.8
                    linewidth = 1.45

                if np.isfinite(effect) and np.isfinite(ci_low) and np.isfinite(ci_high):
                    ax.hlines(y_pos, ci_low, ci_high, color=color, linewidth=linewidth, zorder=2)
                    ax.plot(effect, y_pos, marker=marker, markersize=markersize, color=color, zorder=3)
                    ax.plot([ci_low, ci_low], [y_pos - 0.10, y_pos + 0.10], color=color, linewidth=1.0, zorder=2)
                    ax.plot([ci_high, ci_high], [y_pos - 0.10, y_pos + 0.10], color=color, linewidth=1.0, zorder=2)

            ax.axvline(0, color=zero_color, linestyle=":", linewidth=1.0, alpha=0.9, zorder=0)
            ax.set_yticks(y_positions)
            ax.set_yticklabels(df_plot["row_label"].tolist(), fontsize=10.5)
            ax.set_xlabel("Effect size (observed minus random mean directional log2FC)", fontsize=11)
            ax.set_ylabel("Cancer type", fontsize=11)
            ax.tick_params(axis="x", labelsize=9.8, length=0, pad=10, color="#333333")
            ax.tick_params(axis="y", length=0)
            ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.28)
            ax.grid(axis="y", visible=False)

            if x_values:
                _set_forest_effect_x_axis(ax, x_values)

            ax.set_ylim(-0.70, len(df_plot) - 0.20)
            ax_table.set_ylim(ax.get_ylim())
            ax_table.set_xlim(0, 1)

            summary_rows = df_plot.index[df_plot["is_summary"].astype(bool)].tolist()
            if summary_rows:
                summary_y = y_positions[summary_rows[0]]
                ax.axhline(summary_y + 0.50, color="#B8B8B8", linewidth=0.75, alpha=0.75)
                ax_table.axhline(summary_y + 0.50, color="#B8B8B8", linewidth=0.75, alpha=0.75)

            header_y = len(df_plot) - 0.02
            col_effect = 0.02
            col_ci = 0.34
            col_fdr = 0.82
            p_value_header = self._plot_p_value_header()
            header_fontsize = 9.0
            table_fontsize = 8.8

            ax_table.text(
                col_effect,
                header_y,
                "Effect",
                ha="left",
                va="bottom",
                fontsize=header_fontsize,
                fontweight="bold",
            )
            ax_table.text(
                col_ci,
                header_y,
                "95% CI",
                ha="left",
                va="bottom",
                fontsize=header_fontsize,
                fontweight="bold",
            )
            ax_table.text(
                col_fdr,
                header_y,
                p_value_header,
                ha="left",
                va="bottom",
                fontsize=header_fontsize,
                fontweight="bold",
            )

            for y_pos, row in zip(y_positions, df_plot.itertuples(index=False)):
                effect = float(getattr(row, "effect_size_mean_diff"))
                ci_low = float(getattr(row, "effect_size_ci_lower"))
                ci_high = float(getattr(row, "effect_size_ci_upper"))
                p_value = self._plot_p_value_from_mapping(row)
                is_summary = bool(getattr(row, "is_summary"))
                text_color = color_summary if is_summary else "#2F2F2F"
                fontweight = "bold" if is_summary else "normal"

                effect_label = f"{effect:.3f}" if np.isfinite(effect) else "NA"
                ci_label = f"[{ci_low:.3f}, {ci_high:.3f}]" if np.isfinite(ci_low) and np.isfinite(ci_high) else "NA"
                fdr_label = "Pooled" if is_summary else self._format_fdr_value(p_value)

                ax_table.text(
                    col_effect,
                    y_pos,
                    effect_label,
                    ha="left",
                    va="center",
                    fontsize=table_fontsize,
                    color=text_color,
                    fontweight=fontweight,
                )
                ax_table.text(
                    col_ci,
                    y_pos,
                    ci_label,
                    ha="left",
                    va="center",
                    fontsize=table_fontsize,
                    color=text_color,
                    fontweight=fontweight,
                )
                ax_table.text(
                    col_fdr,
                    y_pos,
                    fdr_label,
                    ha="left",
                    va="center",
                    fontsize=table_fontsize,
                    color=text_color,
                    fontweight=fontweight,
                )

            from matplotlib.lines import Line2D

            handles = [
                Line2D([0], [0], marker="o", linestyle="", markersize=5.8, color=color_main, label="Per cancer"),
                Line2D([0], [0], marker="D", linestyle="", markersize=6.8, color=color_summary, label="Summary"),
                Line2D([0], [0], color=zero_color, linewidth=1.0, linestyle=":", label="Zero effect"),
            ]
            ax.legend(
                handles=handles,
                frameon=False,
                fontsize=8.8,
                loc="upper center",
                bbox_to_anchor=(0.50, -0.12),
                ncol=3,
                handlelength=1.2,
                handletextpad=0.45,
                columnspacing=1.1,
            )

            fig.subplots_adjust(
                left=0.14,
                right=0.98,
                bottom=0.15,
                top=0.95,
                wspace=0.03,
            )

            prefix = self._sanitize_filename(
                f"FigureA_{direction_short}_{target_regulation}_gene_level_observed_vs_random_directional_log2fc_by_cancer_forest"
            )
            for ext in ["png", "pdf", "svg"]:
                _savefig_with_numeric_ticks(
                    fig,
                    plot_dir / f"{prefix}.{ext}",
                    ax,
                    dpi=self.config.dpi,
                    numeric_x=True,
                    numeric_y=False,
                )
            plt.close(fig)

            self._plot_figureA_gene_level_forest_pointlabels(
                df_plot=df_plot,
                y_positions=y_positions,
                direction_short=str(direction_short),
                target_regulation=str(target_regulation),
                plot_dir=plot_dir,
                color_main=color_main,
                color_summary=color_summary,
                zero_color=zero_color,
            )

        if meta_rows:
            pd.DataFrame(meta_rows).to_csv(
                plot_dir / "FigureA_gene_level_observed_vs_random_meta_summary.csv",
                index=False,
            )

    def _plot_figureA_gene_level_forest_pointlabels(
        self,
        df_plot: pd.DataFrame,
        y_positions: np.ndarray,
        direction_short: str,
        target_regulation: str,
        plot_dir: Path,
        color_main: str,
        color_summary: str,
        zero_color: str,
    ) -> None:
        fig_height = max(5.0, 0.56 * len(df_plot) + 1.9)
        fig, ax = plt.subplots(figsize=(8.8, fig_height))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        x_values: List[float] = []
        label_specs: List[Dict[str, object]] = []

        for y_pos, row in zip(y_positions, df_plot.itertuples(index=False)):
            effect = float(getattr(row, "effect_size_mean_diff"))
            ci_low = float(getattr(row, "effect_size_ci_lower"))
            ci_high = float(getattr(row, "effect_size_ci_upper"))
            p_value = self._plot_p_value_from_mapping(row)
            is_summary = bool(getattr(row, "is_summary"))
            x_values.extend([effect, ci_low, ci_high])

            if is_summary:
                color = color_summary
                marker = "D"
                markersize = 7.2
                linewidth = 2.0
                text_color = color_summary
                fontweight = "bold"
            else:
                color = color_main
                marker = "o"
                markersize = 5.8
                linewidth = 1.45
                text_color = "#2F2F2F"
                fontweight = "normal"

            if np.isfinite(effect) and np.isfinite(ci_low) and np.isfinite(ci_high):
                ax.hlines(y_pos, ci_low, ci_high, color=color, linewidth=linewidth, zorder=2)
                ax.plot(effect, y_pos, marker=marker, markersize=markersize, color=color, zorder=3)
                ax.plot([ci_low, ci_low], [y_pos - 0.10, y_pos + 0.10], color=color, linewidth=1.0, zorder=2)
                ax.plot([ci_high, ci_high], [y_pos - 0.10, y_pos + 0.10], color=color, linewidth=1.0, zorder=2)

                if is_summary:
                    label = f"{effect:.3f} [{ci_low:.3f}, {ci_high:.3f}] (Pooled)"
                else:
                    label = (
                        f"{effect:.3f} [{ci_low:.3f}, {ci_high:.3f}], "
                        f"{self._plot_p_value_header()}={self._format_fdr_value(p_value)}"
                    )
                label_specs.append(
                    {
                        "x_anchor": float(max(ci_high, effect)),
                        "y": float(y_pos),
                        "label": label,
                        "color": text_color,
                        "fontweight": fontweight,
                    }
                )

        ax.axvline(0, color=zero_color, linestyle=":", linewidth=1.0, alpha=0.9, zorder=0)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(df_plot["row_label"].tolist(), fontsize=10.5)
        ax.set_xlabel("Effect size (observed minus random mean directional log2FC)", fontsize=11)
        ax.set_ylabel("Cancer type", fontsize=11)
        ax.tick_params(axis="x", labelsize=9.8)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.28)
        ax.grid(axis="y", visible=False)

        if x_values:
            _set_forest_effect_x_axis(ax, x_values, left_padding_frac=0.10, right_padding_frac=0.42)
            x_min, x_max = ax.get_xlim()
            x_span = x_max - x_min if x_max > x_min else 1.0
            label_offset = 0.012 * x_span
            for spec in label_specs:
                ax.text(
                    float(spec["x_anchor"]) + label_offset,
                    float(spec["y"]),
                    str(spec["label"]),
                    ha="left",
                    va="center",
                    fontsize=8.6,
                    color=str(spec["color"]),
                    fontweight=str(spec["fontweight"]),
                    clip_on=False,
                )

        ax.set_ylim(-0.70, len(df_plot) - 0.20)

        summary_rows = df_plot.index[df_plot["is_summary"].astype(bool)].tolist()
        if summary_rows:
            summary_y = y_positions[summary_rows[0]]
            ax.axhline(summary_y + 0.50, color="#B8B8B8", linewidth=0.75, alpha=0.75)

        from matplotlib.lines import Line2D

        handles = [
            Line2D([0], [0], marker="o", linestyle="", markersize=5.8, color=color_main, label="Per cancer"),
            Line2D([0], [0], marker="D", linestyle="", markersize=6.8, color=color_summary, label="Summary"),
            Line2D([0], [0], color=zero_color, linewidth=1.0, linestyle=":", label="Zero effect"),
        ]
        ax.legend(
            handles=handles,
            frameon=False,
            fontsize=8.8,
            loc="upper center",
            bbox_to_anchor=(0.50, -0.10),
            ncol=3,
            handlelength=1.2,
            handletextpad=0.45,
            columnspacing=1.1,
        )
        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.14, top=0.95)

        prefix = self._sanitize_filename(
            f"FigureA_{direction_short}_{target_regulation}_gene_level_observed_vs_random_directional_log2fc_by_cancer_forest_pointlabels"
        )
        for ext in ["png", "pdf", "svg"]:
            _savefig_with_numeric_ticks(
                fig,
                plot_dir / f"{prefix}.{ext}",
                ax,
                dpi=self.config.dpi,
                numeric_x=True,
                numeric_y=False,
            )
        plt.close(fig)

    def plot_figureC_global_random_mean_null(self, df_global: pd.DataFrame, df_summary: pd.DataFrame, output_dir: Path) -> None:
        if df_global.empty or df_summary.empty:
            return

        plot_dir = output_dir / "figureC_global_random_mean_null"
        plot_dir.mkdir(parents=True, exist_ok=True)

        df_global = df_global.copy()
        df_summary = df_summary.copy()
        df_global["random_global_mean_directional_logFC"] = pd.to_numeric(df_global["random_global_mean_directional_logFC"], errors="coerce")
        df_summary["observed_global_mean_directional_logFC"] = pd.to_numeric(df_summary["observed_global_mean_directional_logFC"], errors="coerce")

        for (direction_short, target_regulation), random_sub in df_global.groupby(["direction_short", "target_regulation"], dropna=False):
            summary_sub = df_summary[
                df_summary["direction_short"].astype(str).eq(str(direction_short))
                & df_summary["target_regulation"].astype(str).eq(str(target_regulation))
            ].copy()
            if summary_sub.empty:
                continue
            random_values = pd.to_numeric(random_sub["random_global_mean_directional_logFC"], errors="coerce").dropna()
            observed_value = pd.to_numeric(summary_sub["observed_global_mean_directional_logFC"], errors="coerce").dropna()
            if len(random_values) < self.config.min_box_points or len(observed_value) == 0:
                continue
            observed_value = float(observed_value.iloc[0])
            random_median = float(random_values.median())
            p_value = summary_sub["empirical_p_observed_mean_gt_random"].iloc[0] if "empirical_p_observed_mean_gt_random" in summary_sub.columns else np.nan
            q_value = summary_sub["empirical_q_bh"].iloc[0] if "empirical_q_bh" in summary_sub.columns else np.nan
            n_obs = int(summary_sub["n_observed_gene_values"].iloc[0]) if "n_observed_gene_values" in summary_sub.columns and not pd.isna(summary_sub["n_observed_gene_values"].iloc[0]) else len(random_values)

            fig, ax = plt.subplots(figsize=(5.3, 4.3))
            if len(random_values) >= 8 and float(random_values.std(ddof=0)) > 0:
                sns.kdeplot(random_values, ax=ax, fill=True, linewidth=1.2, color="#8FAFBC", alpha=0.34, label="Random global means")
            ax.hist(random_values, bins=min(40, max(10, int(np.sqrt(len(random_values))))), density=True, alpha=0.20, color="#8FAFBC", edgecolor="#FFFFFF", linewidth=0.35)
            ax.axvline(random_median, color="#4A4A4A", linestyle=":", linewidth=1.1, label=f"Random median = {random_median:.3f}")
            ax.axvline(observed_value, color="#C97B49", linewidth=1.5, label=f"Observed mean = {observed_value:.3f}")
            ax.set_xlabel("Global mean directional gene logFC", fontsize=15.5)
            ax.set_ylabel("Density", fontsize=15.5)
            ax.tick_params(axis="both", labelsize=14.5)
            ax.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.30)
            ax.grid(axis="x", visible=False)
            annotation = f"n = {n_obs}\nempirical p = {self._format_p_value_for_plot(p_value)}\nBH q = {self._format_p_value_for_plot(q_value)}"
            ax.text(0.98, 0.94, annotation, transform=ax.transAxes, ha="right", va="top", fontsize=14)
            ax.legend(frameon=False, fontsize=14, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
            sns.despine(ax=ax)
            plt.tight_layout(rect=[0, 0, 0.82, 1])
            prefix = self._sanitize_filename(f"FigureC_{direction_short}_{target_regulation}_global_random_mean_directional_logfc_null")
            for ext in ["png", "pdf", "svg"]:
                _savefig_with_numeric_ticks(
                    fig,
                    plot_dir / f"{prefix}.{ext}",
                    ax,
                    dpi=self.config.dpi,
                    numeric_x=True,
                    numeric_y=True,
                )
            plt.close(fig)

    def plot_figureB_tf_activity_score_pooled_boxplots(self, df_activity: pd.DataFrame, df_stats: pd.DataFrame, output_dir: Path) -> None:
        if df_activity.empty:
            return

        plot_dir = output_dir / "figureB_tf_activity_score"
        plot_dir.mkdir(parents=True, exist_ok=True)

        df = df_activity.copy()
        df["tf_activity_score"] = pd.to_numeric(df["tf_activity_score"], errors="coerce")
        df = df.dropna(subset=["tf_activity_score"])
        if df.empty:
            return

        group_color_map = {"low": "#95B0B5", "high": "#E7A983"}
        group_label_map = {"low": "Low phospho", "high": "High phospho"}

        df.to_csv(plot_dir / "FigureB_signed_tf_activity_score_points.csv", index=False)

        for direction_short, sub in df.groupby("direction_short", dropna=False):
            sub = sub.copy()
            available_cancers = [c for c in self.config.cancer_types if c in set(sub["cancer_type"].astype(str))]
            if not available_cancers:
                available_cancers = sorted(sub["cancer_type"].astype(str).unique())
            cancers = self._ordered_cancers_by_activity_significance(
                df_stats,
                str(direction_short),
                available_cancers,
            )

            positions = []
            data_lists = []
            box_colors = []
            scatter_colors = []
            x_centers = []
            ticklabels = []
            stat_pos = {}
            current_x = 1.0
            pair_offset = 0.20

            plotted_rows = []
            for cancer_type in cancers:
                cancer_sub = sub[sub["cancer_type"].astype(str).eq(cancer_type)].copy()
                low_values = pd.to_numeric(
                    cancer_sub.loc[cancer_sub["phospho_group"].eq("low"), "tf_activity_score"],
                    errors="coerce",
                ).dropna()
                high_values = pd.to_numeric(
                    cancer_sub.loc[cancer_sub["phospho_group"].eq("high"), "tf_activity_score"],
                    errors="coerce",
                ).dropna()

                if len(low_values) < self.config.min_group_samples or len(high_values) < self.config.min_group_samples:
                    continue

                positions.extend([current_x - pair_offset, current_x + pair_offset])
                data_lists.extend([low_values.tolist(), high_values.tolist()])
                box_colors.extend([group_color_map["low"], group_color_map["high"]])
                scatter_colors.extend([group_color_map["low"], group_color_map["high"]])
                x_centers.append(current_x)
                ticklabels.append(cancer_type)
                stat_pos[cancer_type] = current_x
                current_x += 1.0

                plotted_rows.append(
                    {
                        "cancer_type": cancer_type,
                        "direction_short": direction_short,
                        "n_low_points": int(len(low_values)),
                        "n_high_points": int(len(high_values)),
                        "mean_low_activity": float(low_values.mean()),
                        "mean_high_activity": float(high_values.mean()),
                        "median_low_activity": float(low_values.median()),
                        "median_high_activity": float(high_values.median()),
                        "delta_high_minus_low": float(high_values.mean() - low_values.mean()),
                    }
                )

            if not data_lists:
                continue

            plotted_summary = pd.DataFrame(plotted_rows)
            prefix_base = self._sanitize_filename(f"FigureB_{direction_short}_signed_tf_activity_score_by_cancer")
            cancer_order_map = {str(cancer): idx + 1 for idx, cancer in enumerate(cancers)}
            plotted_summary["plot_order"] = plotted_summary["cancer_type"].astype(str).map(cancer_order_map)
            stat_sub = df_stats[df_stats["direction_short"].astype(str).eq(str(direction_short))].copy() if not df_stats.empty else pd.DataFrame()
            if not stat_sub.empty:
                plotted_summary = plotted_summary.merge(
                    stat_sub[
                        [
                            "cancer_type",
                            "mannwhitney_p",
                            "mannwhitney_q_bh",
                            "significance_bh",
                            "significance",
                        ]
                    ],
                    on="cancer_type",
                    how="left",
                )
            plotted_summary.to_csv(plot_dir / f"{prefix_base}_summary.csv", index=False)

            fig_width = max(11.0, 0.85 * len(x_centers) + 3.0)
            fig_height = 5.0
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            rng = np.random.default_rng(self.config.random_seed)
            self._draw_scatter_then_boxplot(
                ax,
                data_lists,
                positions,
                box_colors,
                scatter_colors=scatter_colors,
                widths=0.34,
                box_alpha=0.88,
                scatter_s=3.5,
                scatter_alpha=0.12,
                jitter_range=0.06,
                rng=rng,
            )

            y_cap = 2.5
            ax.set_ylim(-y_cap, y_cap)

            ax.set_xticks(x_centers)
            ax.set_xticklabels(ticklabels, rotation=35, ha="right", fontsize=14.5)
            ax.set_ylabel("Signed regulon activity score", fontsize=15.5)
            ax.set_xlabel("Cancer type", fontsize=15.5)
            ax.set_yticks([-2, -1, 0, 1, 2])
            ax.tick_params(axis="y", labelsize=14.5)
            ax.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.30)
            ax.grid(axis="x", visible=False)
            sns.despine(ax=ax)
            ax.spines["left"].set_linewidth(0.85)
            ax.spines["left"].set_color("#4A4A4A")

            star_y = y_cap * 0.90
            stat_sub = df_stats[df_stats["direction_short"].astype(str).eq(str(direction_short))].copy() if not df_stats.empty else pd.DataFrame()
            if not stat_sub.empty:
                for _, stat_row in stat_sub.iterrows():
                    cancer_type = str(stat_row.get("cancer_type", ""))
                    sig_label = self._plot_significance_from_row(stat_row)
                    if sig_label == "ns" or cancer_type not in stat_pos:
                        continue
                    ax.text(
                        stat_pos[cancer_type],
                        star_y,
                        sig_label,
                        ha="center",
                        va="center",
                        fontsize=17,
                        color="#2F2F2F",
                    )

            handles = [
                plt.Line2D(
                    [0], [0], marker="s", linestyle="", markersize=7,
                    markerfacecolor=group_color_map["low"], markeredgecolor="#3A3A3A",
                    label=group_label_map["low"],
                ),
                plt.Line2D(
                    [0], [0], marker="s", linestyle="", markersize=7,
                    markerfacecolor=group_color_map["high"], markeredgecolor="#3A3A3A",
                    label=group_label_map["high"],
                ),
            ]
            ax.legend(
                handles=handles,
                frameon=False,
                fontsize=14,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0.0,
            )
            plt.tight_layout(rect=[0, 0, 0.88, 1])

            for ext in ["png", "pdf", "svg"]:
                _savefig_with_numeric_ticks(
                    fig,
                    plot_dir / f"{prefix_base}_boxplot.{ext}",
                    ax,
                    dpi=self.config.dpi,
                    numeric_x=False,
                    numeric_y=True,
                )
            plt.close(fig)

    def run_target_logfc_activity_random_analysis(
        self,
        df_all_points: pd.DataFrame,
        output_dir: Path,
        df_all_summary: Optional[pd.DataFrame] = None,
        run_confounder: bool = True,
    ) -> pd.DataFrame:
        analysis_dir = output_dir / "target_logfc_activity_random_analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        df_summary = df_all_summary.copy() if df_all_summary is not None else pd.DataFrame()

        obsolete_dirs = [
            "figureC_global_random_mean_null",
            "import_activate_focus",
            "target_logfc_distribution",
            "tf_activity_score_boxplots",
        ]
        for dirname in obsolete_dirs:
            old_dir = analysis_dir / dirname
            if old_dir.exists():
                shutil.rmtree(old_dir)
                print(f"Removed obsolete extended-analysis directory: {old_dir}")

        if df_all_points.empty:
            print("No expression points available for extended analysis.")
            return df_summary

        print("Building target logFC table for Figure A...")
        df_logfc = self.build_target_logfc_table(df_all_points)

        if not df_logfc.empty:
            print(
                f"Running pooled matched random control for Figure A "
                f"({self.config.random_iterations} iterations each)..."
            )
            df_matched_gene, _, _ = self.build_pooled_matched_random_control(
                df_logfc,
                n_random_sets=self.config.random_iterations,
                seed=self.config.random_seed,
                max_random_gene_rows_per_group=self.config.max_random_gene_rows_per_group,
            )

            figure_a_dir = analysis_dir / "figureA_gene_level_observed_vs_random_logfc_by_cancer"
            figure_a_dir.mkdir(parents=True, exist_ok=True)
            df_logfc.to_csv(figure_a_dir / "FigureA_target_gene_logfc_by_site_target.csv", index=False)
            df_matched_gene.to_csv(
                figure_a_dir / "FigureA_gene_level_observed_and_matched_random_logfc.csv",
                index=False,
            )

            print("Plotting Figure A gene-level observed vs matched-random logFC by cancer...")
            self.plot_figureA_gene_level_logfc_observed_vs_random(df_matched_gene, analysis_dir)
            print("Plotting Figure A forest plot (observed vs matched-random effect size by cancer)...")
            self.plot_figureA_gene_level_forest(df_matched_gene, analysis_dir)

        df_activity = pd.DataFrame()
        if not df_all_points.empty:
            print("Building TF activity scores for Figure B...")
            df_activity = self.build_tf_activity_table(df_all_points)

        if not df_activity.empty:
            print("Comparing TF activity between high/low phospho groups by cancer...")
            df_activity_cancer_stats = self.compare_tf_activity_by_cancer_direction(df_activity)
            figure_b_dir = analysis_dir / "figureB_tf_activity_score"
            figure_b_dir.mkdir(parents=True, exist_ok=True)
            df_activity_cancer_stats.to_csv(
                figure_b_dir / "FigureB_tf_activity_by_cancer_stats.csv",
                index=False,
            )

            print("Plotting Figure B TF activity score boxplots by cancer (ordered by significance)...")
            self.plot_figureB_tf_activity_score_pooled_boxplots(
                df_activity,
                df_activity_cancer_stats,
                analysis_dir,
            )

            print("Plotting sitewise TF activity score boxplots within each cancer...")
            self.plot_all_sitewise_activity_boxplots(df_activity, analysis_dir)

        print("Extended analysis complete. Kept Figure A, Figure B pooled, and sitewise activity outputs.")

        if run_confounder and not df_summary.empty:
            df_summary = self.run_confounder_sensitivity_analysis(
                df_all_points,
                df_summary,
                analysis_dir,
                df_activity=df_activity if not df_activity.empty else None,
            )

        return df_summary

    def _get_site_phospho_series(
        self,
        site: str,
        df_phospho: pd.DataFrame,
        df_rna: pd.DataFrame,
        df_protein: Optional[pd.DataFrame] = None,
        phospho_value_mode: Optional[str] = None,
    ) -> pd.Series:
        matched_samples = [s for s in df_rna.columns if s in df_phospho.columns]
        if len(matched_samples) == 0:
            matched_samples = list(df_rna.columns)
        phospho_series = pd.to_numeric(df_phospho.loc[site, matched_samples], errors="coerce")
        return self._transform_phospho_series_for_split(
            site,
            phospho_series,
            df_protein,
            phospho_value_mode,
        )

    def _fit_adjusted_phospho_association(
        self,
        y: pd.Series,
        phospho: pd.Series,
        tf_gene_id: str,
        df_rna: pd.DataFrame,
        df_protein: pd.DataFrame,
        df_phenotype: pd.DataFrame,
    ) -> Tuple[float, float, int]:
        samples = phospho.dropna().index.intersection(y.index)
        if len(samples) == 0:
            return np.nan, np.nan, 0

        design = pd.DataFrame({"phospho": pd.to_numeric(phospho.loc[samples], errors="coerce")})
        design["y"] = pd.to_numeric(y.loc[samples], errors="coerce")

        for covariate in self.config.adjustment_covariates:
            if covariate == "tf_mrna":
                design["tf_mrna"] = self._gene_abundance_series(tf_gene_id, df_rna, samples)
            elif covariate == "tf_protein":
                design["tf_protein"] = self._gene_abundance_series(tf_gene_id, df_protein, samples)
            elif covariate == "purity":
                design["purity"] = self._sample_covariate_series(
                    df_phenotype,
                    samples,
                    self.config.purity_column,
                )
            else:
                raise ValueError(f"Unsupported adjustment covariate: {covariate}")

        design = design.apply(pd.to_numeric, errors="coerce").dropna()
        min_samples = self.config.min_samples_for_adjustment
        if len(design) < min_samples:
            return np.nan, np.nan, int(len(design))

        x_cols = ["phospho"] + [
            col
            for col in ["tf_mrna", "tf_protein", "purity"]
            if col in design.columns and col != "phospho"
        ]
        x_matrix = np.column_stack([np.ones(len(design), dtype=float), design[x_cols].to_numpy(dtype=float)])
        coef, p_value = self._fit_ols_coef_pvalue(design["y"].to_numpy(dtype=float), x_matrix, coef_idx=1)
        return coef, p_value, int(len(design))

    def build_abundance_diagnostic_table(self, df_all_summary: pd.DataFrame) -> pd.DataFrame:
        if df_all_summary.empty:
            return pd.DataFrame()

        success = df_all_summary.loc[df_all_summary["status"].eq("success")].copy()
        if success.empty:
            return pd.DataFrame()

        rows: List[Dict[str, object]] = []
        rna_cache: Dict[str, pd.DataFrame] = {}
        protein_cache: Dict[str, pd.DataFrame] = {}
        phospho_cache: Dict[str, pd.DataFrame] = {}

        for _, site_row in success.iterrows():
            cancer_type = str(site_row["cancer_type"])
            direction = str(site_row["direction"])
            site = str(site_row["site"])
            tf_name = str(site_row.get("tf_name", ""))
            tf_gene_id = self._site_tf_gene_id(site)

            if cancer_type not in rna_cache:
                rna_cache[cancer_type] = self.load_rna(cancer_type)
                protein_cache[cancer_type] = self.load_protein(cancer_type)
                phospho_cache[cancer_type] = self.load_phospho(cancer_type)

            df_rna = rna_cache[cancer_type]
            df_protein = protein_cache[cancer_type]
            df_phospho = phospho_cache[cancer_type]

            split_info = self._get_phospho_high_low_samples(
                site,
                df_phospho,
                df_rna,
                phospho_value_mode="site_abundance",
            )
            if split_info.get("status") != "success":
                continue

            low_samples = list(split_info["low_samples"])
            high_samples = list(split_info["high_samples"])
            valid_samples = low_samples + high_samples
            phospho_series = self._get_site_phospho_series(
                site,
                df_phospho,
                df_rna,
                phospho_value_mode="site_abundance",
            )
            tf_mrna = self._gene_abundance_series(tf_gene_id, df_rna, valid_samples)
            tf_protein = self._gene_abundance_series(tf_gene_id, df_protein, valid_samples)

            rows.append(
                {
                    "cancer_type": cancer_type,
                    "direction": direction,
                    "direction_short": self._direction_short(direction),
                    "site": site,
                    "tf_name": tf_name,
                    "tf_gene_id": tf_gene_id,
                    "n_low_samples": len(low_samples),
                    "n_high_samples": len(high_samples),
                    "tf_mrna_mean_low": float(tf_mrna.loc[low_samples].mean()) if low_samples else np.nan,
                    "tf_mrna_mean_high": float(tf_mrna.loc[high_samples].mean()) if high_samples else np.nan,
                    "tf_mrna_mw_p": self._safe_mannwhitney(tf_mrna.loc[high_samples], tf_mrna.loc[low_samples]),
                    "tf_protein_mean_low": float(tf_protein.loc[low_samples].mean()) if low_samples else np.nan,
                    "tf_protein_mean_high": float(tf_protein.loc[high_samples].mean()) if high_samples else np.nan,
                    "tf_protein_mw_p": self._safe_mannwhitney(tf_protein.loc[high_samples], tf_protein.loc[low_samples]),
                    "spearman_phospho_tf_mrna": self._safe_spearman(phospho_series, tf_mrna),
                    "spearman_phospho_tf_protein": self._safe_spearman(phospho_series, tf_protein),
                    "tf_mrna_high_gt_low": bool(tf_mrna.loc[high_samples].mean() > tf_mrna.loc[low_samples].mean())
                    if low_samples and high_samples
                    else np.nan,
                    "tf_protein_high_gt_low": bool(tf_protein.loc[high_samples].mean() > tf_protein.loc[low_samples].mean())
                    if low_samples and high_samples
                    else np.nan,
                }
            )

        return pd.DataFrame(rows)

    def run_phospho_protein_ratio_sensitivity(
        self,
        df_all_summary: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if df_all_summary.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        success = df_all_summary.loc[df_all_summary["status"].eq("success")].copy()
        if success.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        point_frames: List[pd.DataFrame] = []
        summary_frames: List[pd.DataFrame] = []
        rna_cache: Dict[str, pd.DataFrame] = {}
        protein_cache: Dict[str, pd.DataFrame] = {}
        phospho_cache: Dict[str, pd.DataFrame] = {}
        chip_cache: Dict[str, pd.DataFrame] = {}

        for _, site_row in success.iterrows():
            cancer_type = str(site_row["cancer_type"])
            direction = str(site_row["direction"])

            if cancer_type not in rna_cache:
                rna_cache[cancer_type] = self.load_rna(cancer_type)
                protein_cache[cancer_type] = self.load_protein(cancer_type)
                phospho_cache[cancer_type] = self.load_phospho(cancer_type)

            try:
                rows, summary = self.build_site_target_expression_rows(
                    cancer_type=cancer_type,
                    direction=direction,
                    site_row=site_row,
                    df_rna=rna_cache[cancer_type],
                    df_phospho=phospho_cache[cancer_type],
                    chip_cache=chip_cache,
                    df_protein=protein_cache[cancer_type],
                    phospho_value_mode="phospho_minus_protein",
                )
            except Exception as exc:
                print(
                    f"Warning: phospho-minus-protein sensitivity skipped "
                    f"{cancer_type} | {site_row.get('site', '')}: {exc}"
                )
                continue

            if summary.get("status") != "success" or len(rows) == 0:
                continue
            point_frames.append(pd.DataFrame(rows))
            summary_frames.append(pd.DataFrame([summary]))

        df_points = pd.concat(point_frames, axis=0).reset_index(drop=True) if point_frames else pd.DataFrame()
        df_summary = pd.concat(summary_frames, axis=0).reset_index(drop=True) if summary_frames else pd.DataFrame()
        df_logfc = self.build_target_logfc_table(df_points)
        return df_points, df_summary, df_logfc

    def build_adjusted_target_association_table(
        self,
        df_all_summary: pd.DataFrame,
        df_all_points: pd.DataFrame,
    ) -> pd.DataFrame:
        if df_all_summary.empty or df_all_points.empty:
            return pd.DataFrame()

        success_keys = df_all_summary.loc[df_all_summary["status"].eq("success"), ["cancer_type", "direction_short", "site"]]
        if success_keys.empty:
            return pd.DataFrame()

        df_points = df_all_points.merge(
            success_keys.drop_duplicates(),
            on=["cancer_type", "direction_short", "site"],
            how="inner",
        )
        if df_points.empty:
            return pd.DataFrame()

        rows: List[Dict[str, object]] = []
        rna_cache: Dict[str, pd.DataFrame] = {}
        protein_cache: Dict[str, pd.DataFrame] = {}
        phospho_cache: Dict[str, pd.DataFrame] = {}
        phenotype_cache: Dict[str, pd.DataFrame] = {}

        group_cols = ["cancer_type", "direction_short", "site", "target_gene_id", "target_regulation"]
        for keys, sub in df_points.groupby(group_cols, dropna=False):
            cancer_type, direction_short, site, target_gene_id, target_regulation = keys
            cancer_type = str(cancer_type)
            direction_short = str(direction_short)
            site = str(site)
            target_gene_id = str(target_gene_id)
            target_regulation = str(target_regulation)
            tf_gene_id = self._site_tf_gene_id(site)

            if cancer_type not in rna_cache:
                rna_cache[cancer_type] = self.load_rna(cancer_type)
                protein_cache[cancer_type] = self.load_protein(cancer_type)
                phospho_cache[cancer_type] = self.load_phospho(cancer_type)
                phenotype_cache[cancer_type] = self.load_phenotype(cancer_type)

            df_rna = rna_cache[cancer_type]
            df_protein = protein_cache[cancer_type]
            df_phospho = phospho_cache[cancer_type]
            df_phenotype = phenotype_cache[cancer_type]

            phospho_series = self._get_site_phospho_series(
                site,
                df_phospho,
                df_rna,
                phospho_value_mode="site_abundance",
            )
            samples = phospho_series.dropna().index
            if target_gene_id not in df_rna.index or len(samples) == 0:
                continue

            y = pd.to_numeric(df_rna.loc[target_gene_id, samples], errors="coerce")
            coef, p_value, n_used = self._fit_adjusted_phospho_association(
                y,
                phospho_series,
                tf_gene_id,
                df_rna,
                df_protein,
                df_phenotype,
            )
            expected_sign = self._expected_effect_sign(direction_short, target_regulation)
            rows.append(
                {
                    "cancer_type": cancer_type,
                    "direction_short": direction_short,
                    "site": site,
                    "site_label": sub["site_label"].dropna().astype(str).iloc[0]
                    if "site_label" in sub.columns and sub["site_label"].notna().any()
                    else site,
                    "tf_name": sub["tf_name"].dropna().astype(str).iloc[0]
                    if "tf_name" in sub.columns and sub["tf_name"].notna().any()
                    else np.nan,
                    "target_gene_id": target_gene_id,
                    "target_gene_name": sub["target_gene_name"].dropna().astype(str).iloc[0]
                    if "target_gene_name" in sub.columns and sub["target_gene_name"].notna().any()
                    else np.nan,
                    "target_regulation": target_regulation,
                    "expected_sign": expected_sign,
                    "n_samples_used": n_used,
                    "adjusted_phospho_coef": coef,
                    "adjusted_phospho_p": p_value,
                    "directional_adjusted_phospho_coef": coef * expected_sign if pd.notna(coef) else np.nan,
                    "adjustment_covariates": ",".join(self.config.adjustment_covariates),
                }
            )

        out = pd.DataFrame(rows)
        if not out.empty:
            out["adjusted_phospho_q_bh"] = self._bh_adjust(out["adjusted_phospho_p"])
        return out

    def build_adjusted_activity_association_table(
        self,
        df_all_summary: pd.DataFrame,
        df_activity: pd.DataFrame,
    ) -> pd.DataFrame:
        if df_all_summary.empty or df_activity.empty:
            return pd.DataFrame()

        success_keys = df_all_summary.loc[df_all_summary["status"].eq("success"), ["cancer_type", "direction_short", "site"]]
        df_activity = df_activity.merge(
            success_keys.drop_duplicates(),
            on=["cancer_type", "direction_short", "site"],
            how="inner",
        )
        if df_activity.empty:
            return pd.DataFrame()

        rows: List[Dict[str, object]] = []
        rna_cache: Dict[str, pd.DataFrame] = {}
        protein_cache: Dict[str, pd.DataFrame] = {}
        phospho_cache: Dict[str, pd.DataFrame] = {}
        phenotype_cache: Dict[str, pd.DataFrame] = {}

        group_cols = ["cancer_type", "direction_short", "site", "tf_name"]
        for keys, sub in df_activity.groupby(group_cols, dropna=False):
            cancer_type, direction_short, site, tf_name = keys
            cancer_type = str(cancer_type)
            direction_short = str(direction_short)
            site = str(site)
            tf_name = str(tf_name)
            tf_gene_id = self._site_tf_gene_id(site)

            if cancer_type not in rna_cache:
                rna_cache[cancer_type] = self.load_rna(cancer_type)
                protein_cache[cancer_type] = self.load_protein(cancer_type)
                phospho_cache[cancer_type] = self.load_phospho(cancer_type)
                phenotype_cache[cancer_type] = self.load_phenotype(cancer_type)

            df_rna = rna_cache[cancer_type]
            df_protein = protein_cache[cancer_type]
            df_phospho = phospho_cache[cancer_type]
            df_phenotype = phenotype_cache[cancer_type]

            phospho_series = self._get_site_phospho_series(
                site,
                df_phospho,
                df_rna,
                phospho_value_mode="site_abundance",
            )
            activity = sub.set_index("sample")["tf_activity_score"]
            coef, p_value, n_used = self._fit_adjusted_phospho_association(
                activity,
                phospho_series,
                tf_gene_id,
                df_rna,
                df_protein,
                df_phenotype,
            )
            rows.append(
                {
                    "cancer_type": cancer_type,
                    "direction_short": direction_short,
                    "site": site,
                    "site_label": sub["site_label"].dropna().astype(str).iloc[0]
                    if "site_label" in sub.columns and sub["site_label"].notna().any()
                    else site,
                    "tf_name": tf_name,
                    "n_samples_used": n_used,
                    "adjusted_phospho_coef": coef,
                    "adjusted_phospho_p": p_value,
                    "adjustment_covariates": ",".join(self.config.adjustment_covariates),
                }
            )

        out = pd.DataFrame(rows)
        if not out.empty:
            out["adjusted_phospho_q_bh"] = self._bh_adjust(out["adjusted_phospho_p"])
        return out

    def build_robustness_summary(
        self,
        df_all_summary: pd.DataFrame,
        df_diagnostic: pd.DataFrame,
        df_ratio_logfc: pd.DataFrame,
        df_adj_target: pd.DataFrame,
        df_adj_activity: pd.DataFrame,
    ) -> pd.DataFrame:
        if df_all_summary.empty:
            return pd.DataFrame()

        base = df_all_summary.loc[
            df_all_summary["status"].eq("success"),
            ["cancer_type", "direction_short", "site", "tf_name"],
        ].drop_duplicates().copy()
        if base.empty:
            return pd.DataFrame()

        if not df_diagnostic.empty:
            diag_cols = [
                "cancer_type",
                "direction_short",
                "site",
                "tf_mrna_mw_p",
                "tf_protein_mw_p",
                "spearman_phospho_tf_mrna",
                "spearman_phospho_tf_protein",
                "tf_mrna_high_gt_low",
                "tf_protein_high_gt_low",
            ]
            base = base.merge(df_diagnostic[diag_cols], on=["cancer_type", "direction_short", "site"], how="left")

        if not df_ratio_logfc.empty:
            ratio_summary = self.summarize_target_logfc(df_ratio_logfc)
            if not ratio_summary.empty:
                ratio_summary = ratio_summary.rename(
                    columns={
                        "mean_directional_logFC": "ratio_mean_directional_logFC",
                        "wilcoxon_p_directional_gt0": "ratio_site_wilcoxon_p",
                        "wilcoxon_q_bh": "ratio_site_wilcoxon_q_bh",
                    }
                )
                ratio_site = (
                    ratio_summary.groupby(["cancer_type", "direction_short", "site"], dropna=False)
                    .agg(
                        ratio_mean_directional_logFC=("ratio_mean_directional_logFC", "mean"),
                        ratio_site_wilcoxon_p=("ratio_site_wilcoxon_p", "min"),
                        ratio_site_wilcoxon_q_bh=("ratio_site_wilcoxon_q_bh", "min"),
                    )
                    .reset_index()
                )
                base = base.merge(
                    ratio_site,
                    on=["cancer_type", "direction_short", "site"],
                    how="left",
                )

        if not df_adj_target.empty:
            target_site = (
                df_adj_target.groupby(["cancer_type", "direction_short", "site"], dropna=False)
                .agg(
                    n_adjusted_targets=("target_gene_id", "nunique"),
                    mean_directional_adjusted_coef=("directional_adjusted_phospho_coef", "mean"),
                    fraction_adjusted_p_lt_0p05=(
                        "adjusted_phospho_p",
                        lambda s: float(np.mean(pd.to_numeric(s, errors="coerce") < 0.05)),
                    ),
                )
                .reset_index()
            )
            base = base.merge(target_site, on=["cancer_type", "direction_short", "site"], how="left")

        if not df_adj_activity.empty:
            activity_site = df_adj_activity[
                ["cancer_type", "direction_short", "site", "adjusted_phospho_coef", "adjusted_phospho_p"]
            ].rename(
                columns={
                    "adjusted_phospho_coef": "activity_adjusted_phospho_coef",
                    "adjusted_phospho_p": "activity_adjusted_phospho_p",
                }
            )
            base = base.merge(activity_site, on=["cancer_type", "direction_short", "site"], how="left")

        base["ratio_site_significant"] = (
            pd.to_numeric(base["ratio_site_wilcoxon_p"], errors="coerce") < 0.05
            if "ratio_site_wilcoxon_p" in base.columns
            else np.nan
        )
        base["activity_adjusted_significant"] = (
            pd.to_numeric(base["activity_adjusted_phospho_p"], errors="coerce") < 0.05
            if "activity_adjusted_phospho_p" in base.columns
            else np.nan
        )
        if "tf_mrna_mw_p" in base.columns or "tf_protein_mw_p" in base.columns:
            tf_mrna_sig = (
                pd.to_numeric(base["tf_mrna_mw_p"], errors="coerce") < 0.05
                if "tf_mrna_mw_p" in base.columns
                else pd.Series(False, index=base.index)
            )
            tf_protein_sig = (
                pd.to_numeric(base["tf_protein_mw_p"], errors="coerce") < 0.05
                if "tf_protein_mw_p" in base.columns
                else pd.Series(False, index=base.index)
            )
            base["tf_abundance_confounding_flag"] = tf_mrna_sig | tf_protein_sig
        else:
            base["tf_abundance_confounding_flag"] = np.nan
        return base

    @staticmethod
    def merge_abundance_into_site_summary(
        df_all_summary: pd.DataFrame,
        df_robustness: pd.DataFrame,
    ) -> pd.DataFrame:
        if df_all_summary.empty or df_robustness.empty:
            return df_all_summary

        merge_keys = ["cancer_type", "direction_short", "site"]
        abundance_cols = [
            col
            for col in df_robustness.columns
            if col not in merge_keys + ["tf_name"]
        ]
        if not abundance_cols:
            return df_all_summary

        robustness_unique = df_robustness[merge_keys + abundance_cols].drop_duplicates(subset=merge_keys)
        out = df_all_summary.drop(columns=[c for c in abundance_cols if c in df_all_summary.columns], errors="ignore")
        return out.merge(robustness_unique, on=merge_keys, how="left")

    def run_confounder_sensitivity_analysis(
        self,
        df_all_points: pd.DataFrame,
        df_all_summary: pd.DataFrame,
        analysis_dir: Path,
        df_activity: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        modules = self._enabled_confounder_modules(self.config.confounder_analysis)
        if not modules:
            return df_all_summary

        abundance_dir = analysis_dir / "tf_abundance_sensitivity"
        abundance_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print(f"TF abundance sensitivity (integrated): {', '.join(sorted(modules))}")
        print("=" * 60)

        df_diagnostic = pd.DataFrame()
        df_ratio_logfc = pd.DataFrame()
        df_adj_target = pd.DataFrame()
        df_adj_activity = pd.DataFrame()

        if "diagnostic" in modules:
            print("TF abundance diagnostic...")
            df_diagnostic = self.build_abundance_diagnostic_table(df_all_summary)
            df_diagnostic.to_csv(abundance_dir / "abundance_diagnostic_by_site.csv", index=False)

        if "ratio" in modules:
            print("Phospho-minus-protein sensitivity split...")
            ratio_dir = abundance_dir / "phospho_minus_protein_split"
            ratio_dir.mkdir(parents=True, exist_ok=True)
            df_ratio_points, df_ratio_summary, df_ratio_logfc = self.run_phospho_protein_ratio_sensitivity(
                df_all_summary
            )
            df_ratio_points.to_csv(ratio_dir / "ratio_target_expression_points.csv", index=False)
            df_ratio_summary.to_csv(ratio_dir / "ratio_site_processing_summary.csv", index=False)
            df_ratio_logfc.to_csv(ratio_dir / "ratio_target_logfc_by_site_target.csv", index=False)
            if not df_ratio_logfc.empty:
                df_ratio_site_summary = self.summarize_target_logfc(df_ratio_logfc)
                df_ratio_site_summary.to_csv(ratio_dir / "ratio_target_logfc_site_summary.csv", index=False)

        if "adjusted" in modules:
            print("Covariate-adjusted association models...")
            df_adj_target = self.build_adjusted_target_association_table(df_all_summary, df_all_points)
            df_adj_target.to_csv(abundance_dir / "adjusted_target_association.csv", index=False)
            if df_activity is None:
                df_activity = self.build_tf_activity_table(df_all_points)
            if df_activity is not None and not df_activity.empty:
                df_adj_activity = self.build_adjusted_activity_association_table(df_all_summary, df_activity)
                df_adj_activity.to_csv(abundance_dir / "adjusted_activity_association.csv", index=False)

        df_robustness = self.build_robustness_summary(
            df_all_summary,
            df_diagnostic,
            df_ratio_logfc,
            df_adj_target,
            df_adj_activity,
        )
        if not df_robustness.empty:
            df_robustness.to_csv(abundance_dir / "robustness_summary_by_site.csv", index=False)

        df_enriched = self.merge_abundance_into_site_summary(df_all_summary, df_robustness)
        print(f"Saved TF abundance outputs: {abundance_dir}")
        print("TF abundance sensitivity complete.")
        return df_enriched

    def run_scan(self) -> Path:

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2)

        print("=" * 60)
        print("Target regulation boxplot scan")
        print("=" * 60)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output directory: {output_dir}")
        print(f"Cancer types: {', '.join(self.config.cancer_types)}")
        print(f"Directions: {', '.join(self.config.directions)}")
        print(f"Phospho split mode: {self.config.phospho_split_mode}")
        print(f"Phospho value mode: {self.config.phospho_value_mode}")
        print(f"Abundance primary mode: {self.config.abundance_primary_mode}")
        print(f"TF adjustment covariates: {', '.join(self.config.adjustment_covariates)}")
        print(f"Confounder analysis: {self.config.confounder_analysis}")
        print(f"Use BH p-value correction for plots: {self.config.use_bh_pvalue_correction}")

        point_frames: List[pd.DataFrame] = []
        summary_frames: List[pd.DataFrame] = []
        for cancer_type in self.config.cancer_types:
            for direction in self.config.directions:
                df_points, df_summary = self.analyze_cancer_direction(cancer_type, direction)
                if not df_points.empty:
                    point_frames.append(df_points)
                if not df_summary.empty:
                    summary_frames.append(df_summary)

        df_all_points = pd.concat(point_frames, axis=0).reset_index(drop=True) if point_frames else pd.DataFrame()
        df_all_summary = pd.concat(summary_frames, axis=0).reset_index(drop=True) if summary_frames else pd.DataFrame()

        if not df_all_points.empty and "expression_mode" not in df_all_points.columns:
            df_all_points["expression_mode"] = EXPRESSION_MODE_UNADJUSTED

        df_all_points.to_csv(output_dir / "all_target_gene_mean_expression_points_all_modes.csv", index=False)
        df_primary_points = self._filter_points_by_expression_mode(df_all_points, self._primary_expression_mode())
        df_baseline_points = self._filter_points_by_expression_mode(df_all_points, EXPRESSION_MODE_UNADJUSTED)

        df_primary_points.to_csv(output_dir / "all_target_gene_mean_expression_points.csv", index=False)
        df_all_summary.to_csv(output_dir / "site_level_processing_summary.csv", index=False)

        print(
            f"Primary expression mode for figures/tables: {self._primary_expression_mode()} "
            f"({len(df_primary_points)} point rows)"
        )
        print("Plotting sitewise high/low phospho target expression boxplots...")
        self.plot_all_boxplots(df_primary_points, output_dir)
        print("Plotting merged all-sites high/low boxplots...")
        self.plot_all_merged_boxplots(df_primary_points, output_dir)

        if self.config.abundance_primary_mode == "dual" and not df_baseline_points.empty:
            baseline_dir = output_dir / "baseline_unadjusted"
            baseline_dir.mkdir(parents=True, exist_ok=True)
            df_baseline_points.to_csv(baseline_dir / "all_target_gene_mean_expression_points.csv", index=False)
            print("Plotting baseline unadjusted boxplots (pre-TF-abundance primary)...")
            self.plot_all_boxplots(df_baseline_points, baseline_dir)
            self.plot_all_merged_boxplots(df_baseline_points, baseline_dir)
            self.save_abundance_impact_reports(df_all_points, output_dir)

        print("Running extended target logFC, random control, TF activity, and abundance sensitivity...")
        df_all_summary = self.run_target_logfc_activity_random_analysis(
            df_primary_points,
            output_dir,
            df_all_summary=df_all_summary,
        )
        if self.config.abundance_primary_mode == "dual" and not df_baseline_points.empty:
            baseline_dir = output_dir / "baseline_unadjusted"
            print("Running baseline unadjusted Figure A/B for comparison...")
            self.run_target_logfc_activity_random_analysis(
                df_baseline_points,
                baseline_dir,
                df_all_summary=df_all_summary,
                run_confounder=False,
            )
        df_all_summary.to_csv(output_dir / "site_level_processing_summary.csv", index=False)

        print("=" * 60)
        print(f"Saved all-mode point table: {output_dir / 'all_target_gene_mean_expression_points_all_modes.csv'}")
        print(f"Saved primary point table: {output_dir / 'all_target_gene_mean_expression_points.csv'}")
        print(f"Saved site summary (with abundance columns): {output_dir / 'site_level_processing_summary.csv'}")
        print(f"Saved extended analysis: {output_dir / 'target_logfc_activity_random_analysis'}")
        if self.config.abundance_primary_mode == "dual":
            print(f"Saved baseline unadjusted analysis: {output_dir / 'baseline_unadjusted'}")
        if self.config.confounder_analysis != "none":
            print(
                f"Saved TF abundance sensitivity: "
                f"{output_dir / 'target_logfc_activity_random_analysis' / 'tf_abundance_sensitivity'}"
            )
        print("Done")
        return output_dir / "all_target_gene_mean_expression_points.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw activate/repress target gene expression boxplots by cancer and transport direction.")
    parser.add_argument("mode", choices=["scan"], nargs="?", default="scan")
    parser.add_argument("--linkedomics-base", default=TempoConfig.linkedomics_base)
    parser.add_argument("--chip-dir", default=TempoConfig.chip_dir)
    parser.add_argument("--signed-regulon-path", default=TempoConfig.signed_regulon_path)
    parser.add_argument("--idmapping-path", default=TempoConfig.idmapping_path)
    parser.add_argument("--prediction-output-dir", default=TempoConfig.prediction_output_dir)
    parser.add_argument("--import-prediction-filename", default=TempoConfig.import_prediction_filename)
    parser.add_argument("--export-prediction-filename", default=TempoConfig.export_prediction_filename)
    parser.add_argument("--output-dir", default=TempoConfig.output_dir)
    parser.add_argument("--known-positive-path", default=TempoConfig.known_positive_path)
    parser.add_argument("--known-site-label-color", default=TempoConfig.known_site_label_color)
    parser.add_argument("--new-site-label-color", default=TempoConfig.new_site_label_color)
    parser.add_argument("--ensembl-release", type=int, default=TempoConfig.ensembl_release)
    parser.add_argument("--species", default=TempoConfig.species)
    parser.add_argument("--cancer-types", nargs="+", default=None)
    parser.add_argument("--directions", nargs="+", default=None, choices=VALID_DIRECTIONS)
    parser.add_argument("--max-missing-ratio", type=float, default=TempoConfig.max_missing_ratio)
    parser.add_argument("--min-nonzero-ratio", type=float, default=TempoConfig.min_nonzero_ratio)
    parser.add_argument("--max-nonzero-ratio", type=float, default=TempoConfig.max_nonzero_ratio)
    parser.add_argument("--chip-threshold", type=float, default=TempoConfig.chip_threshold)
    parser.add_argument("--min-chip-sample-frac", type=float, default=TempoConfig.min_chip_sample_frac)
    parser.add_argument("--chip-top-n", type=int, default=TempoConfig.chip_top_n)
    parser.add_argument("--signed-target-mode", choices=["chip_intersection", "regulon_only"], default=TempoConfig.signed_target_mode)
    parser.add_argument("--phospho-group-frac", type=float, default=TempoConfig.phospho_group_frac)
    parser.add_argument("--phospho-split-mode", choices=VALID_PHOSPHO_SPLIT_MODES, default=TempoConfig.phospho_split_mode)
    parser.add_argument("--min-group-samples", type=int, default=TempoConfig.min_group_samples)
    parser.add_argument("--min-box-points", type=int, default=TempoConfig.min_box_points)
    parser.add_argument("--exclude-zero-phospho-for-split", action="store_true")
    parser.add_argument("--dpi", type=int, default=TempoConfig.dpi)
    parser.add_argument("--random-iterations", type=int, default=TempoConfig.random_iterations)
    parser.add_argument("--random-seed", type=int, default=TempoConfig.random_seed)
    parser.add_argument("--max-random-gene-rows-per-group", type=int, default=TempoConfig.max_random_gene_rows_per_group)
    parser.add_argument(
        "--abundance-primary-mode",
        choices=VALID_ABUNDANCE_PRIMARY_MODES,
        default=TempoConfig.abundance_primary_mode,
        help=(
            "How TF abundance enters the primary analysis: unadjusted (legacy), "
            "residual (TF-adjusted expression/activity only; uses --adjustment-covariates), "
            "or dual (both with comparison)."
        ),
    )
    parser.add_argument(
        "--confounder-analysis",
        choices=VALID_CONFOUNDER_ANALYSES,
        default=TempoConfig.confounder_analysis,
        help="Optional TF-abundance sensitivity modules: diagnostic, ratio, adjusted, or full.",
    )
    parser.add_argument(
        "--phospho-value-mode",
        choices=VALID_PHOSPHO_VALUE_MODES,
        default=TempoConfig.phospho_value_mode,
        help="Primary phospho split metric. Use site_abundance for the default main analysis.",
    )
    parser.add_argument(
        "--adjustment-covariates",
        nargs="+",
        default=None,
        choices=["tf_mrna", "tf_protein", "purity"],
        help=(
            "TF abundance covariates for residual primary analysis and adjusted association models "
            "(default: tf_protein only)."
        ),
    )
    parser.add_argument("--purity-column", default=TempoConfig.purity_column)
    parser.add_argument(
        "--min-samples-for-adjustment",
        type=int,
        default=TempoConfig.min_samples_for_adjustment,
    )
    parser.add_argument(
        "--use-bh-pvalue-correction",
        action=argparse.BooleanOptionalAction,
        default=TempoConfig.use_bh_pvalue_correction,
        help="Apply Benjamini-Hochberg correction when annotating plots with significance stars.",
    )
    return parser.parse_args()


def _build_config_from_args(args: argparse.Namespace, split_mode: str, output_dir: str) -> TempoConfig:
    return TempoConfig(
        linkedomics_base=args.linkedomics_base,
        chip_dir=args.chip_dir,
        signed_regulon_path=args.signed_regulon_path,
        idmapping_path=args.idmapping_path,
        prediction_output_dir=args.prediction_output_dir,
        import_prediction_filename=args.import_prediction_filename,
        export_prediction_filename=args.export_prediction_filename,
        output_dir=output_dir,
        known_positive_path=args.known_positive_path,
        known_site_label_color=args.known_site_label_color,
        new_site_label_color=args.new_site_label_color,
        ensembl_release=args.ensembl_release,
        species=args.species,
        cancer_types=args.cancer_types if args.cancer_types is not None else list(DEFAULT_CANCER_LIST),
        directions=args.directions if args.directions is not None else list(DEFAULT_DIRECTIONS),
        max_missing_ratio=args.max_missing_ratio,
        min_nonzero_ratio=args.min_nonzero_ratio,
        max_nonzero_ratio=args.max_nonzero_ratio,
        chip_threshold=args.chip_threshold,
        min_chip_sample_frac=args.min_chip_sample_frac,
        chip_top_n=args.chip_top_n,
        signed_target_mode=args.signed_target_mode,
        phospho_group_frac=args.phospho_group_frac,
        phospho_split_mode=split_mode,
        min_group_samples=args.min_group_samples,
        min_box_points=args.min_box_points,
        exclude_zero_phospho_for_split=args.exclude_zero_phospho_for_split,
        dpi=args.dpi,
        random_iterations=args.random_iterations,
        random_seed=args.random_seed,
        max_random_gene_rows_per_group=args.max_random_gene_rows_per_group,
        confounder_analysis=args.confounder_analysis,
        abundance_primary_mode=args.abundance_primary_mode,
        phospho_value_mode=args.phospho_value_mode,
        adjustment_covariates=(
            args.adjustment_covariates
            if args.adjustment_covariates is not None
            else list(DEFAULT_ADJUSTMENT_COVARIATES)
        ),
        purity_column=args.purity_column,
        min_samples_for_adjustment=args.min_samples_for_adjustment,
        use_bh_pvalue_correction=args.use_bh_pvalue_correction,
    )


def main() -> None:
    args = parse_args()
    if args.phospho_split_mode == "both":
        split_modes = ["observed_missing", "median_nonmissing"]
    else:
        split_modes = [args.phospho_split_mode]

    base_output_dir = Path(args.output_dir)
    for split_mode in split_modes:
        if len(split_modes) == 1:
            output_dir = str(base_output_dir)
        else:
            output_dir = str(base_output_dir / f"split_{split_mode}")
        config = _build_config_from_args(args, split_mode=split_mode, output_dir=output_dir)
        pipeline = TargetRegulationBoxplotPipeline(config)
        pipeline.run_scan()

if __name__ == "__main__":
    main()
