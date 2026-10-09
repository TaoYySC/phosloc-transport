#!/usr/bin/env python3
"""CPTAC three-stage phospho hotspot validation (mixed/pred-filter definition).

Stage 1: within-hotspot co-phosphorylation coherence (no Import/Export/target filter)
Stage 2: hotspot vs single-site coverage & Import×Activate association
Stage 3: Known_only / Mixed / Predicted_only signed-beta validation

Does NOT overwrite existing mixed_pred_filter or three_level result directories.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from run_hotspot_target_regulation_analysis import zscore_rows  # noqa: E402
from run_import_target_regulation_analysis import (  # noqa: E402
    DEFAULT_CANCER_LIST,
    TargetRegulationBoxplotPipeline,
    TempoConfig,
)

_CPTAC_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _CPTAC_ROOT.parent

DEFAULT_HOTSPOT_DIR = (
    _CPTAC_ROOT
    / "results/import_target_regulation_hotspot_v11_147pos_d3_platt/hotspots_mixed_pred_filter"
)
DEFAULT_SUMMARY_CSV = (
    _REPO_ROOT
    / "import_export/results/tf_phos_site_FuncTransport_Direction_summary_with_PMID_annotation_v11_147pos_d3_platt.csv"
)
DEFAULT_OUTPUT_DIR = (
    _CPTAC_ROOT
    / "results/three_stage_hotspot_validation_v11_147pos_d3_platt_mixed_pred_filter"
)

TYPE_MAP = {
    "Mixed evidence": "Mixed",
    "Predicted candidate": "Predicted_only",
    "Known_only": "Known_only",
}

MIN_OVERLAP = 20
DIST_TOL = 3
MIN_SCORE_SAMPLES = 10
MIN_OLS_SAMPLES = 10
# Far-distance same-protein control (aa); excludes catalog hotspot members.
FAR_DIST_MIN = 40
FAR_DIST_MAX = 100
# Sensitivity Stage3 for sparse Predicted_only coverage.
MIN_SCORE_SAMPLES_RELAXED = 5
# Figure 1B default exemplars (coverage / n_sites; not selected by Import/Export).
DEFAULT_FIGURE1B_HOTSPOT_IDS = (
    "P40763_H1_Y686_T708",  # STAT3_Y686-T708 Mixed
    "Q00613_H3_S292_S326",  # HSF1_S292-S326 Mixed
)


def bh_fdr(pvals: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(pvals), dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    valid = np.isfinite(arr)
    if valid.sum() == 0:
        return out
    p = arr[valid]
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(1, n + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    restored = np.empty(n, dtype=float)
    restored[order] = adj
    out[valid] = restored
    return out


def parse_site_token(site: str) -> Tuple[str, int]:
    s = str(site).strip()
    residue = s[0].upper()
    position = int(s[1:])
    return residue, position


def is_nuclear_localization(value: object) -> bool:
    text = str(value) if pd.notna(value) else ""
    return "Nuclear" in text


def map_stage3_type(raw: object) -> str:
    key = str(raw).strip() if pd.notna(raw) else ""
    if key in TYPE_MAP:
        return TYPE_MAP[key]
    if key in {"Mixed", "Predicted_only", "Known_only"}:
        return key
    raise ValueError(f"Unrecognized hotspot_type: {raw!r}")


def load_candidate_tables(hotspot_dir: Path, distance: int = 15) -> Tuple[pd.DataFrame, pd.DataFrame]:
    hotspots = pd.read_csv(hotspot_dir / f"hotspots_d{distance}.csv")
    members = pd.read_csv(hotspot_dir / f"hotspot_members_d{distance}.csv")
    hotspots = hotspots.copy()
    members = members.copy()
    hotspots["stage3_type"] = hotspots["hotspot_type"].map(map_stage3_type)
    members["stage3_type"] = members["hotspot_type"].map(map_stage3_type)
    hotspots["source_cohort"] = "candidate_mixed_pred"
    members["source_cohort"] = "candidate_mixed_pred"
    return hotspots, members


def recover_known_only(
    hotspot_dir: Path,
    summary_csv: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    clusters = pd.read_csv(hotspot_dir / "all_clusters_adj15_span40_unfiltered.csv")
    known = clusters[
        (pd.to_numeric(clusters["n_known"], errors="coerce") >= 1)
        & (pd.to_numeric(clusters["n_predicted"], errors="coerce") == 0)
        & (pd.to_numeric(clusters["n_sites"], errors="coerce") >= 2)
    ].copy()
    summary = pd.read_csv(summary_csv)
    summary["protein_acc"] = summary["protein_acc"].astype(str)
    summary["site"] = summary["site"].astype(str)

    hotspot_rows: List[Dict[str, object]] = []
    member_rows: List[Dict[str, object]] = []
    for i, row in enumerate(known.itertuples(index=False), start=1):
        acc = str(row.protein_acc)
        gene = str(row.gene_name)
        sites = [s.strip() for s in str(row.member_sites).split(";") if s.strip()]
        start_pos = int(row.start_position)
        end_pos = int(row.end_position)
        hotspot_id = f"{acc}_K{i}_{row.start_site}_{row.end_site}"
        hotspot_label = f"{gene}_{row.start_site}-{row.end_site}"
        sub = summary[(summary["protein_acc"] == acc) & (summary["site"].isin(sites))].copy()
        # Keep cluster member order; fill missing rows from site tokens if needed.
        by_site = {str(r.site): r for r in sub.itertuples(index=False)}
        loc_annots: List[str] = []
        is_nuclear_flags: List[bool] = []
        for site in sites:
            residue, position = parse_site_token(site)
            if site in by_site:
                r = by_site[site]
                evidence = str(r.evidence)
                loc = r.Localization_annotation
                ft = r.FuncTransport_score
                ds = r.Direction_score
                region = r.Region
                tf_family = r.TF_family
            else:
                evidence = "Known"
                loc = np.nan
                ft = np.nan
                ds = np.nan
                region = np.nan
                tf_family = np.nan
            nuc = is_nuclear_localization(loc)
            loc_annots.append(str(loc) if pd.notna(loc) else "")
            is_nuclear_flags.append(nuc)
            member_rows.append(
                {
                    "hotspot_id": hotspot_id,
                    "hotspot_label": hotspot_label,
                    "protein_acc": acc,
                    "gene_name": gene,
                    "INDEX": f"{acc}_{site}",
                    "site": site,
                    "residue": residue,
                    "position": position,
                    "evidence": evidence,
                    "FuncTransport_score": ft,
                    "Direction_score": ds,
                    "Localization_annotation": loc,
                    "Region": region,
                    "TF_family": tf_family,
                    "is_nuclear_site": nuc,
                    "is_cytoplasmic_site": ("Cytoplasmic" in str(loc)) if pd.notna(loc) else False,
                    "hotspot_type": "Known_only",
                    "stage3_type": "Known_only",
                    "hotspot_class": "known_only_benchmark",
                    "direction_call": "nuclear" if nuc else "cytoplasmic",
                    "has_nuclear_member": False,  # filled below
                    "has_cytoplasmic_member": False,
                    "cluster_distance": 15,
                    "source_cohort": "known_only_benchmark",
                }
            )
        has_nuclear = any(is_nuclear_flags)
        has_cyto = any(("Cytoplasmic" in a) for a in loc_annots)
        for m in member_rows:
            if m["hotspot_id"] == hotspot_id:
                m["has_nuclear_member"] = has_nuclear
                m["has_cytoplasmic_member"] = has_cyto
                m["direction_call"] = "nuclear" if has_nuclear else ("cytoplasmic" if has_cyto else "unknown")
        hotspot_rows.append(
            {
                "hotspot_id": hotspot_id,
                "hotspot_label": hotspot_label,
                "protein_acc": acc,
                "gene_name": gene,
                "n_members": len(sites),
                "member_sites": ";".join(sites),
                "start_position": start_pos,
                "end_position": end_pos,
                "span_aa": int(row.span_aa),
                "n_known": int(row.n_known),
                "n_predicted": int(row.n_predicted),
                "n_sites": int(row.n_sites),
                "has_nuclear_member": has_nuclear,
                "has_cytoplasmic_member": has_cyto,
                "direction_call": "nuclear" if has_nuclear else ("cytoplasmic" if has_cyto else "unknown"),
                "hotspot_type": "Known_only",
                "stage3_type": "Known_only",
                "hotspot_class": "known_only_benchmark",
                "cluster_distance": 15,
                "cluster_span_max": 40,
                "source_cohort": "known_only_benchmark",
            }
        )
    return pd.DataFrame(hotspot_rows), pd.DataFrame(member_rows)


def annotate_members_with_ensembl(
    members: pd.DataFrame,
    pipeline: TargetRegulationBoxplotPipeline,
) -> pd.DataFrame:
    out = members.copy()
    out["ACC_ID"] = out["protein_acc"].astype(str)
    out["RESIDUE"] = out["residue"].astype(str).str.upper()
    out["POSITION"] = pd.to_numeric(out["position"], errors="coerce").astype("Int64")
    idmap = pipeline.load_idmapping().drop_duplicates("ACC_ID")
    out = out.merge(idmap, on="ACC_ID", how="left")
    out["ENSEMBL_GENE_ID"] = out["ENSEMBL_GENE_ID"].astype(str).str.split(".").str[0]
    out["cptac_site"] = (
        out["ENSEMBL_GENE_ID"] + "|" + out["RESIDUE"] + out["POSITION"].astype(str)
    )
    return out


def nuclear_mask(hotspots: pd.DataFrame, members: pd.DataFrame) -> pd.Series:
    """Prefer has_nuclear_member; else any member Localization_annotation Nuclear."""
    if "has_nuclear_member" in hotspots.columns:
        mask = hotspots["has_nuclear_member"].astype(bool)
        # If all False/NaN for a cohort without the column filled, fall back per-row.
        if mask.any() or "Localization_annotation" not in members.columns:
            return mask
    flags = []
    for hid in hotspots["hotspot_id"].astype(str):
        mem = members[members["hotspot_id"].astype(str).eq(hid)]
        if "has_nuclear_member" in mem.columns and mem["has_nuclear_member"].astype(bool).any():
            flags.append(True)
        elif "Localization_annotation" in mem.columns:
            flags.append(any(is_nuclear_localization(v) for v in mem["Localization_annotation"]))
        else:
            flags.append(False)
    return pd.Series(flags, index=hotspots.index)


def spearman_pair(x: pd.Series, y: pd.Series, min_overlap: int = MIN_OVERLAP) -> Tuple[float, float, int]:
    aligned = pd.concat([x, y], axis=1).apply(pd.to_numeric, errors="coerce").dropna()
    n = int(len(aligned))
    if n < min_overlap:
        return np.nan, np.nan, n
    if aligned.iloc[:, 0].nunique(dropna=True) < 2 or aligned.iloc[:, 1].nunique(dropna=True) < 2:
        return np.nan, np.nan, n
    rho, p = spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return float(rho), float(p), n


def protein_residuals(
    site_matrix: pd.DataFrame,
    protein_series: pd.Series,
) -> pd.DataFrame:
    """Regress each site on protein abundance; return residual matrix (same shape)."""
    out = pd.DataFrame(index=site_matrix.index, columns=site_matrix.columns, dtype=float)
    prot = pd.to_numeric(protein_series, errors="coerce")
    for site in site_matrix.index:
        y = pd.to_numeric(site_matrix.loc[site], errors="coerce")
        aligned = pd.concat([y, prot], axis=1).dropna()
        aligned.columns = ["y", "x"]
        if len(aligned) < MIN_OLS_SAMPLES or aligned["x"].nunique() < 2:
            continue
        xmat = np.column_stack([np.ones(len(aligned)), aligned["x"].to_numpy(dtype=float)])
        try:
            beta, _, rank, _ = np.linalg.lstsq(xmat, aligned["y"].to_numpy(dtype=float), rcond=None)
        except np.linalg.LinAlgError:
            continue
        if rank < 2:
            continue
        fitted = xmat @ beta
        resid = aligned["y"].to_numpy(dtype=float) - fitted
        out.loc[site, aligned.index] = resid
    return out


def fit_ols_with_covariate(
    pipeline: TargetRegulationBoxplotPipeline,
    y: pd.Series,
    x: pd.Series,
    covariate: Optional[pd.Series],
) -> Tuple[float, float, int]:
    df = pd.concat(
        {
            "y": pd.to_numeric(y, errors="coerce"),
            "x": pd.to_numeric(x, errors="coerce"),
            "cov": pd.to_numeric(covariate, errors="coerce") if covariate is not None else np.nan,
        },
        axis=1,
    )
    if covariate is None:
        df = df[["y", "x"]].dropna()
        if len(df) < MIN_OLS_SAMPLES:
            return np.nan, np.nan, int(len(df))
        xmat = np.column_stack([np.ones(len(df)), df["x"].to_numpy(dtype=float)])
    else:
        df = df.dropna()
        if len(df) < MIN_OLS_SAMPLES:
            return np.nan, np.nan, int(len(df))
        xmat = np.column_stack(
            [np.ones(len(df)), df["x"].to_numpy(dtype=float), df["cov"].to_numpy(dtype=float)]
        )
    beta, p = pipeline._fit_ols_coef_pvalue(df["y"].to_numpy(dtype=float), xmat, coef_idx=1)
    return beta, p, int(len(df))


class ThreeStageHotspotValidation:
    def __init__(
        self,
        config: TempoConfig,
        hotspots: pd.DataFrame,
        members: pd.DataFrame,
        output_dir: Path,
        min_overlap: int = MIN_OVERLAP,
        dist_tol: int = DIST_TOL,
        min_score_samples: int = MIN_SCORE_SAMPLES,
    ):
        self.config = config
        self.pipeline = TargetRegulationBoxplotPipeline(config)
        self.hotspots = hotspots.copy()
        self.members = annotate_members_with_ensembl(members, self.pipeline)
        self.hotspot_meta = self.hotspots.set_index("hotspot_id", drop=False)
        self.output_dir = Path(output_dir)
        self.min_overlap = int(min_overlap)
        self.dist_tol = int(dist_tol)
        self.min_score_samples = int(min_score_samples)
        self._phospho: Dict[str, pd.DataFrame] = {}
        self._rna: Dict[str, pd.DataFrame] = {}
        self._protein: Dict[str, pd.DataFrame] = {}
        self._activate_cache: Dict[str, pd.DataFrame] = {}
        self._chip_cache: Dict[str, pd.DataFrame] = {}

        for d in [
            self.output_dir / "stage1",
            self.output_dir / "stage2",
            self.output_dir / "stage3",
            self.output_dir / "funnel",
            self.output_dir / "figures",
            self.output_dir / "catalog",
        ]:
            d.mkdir(parents=True, exist_ok=True)

        # All CPTAC-style member tokens in the analysis catalog (any hotspot type).
        self.catalog_cptac_sites = set(
            self.members["cptac_site"].dropna().astype(str).tolist()
        ) if "cptac_site" in self.members.columns else set()

    def load_phospho(self, cancer: str) -> pd.DataFrame:
        if cancer not in self._phospho:
            self._phospho[cancer] = self.pipeline.load_phospho(cancer)
        return self._phospho[cancer]

    def load_rna(self, cancer: str) -> pd.DataFrame:
        if cancer not in self._rna:
            self._rna[cancer] = self.pipeline.load_rna(cancer)
        return self._rna[cancer]

    def load_protein(self, cancer: str) -> Optional[pd.DataFrame]:
        if cancer not in self._protein:
            try:
                self._protein[cancer] = self.pipeline.load_protein(cancer)
            except FileNotFoundError:
                self._protein[cancer] = None  # type: ignore
        return self._protein[cancer]

    def get_activate_targets(self, tf_name: str, df_rna: pd.DataFrame) -> pd.DataFrame:
        key = str(tf_name)
        if key in self._activate_cache:
            cached = self._activate_cache[key]
            return cached[cached["gene_id"].isin(df_rna.index)].copy()

        if key in self._chip_cache:
            df_chip = self._chip_cache[key]
        else:
            df_chip = self.pipeline.load_chip_targets(key)
            self._chip_cache[key] = df_chip
        if df_chip.empty:
            empty = pd.DataFrame(columns=["gene_id", "gene_name", "target_regulation"])
            self._activate_cache[key] = empty
            return empty.copy()
        chip_targets = self.pipeline.classify_chip_targets(df_chip)
        if chip_targets.empty:
            empty = pd.DataFrame(columns=["gene_id", "gene_name", "target_regulation"])
            self._activate_cache[key] = empty
            return empty.copy()
        chip_names = chip_targets["target"].dropna().astype(str).tolist()
        signed = self.pipeline.get_signed_targets_for_tf(key, chip_names)
        if signed.empty:
            empty = pd.DataFrame(columns=["gene_id", "gene_name", "target_regulation"])
            self._activate_cache[key] = empty
            return empty.copy()
        genes = self.pipeline.load_genes()[["gene_id", "gene_name", "gene_name_upper"]]
        info = signed.merge(genes, left_on="target_upper", right_on="gene_name_upper", how="left")
        info = info.dropna(subset=["gene_id"]).copy()
        info = info[info["target_regulation"].astype(str).eq("activate")].copy()
        info = info.drop_duplicates(subset=["gene_id"])
        self._activate_cache[key] = info[["gene_id", "gene_name", "target_regulation"]].copy()
        return self._activate_cache[key][self._activate_cache[key]["gene_id"].isin(df_rna.index)].copy()

    def target_activity_series(
        self,
        tf_name: str,
        df_rna: pd.DataFrame,
        samples: Sequence[str],
    ) -> Tuple[pd.Series, int]:
        targets = self.get_activate_targets(tf_name, df_rna)
        if targets.empty:
            return pd.Series(index=list(samples), dtype=float), 0
        gene_ids = [g for g in targets["gene_id"].astype(str) if g in df_rna.index]
        if not gene_ids:
            return pd.Series(index=list(samples), dtype=float), 0
        use_samples = [s for s in samples if s in df_rna.columns]
        mat = df_rna.loc[gene_ids, use_samples].apply(pd.to_numeric, errors="coerce")
        z = zscore_rows(mat)
        activity = z.mean(axis=0, skipna=True)
        out = pd.Series(index=list(samples), dtype=float)
        out.loc[activity.index] = activity
        return out, len(gene_ids)

    def hotspot_score_mean_z(
        self,
        member_sites: Sequence[str],
        df_phospho: pd.DataFrame,
    ) -> Tuple[pd.Series, List[str]]:
        measured = [s for s in member_sites if s in df_phospho.index]
        if not measured:
            return pd.Series(dtype=float), []
        sub = df_phospho.loc[measured].apply(pd.to_numeric, errors="coerce")
        z = zscore_rows(sub)
        score = z.mean(axis=0, skipna=True)
        # NaN when no members observed in that sample
        any_obs = sub.notna().any(axis=0)
        score = score.where(any_obs, np.nan)
        return score, measured

    # ------------------------------------------------------------------ Stage 1
    def _protein_pair_cache(
        self,
        cancer: str,
        ensembl: str,
        df_phospho: pd.DataFrame,
        site_to_pos: Dict[str, int],
        ensembl_to_sites: Dict[str, List[str]],
        cache: Dict[Tuple[str, str], List[Tuple[str, str, int, float, int]]],
    ) -> List[Tuple[str, str, int, float, int]]:
        """Cache all valid site-pairs on a protein: (s1,s2,dist,rho,n_overlap)."""
        key = (cancer, ensembl)
        if key in cache:
            return cache[key]
        sites = ensembl_to_sites.get(ensembl, [])
        pairs: List[Tuple[str, str, int, float, int]] = []
        # Keep local + far-distance pairs for matched / far backgrounds.
        max_gap = FAR_DIST_MAX
        for i in range(len(sites)):
            a = sites[i]
            da = site_to_pos.get(a)
            if da is None:
                continue
            for j in range(i + 1, len(sites)):
                b = sites[j]
                db = site_to_pos.get(b)
                if db is None:
                    continue
                d = abs(da - db)
                if d > max_gap:
                    continue
                rho, _, n_ov = spearman_pair(
                    df_phospho.loc[a], df_phospho.loc[b], self.min_overlap
                )
                if not np.isfinite(rho):
                    continue
                pairs.append((a, b, d, float(rho), int(n_ov)))
        cache[key] = pairs
        return pairs

    def run_stage1(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        pair_rows: List[Dict[str, object]] = []
        hotspot_rows: List[Dict[str, object]] = []
        bg_rows: List[Dict[str, object]] = []
        pair_cache: Dict[Tuple[str, str], List[Tuple[str, str, int, float, int]]] = {}

        # Only compute background for proteins that host analyzed hotspots
        hotspot_ensembls = set(
            self.members.dropna(subset=["ENSEMBL_GENE_ID"])["ENSEMBL_GENE_ID"]
            .astype(str)
            .tolist()
        )

        cancers = list(self.config.cancer_types)
        for cancer in cancers:
            print(f"[Stage1] {cancer}", flush=True)
            df_phospho = self.load_phospho(cancer)
            df_protein = self.load_protein(cancer)

            site_to_pos: Dict[str, int] = {}
            for idx in df_phospho.index.astype(str):
                if "|" not in idx:
                    continue
                try:
                    _, pos = parse_site_token(idx.split("|", 1)[1])
                    site_to_pos[idx] = pos
                except Exception:
                    continue

            ensembl_to_sites: Dict[str, List[str]] = {}
            for idx in site_to_pos:
                ens = idx.split("|", 1)[0]
                if ens in hotspot_ensembls:
                    ensembl_to_sites.setdefault(ens, []).append(idx)

            for hotspot_id, mem in self.members.groupby("hotspot_id", sort=False):
                mem = mem.dropna(subset=["ENSEMBL_GENE_ID", "POSITION"]).copy()
                mem = mem[mem["ENSEMBL_GENE_ID"].astype(str).str.startswith("ENSG")]
                if mem.empty:
                    continue
                hmeta = self.hotspot_meta.loc[hotspot_id]
                if isinstance(hmeta, pd.DataFrame):
                    hmeta = hmeta.iloc[0]
                gene = str(hmeta.get("gene_name", mem["gene_name"].iloc[0]))
                ensembl = str(mem["ENSEMBL_GENE_ID"].iloc[0])
                protein_acc = str(hmeta.get("protein_acc", mem["protein_acc"].iloc[0]))
                stage3_type = str(
                    hmeta.get("stage3_type", map_stage3_type(hmeta.get("hotspot_type")))
                )

                member_sites = list(
                    dict.fromkeys(
                        [s for s in mem["cptac_site"].astype(str).tolist() if s in df_phospho.index]
                    )
                )
                if len(member_sites) < 2:
                    continue

                pos_lookup = {
                    str(r.cptac_site): int(r.POSITION)
                    for r in mem.itertuples(index=False)
                    if pd.notna(r.POSITION)
                }

                sub = df_phospho.loc[member_sites].apply(pd.to_numeric, errors="coerce")
                resid = None
                if df_protein is not None and ensembl in df_protein.index:
                    prot = self.pipeline._gene_abundance_series(
                        ensembl, df_protein, list(df_phospho.columns)
                    )
                    resid = protein_residuals(sub, prot)

                within_rhos: List[float] = []
                within_adj: List[float] = []
                within_pairs_meta: List[Tuple[str, str, int, float]] = []

                for i in range(len(member_sites)):
                    for j in range(i + 1, len(member_sites)):
                        s1, s2 = member_sites[i], member_sites[j]
                        pos1 = pos_lookup.get(s1)
                        pos2 = pos_lookup.get(s2)
                        if pos1 is None or pos2 is None:
                            continue
                        dist = abs(pos1 - pos2)
                        rho, p, n_ov = spearman_pair(sub.loc[s1], sub.loc[s2], self.min_overlap)
                        adj_rho, adj_p, adj_n = (np.nan, np.nan, 0)
                        if resid is not None:
                            adj_rho, adj_p, adj_n = spearman_pair(
                                resid.loc[s1], resid.loc[s2], self.min_overlap
                            )
                        pair_rows.append(
                            {
                                "Cancer": cancer,
                                "Hotspot_ID": hotspot_id,
                                "Protein": protein_acc,
                                "Gene": gene,
                                "ENSEMBL_GENE_ID": ensembl,
                                "Hotspot_Type": stage3_type,
                                "Site_1": s1,
                                "Site_2": s2,
                                "Sequence_Distance": dist,
                                "N_Overlap": n_ov,
                                "Spearman_Rho": rho,
                                "P_Value": p,
                                "Adjusted_Rho": adj_rho,
                                "Adjusted_P_Value": adj_p,
                                "Adjusted_N_Overlap": adj_n,
                                "Background_Type": "within_hotspot",
                            }
                        )
                        if np.isfinite(rho):
                            within_rhos.append(float(rho))
                            within_pairs_meta.append((s1, s2, dist, float(rho)))
                        if np.isfinite(adj_rho):
                            within_adj.append(float(adj_rho))

                hotspot_set = set(member_sites)
                catalog_sites = self.catalog_cptac_sites
                all_pairs = self._protein_pair_cache(
                    cancer, ensembl, df_phospho, site_to_pos, ensembl_to_sites, pair_cache
                )

                def _is_catalog_free(a: str, b: str) -> bool:
                    """Neither site belongs to any catalog hotspot (stricter background)."""
                    return (a not in catalog_sites) and (b not in catalog_sites)

                def _is_non_query_hotspot(a: str, b: str) -> bool:
                    """Legacy: exclude only pairs fully inside the query hotspot."""
                    return not (a in hotspot_set and b in hotspot_set)

                matched_bg_for_pairs: List[float] = []
                matched_bg_rhos: List[float] = []
                legacy_bg_rhos: List[float] = []
                far_bg_rhos: List[float] = []

                for _, _, dist, _ in within_pairs_meta:
                    matched = []
                    for a, b, d, rho_bg, n_ov in all_pairs:
                        if not _is_catalog_free(a, b):
                            continue
                        if abs(d - dist) <= self.dist_tol:
                            matched.append(rho_bg)
                            bg_rows.append(
                                {
                                    "Cancer": cancer,
                                    "Hotspot_ID": hotspot_id,
                                    "Protein": protein_acc,
                                    "Gene": gene,
                                    "Site_1": a,
                                    "Site_2": b,
                                    "Sequence_Distance": d,
                                    "Query_Distance": dist,
                                    "N_Overlap": n_ov,
                                    "Spearman_Rho": rho_bg,
                                    "Background_Type": "distance_matched_exclude_catalog",
                                }
                            )
                    if matched:
                        matched_bg_for_pairs.append(float(np.median(matched)))
                        matched_bg_rhos.extend(matched)

                    # Legacy sensitivity: distance match; exclude only fully-within-query pairs
                    legacy_matched = []
                    for a, b, d, rho_bg, n_ov in all_pairs:
                        if a in hotspot_set and b in hotspot_set:
                            continue
                        if abs(d - dist) <= self.dist_tol:
                            legacy_matched.append(rho_bg)
                            bg_rows.append(
                                {
                                    "Cancer": cancer,
                                    "Hotspot_ID": hotspot_id,
                                    "Protein": protein_acc,
                                    "Gene": gene,
                                    "Site_1": a,
                                    "Site_2": b,
                                    "Sequence_Distance": d,
                                    "Query_Distance": dist,
                                    "N_Overlap": n_ov,
                                    "Spearman_Rho": rho_bg,
                                    "Background_Type": "distance_matched_legacy_non_query",
                                }
                            )
                    if legacy_matched:
                        legacy_bg_rhos.extend(legacy_matched)

                # Far-distance control: same protein, 40–100 aa, no catalog members
                for a, b, d, rho_bg, n_ov in all_pairs:
                    if not _is_catalog_free(a, b):
                        continue
                    if FAR_DIST_MIN <= d <= FAR_DIST_MAX:
                        far_bg_rhos.append(rho_bg)
                        bg_rows.append(
                            {
                                "Cancer": cancer,
                                "Hotspot_ID": hotspot_id,
                                "Protein": protein_acc,
                                "Gene": gene,
                                "Site_1": a,
                                "Site_2": b,
                                "Sequence_Distance": d,
                                "Query_Distance": np.nan,
                                "N_Overlap": n_ov,
                                "Spearman_Rho": rho_bg,
                                "Background_Type": "far_distance_exclude_catalog",
                            }
                        )

                # Fallback if stricter background empty: legacy distance-matched
                if not matched_bg_rhos and legacy_bg_rhos:
                    matched_bg_rhos = list(legacy_bg_rhos)
                    for _, _, dist, _ in within_pairs_meta:
                        # rebuild per-pair medians from legacy for Wilcoxon
                        pass
                    # Recompute matched_bg_for_pairs from legacy if needed
                    if not matched_bg_for_pairs:
                        for _, _, dist, _ in within_pairs_meta:
                            legacy_m = [
                                rho
                                for a, b, d, rho, _n in all_pairs
                                if _is_non_query_hotspot(a, b)
                                and not (a in hotspot_set and b in hotspot_set)
                                and abs(d - dist) <= self.dist_tol
                            ]
                            if legacy_m:
                                matched_bg_for_pairs.append(float(np.median(legacy_m)))

                bg_rhos_all = list(matched_bg_rhos)

                median_within = float(np.median(within_rhos)) if within_rhos else np.nan
                mean_within = float(np.mean(within_rhos)) if within_rhos else np.nan
                median_adj = float(np.median(within_adj)) if within_adj else np.nan
                median_bg = (
                    float(np.median(matched_bg_for_pairs))
                    if matched_bg_for_pairs
                    else (float(np.median(bg_rhos_all)) if bg_rhos_all else np.nan)
                )
                median_far = float(np.median(far_bg_rhos)) if far_bg_rhos else np.nan
                median_legacy = float(np.median(legacy_bg_rhos)) if legacy_bg_rhos else np.nan

                mw_p = np.nan
                mw_p_far = np.nan
                mw_p_legacy = np.nan
                w_p = np.nan
                if len(within_rhos) >= 3 and len(matched_bg_rhos) >= 3:
                    try:
                        mw_p = float(
                            mannwhitneyu(within_rhos, matched_bg_rhos, alternative="greater").pvalue
                        )
                    except ValueError:
                        mw_p = np.nan
                if len(within_rhos) >= 3 and len(far_bg_rhos) >= 3:
                    try:
                        mw_p_far = float(
                            mannwhitneyu(within_rhos, far_bg_rhos, alternative="greater").pvalue
                        )
                    except ValueError:
                        mw_p_far = np.nan
                if len(within_rhos) >= 3 and len(legacy_bg_rhos) >= 3:
                    try:
                        mw_p_legacy = float(
                            mannwhitneyu(within_rhos, legacy_bg_rhos, alternative="greater").pvalue
                        )
                    except ValueError:
                        mw_p_legacy = np.nan
                if len(within_rhos) >= 3 and len(matched_bg_for_pairs) == len(within_rhos):
                    try:
                        diff = np.asarray(within_rhos) - np.asarray(matched_bg_for_pairs)
                        if np.any(diff != 0):
                            w_p = float(wilcoxon(diff, alternative="greater").pvalue)
                    except ValueError:
                        w_p = np.nan

                hotspot_rows.append(
                    {
                        "Cancer": cancer,
                        "Hotspot_ID": hotspot_id,
                        "Protein": protein_acc,
                        "Gene": gene,
                        "Hotspot_Type": stage3_type,
                        "N_Members": int(len(member_sites)),
                        "N_Valid_Pairs": int(len(within_rhos)),
                        "Median_Rho": median_within,
                        "Mean_Rho": mean_within,
                        "Median_Adjusted_Rho": median_adj,
                        "Matched_Background_Rho": median_bg,
                        "Far_Background_Rho": median_far,
                        "Legacy_Background_Rho": median_legacy,
                        "Delta_Rho": (
                            median_within - median_bg
                            if np.isfinite(median_within) and np.isfinite(median_bg)
                            else np.nan
                        ),
                        "Delta_Rho_Far": (
                            median_within - median_far
                            if np.isfinite(median_within) and np.isfinite(median_far)
                            else np.nan
                        ),
                        "N_Matched_Background_Pairs": int(len(matched_bg_rhos)),
                        "N_Far_Background_Pairs": int(len(far_bg_rhos)),
                        "N_Legacy_Background_Pairs": int(len(legacy_bg_rhos)),
                        "MannWhitney_P_greater": mw_p,
                        "MannWhitney_P_greater_vs_far": mw_p_far,
                        "MannWhitney_P_greater_vs_legacy": mw_p_legacy,
                        "Wilcoxon_P_greater": w_p,
                        "Background_Definition": "distance_matched_exclude_catalog",
                    }
                )

        pair_df = pd.DataFrame(pair_rows)
        hotspot_df = pd.DataFrame(hotspot_rows)
        bg_df = pd.DataFrame(bg_rows)
        pair_df.to_csv(self.output_dir / "stage1" / "stage1_pair_level.csv", index=False)
        hotspot_df.to_csv(self.output_dir / "stage1" / "stage1_hotspot_level.csv", index=False)
        bg_df.to_csv(self.output_dir / "stage1" / "stage1_background_pairs.csv", index=False)
        return pair_df, hotspot_df, bg_df

    # ------------------------------------------------------------------ Stage 2/3
    def run_stage2_stage3(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        is_nuclear = nuclear_mask(self.hotspots, self.members)
        nuclear_ids = set(self.hotspots.loc[is_nuclear, "hotspot_id"].astype(str))
        print(f"[Stage2/3] nuclear hotspots: {len(nuclear_ids)}", flush=True)

        comparison_rows: List[Dict[str, object]] = []
        loso_rows: List[Dict[str, object]] = []
        site_beta_rows: List[Dict[str, object]] = []

        for cancer in self.config.cancer_types:
            print(f"[Stage2/3] {cancer}", flush=True)
            df_phospho = self.load_phospho(cancer)
            df_rna = self.load_rna(cancer)
            df_protein = self.load_protein(cancer)
            samples = [s for s in df_phospho.columns if s in df_rna.columns]
            n_total = len(samples)
            if n_total < self.min_score_samples:
                continue

            for hotspot_id in nuclear_ids:
                mem = self.members[self.members["hotspot_id"].astype(str).eq(hotspot_id)].copy()
                mem = mem.dropna(subset=["ENSEMBL_GENE_ID", "POSITION"])
                mem = mem[mem["ENSEMBL_GENE_ID"].astype(str).str.startswith("ENSG")]
                if mem.empty:
                    continue
                hmeta = self.hotspot_meta.loc[hotspot_id]
                if isinstance(hmeta, pd.DataFrame):
                    hmeta = hmeta.iloc[0]
                tf_name = str(hmeta.get("gene_name", mem["gene_name"].iloc[0]))
                stage3_type = str(hmeta.get("stage3_type", map_stage3_type(hmeta.get("hotspot_type"))))
                ensembl = str(mem["ENSEMBL_GENE_ID"].iloc[0])
                protein_acc = str(hmeta.get("protein_acc", mem["protein_acc"].iloc[0]))

                member_sites = list(dict.fromkeys(
                    [s for s in mem["cptac_site"].astype(str) if s in df_phospho.index]
                ))
                if not member_sites:
                    continue

                hs_score, measured = self.hotspot_score_mean_z(member_sites, df_phospho)
                if not measured:
                    continue
                hs_score = hs_score.reindex(samples)
                n_detected = int(hs_score.notna().sum())
                if n_detected < self.min_score_samples:
                    continue

                activity, n_targets = self.target_activity_series(tf_name, df_rna, samples)
                if n_targets < 1 or activity.notna().sum() < self.min_score_samples:
                    continue

                tf_prot = None
                if df_protein is not None:
                    tf_prot = self.pipeline._gene_abundance_series(ensembl, df_protein, samples)

                hs_beta, hs_p, hs_n = fit_ols_with_covariate(
                    self.pipeline, activity, hs_score, tf_prot
                )
                if not np.isfinite(hs_beta):
                    continue

                # Coverage
                sub = df_phospho.loc[measured, samples].apply(pd.to_numeric, errors="coerce")
                site_cov = sub.notna().sum(axis=1) / float(n_total)
                hotspot_cov = float(sub.notna().any(axis=0).sum() / float(n_total))
                median_site_cov = float(site_cov.median()) if len(site_cov) else np.nan
                best_site_cov = float(site_cov.max()) if len(site_cov) else np.nan

                # Per-site betas
                site_betas: List[float] = []
                site_expected = 0
                z_sites = zscore_rows(sub)
                for site in measured:
                    site_z = z_sites.loc[site].reindex(samples)
                    if site_z.notna().sum() < self.min_score_samples:
                        continue
                    b, p, n = fit_ols_with_covariate(self.pipeline, activity, site_z, tf_prot)
                    if not np.isfinite(b):
                        continue
                    site_betas.append(float(b))
                    if b > 0:
                        site_expected += 1
                    site_beta_rows.append(
                        {
                            "Cancer": cancer,
                            "Hotspot_ID": hotspot_id,
                            "TF": tf_name,
                            "Hotspot_Type": stage3_type,
                            "Site": site,
                            "Site_Beta": b,
                            "Site_P": p,
                            "N_Samples": n,
                            "Site_Coverage": float(site_cov.get(site, np.nan)),
                            "Expected_Direction": True if b > 0 else False,
                        }
                    )

                median_site_beta = float(np.median(site_betas)) if site_betas else np.nan
                best_site_beta = float(np.max(site_betas)) if site_betas else np.nan
                prop_site_expected = (
                    float(site_expected / len(site_betas)) if site_betas else np.nan
                )

                comparison_rows.append(
                    {
                        "Cancer": cancer,
                        "Hotspot_ID": hotspot_id,
                        "TF": tf_name,
                        "Protein": protein_acc,
                        "Direction": "Nuclear Import",
                        "Target_Regulation": "activate",
                        "Hotspot_Type": stage3_type,
                        "N_Members": int(len(member_sites)),
                        "N_Measured_Members": int(len(measured)),
                        "N_Target_Genes": int(n_targets),
                        "N_Score_Samples": int(n_detected),
                        "Hotspot_Coverage": hotspot_cov,
                        "Median_Site_Coverage": median_site_cov,
                        "Best_Site_Coverage": best_site_cov,
                        "Coverage_Gain": hotspot_cov - median_site_cov
                        if np.isfinite(median_site_cov)
                        else np.nan,
                        "Hotspot_Beta": hs_beta,
                        "Hotspot_P": hs_p,
                        "N_OLS_Samples": hs_n,
                        "Median_Site_Beta": median_site_beta,
                        "Best_Site_Beta": best_site_beta,
                        "Delta_Beta": hs_beta - median_site_beta
                        if np.isfinite(median_site_beta)
                        else np.nan,
                        "Hotspot_Expected_Direction": bool(hs_beta > 0),
                        "Site_Expected_Direction_Proportion": prop_site_expected,
                        "Signed_Beta": hs_beta,  # Import×Activate expected +
                        "Coverage": hotspot_cov,
                    }
                )

                # Leave-one-site-out
                if len(measured) >= 3:
                    for leave in measured:
                        remain = [s for s in measured if s != leave]
                        score_loso, _ = self.hotspot_score_mean_z(remain, df_phospho)
                        score_loso = score_loso.reindex(samples)
                        if score_loso.notna().sum() < self.min_score_samples:
                            continue
                        b, p, n = fit_ols_with_covariate(
                            self.pipeline, activity, score_loso, tf_prot
                        )
                        if not np.isfinite(b):
                            continue
                        loso_rows.append(
                            {
                                "Cancer": cancer,
                                "Hotspot_ID": hotspot_id,
                                "TF": tf_name,
                                "Hotspot_Type": stage3_type,
                                "Removed_Site": leave,
                                "N_Remaining_Sites": len(remain),
                                "Beta": b,
                                "P_Value": p,
                                "N_Samples": n,
                                "Direction_Correct": bool(b > 0),
                                "Full_Hotspot_Beta": hs_beta,
                            }
                        )

        comparison = pd.DataFrame(comparison_rows)
        loso = pd.DataFrame(loso_rows)
        site_betas_df = pd.DataFrame(site_beta_rows)

        if not comparison.empty:
            comparison["Hotspot_FDR"] = bh_fdr(comparison["Hotspot_P"].tolist())
            comparison["FDR"] = comparison["Hotspot_FDR"]
            comparison["Expected_Direction"] = comparison["Hotspot_Expected_Direction"]
            comparison["Direction_Correct"] = comparison["Signed_Beta"] > 0

        comparison.to_csv(self.output_dir / "stage2" / "stage2_comparison_table.csv", index=False)
        loso.to_csv(self.output_dir / "stage2" / "stage2_loso_table.csv", index=False)
        site_betas_df.to_csv(self.output_dir / "stage2" / "stage2_site_betas.csv", index=False)

        # Stage 3 is the same association rows, organized by class
        stage3 = comparison.copy()
        if not stage3.empty:
            stage3 = stage3[
                [
                    "Cancer",
                    "Hotspot_ID",
                    "TF",
                    "Hotspot_Type",
                    "N_Members",
                    "Coverage",
                    "N_Target_Genes",
                    "Signed_Beta",
                    "Hotspot_P",
                    "FDR",
                    "Expected_Direction",
                    "Direction_Correct",
                    "N_Score_Samples",
                    "N_OLS_Samples",
                    "N_Measured_Members",
                ]
            ].rename(columns={"Hotspot_P": "P_Value"})
        stage3.to_csv(self.output_dir / "stage3" / "stage3_category_table.csv", index=False)
        return comparison, loso, stage3

    # ------------------------------------------------------------------ Funnel + relaxed Stage3
    def run_dropout_funnel(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Attrition funnel for Stage2/3 gates, with emphasis on Predicted_only."""
        is_nuclear = nuclear_mask(self.hotspots, self.members)
        nuclear_ids = set(self.hotspots.loc[is_nuclear, "hotspot_id"].astype(str))
        rows: List[Dict[str, object]] = []

        for cancer in self.config.cancer_types:
            print(f"[Funnel] {cancer}", flush=True)
            df_phospho = self.load_phospho(cancer)
            df_rna = self.load_rna(cancer)
            df_protein = self.load_protein(cancer)
            samples = [s for s in df_phospho.columns if s in df_rna.columns]
            n_total = len(samples)

            for _, hrow in self.hotspots.iterrows():
                hotspot_id = str(hrow["hotspot_id"])
                stage3_type = str(hrow.get("stage3_type", map_stage3_type(hrow.get("hotspot_type"))))
                mem = self.members[self.members["hotspot_id"].astype(str).eq(hotspot_id)].copy()
                tf_name = str(hrow.get("gene_name", ""))
                if not tf_name and not mem.empty:
                    tf_name = str(mem["gene_name"].iloc[0])

                gate_catalog = True
                gate_nuclear = hotspot_id in nuclear_ids
                mem_ens = mem.dropna(subset=["ENSEMBL_GENE_ID"]).copy()
                mem_ens = mem_ens[mem_ens["ENSEMBL_GENE_ID"].astype(str).str.startswith("ENSG")]
                gate_ensembl = not mem_ens.empty

                member_sites: List[str] = []
                if gate_ensembl:
                    member_sites = list(
                        dict.fromkeys(
                            [
                                s
                                for s in mem_ens["cptac_site"].astype(str).tolist()
                                if s in df_phospho.index
                            ]
                        )
                    )
                gate_cptac_present = len(member_sites) >= 1
                gate_cptac_pair = len(member_sites) >= 2

                n_detected = 0
                n_targets = 0
                gate_score = False
                gate_activate = False
                gate_ols = False
                signed_beta = np.nan
                drop_reason = "pass"

                if not gate_nuclear:
                    drop_reason = "not_nuclear"
                elif not gate_ensembl:
                    drop_reason = "no_ensembl_mapping"
                elif n_total < self.min_score_samples:
                    drop_reason = "cancer_too_few_samples"
                elif not gate_cptac_present:
                    drop_reason = "no_cptac_member_in_matrix"
                else:
                    hs_score, measured = self.hotspot_score_mean_z(member_sites, df_phospho)
                    hs_score = hs_score.reindex(samples)
                    n_detected = int(hs_score.notna().sum())
                    gate_score = n_detected >= self.min_score_samples
                    if not gate_score:
                        drop_reason = f"score_samples_lt_{self.min_score_samples}"
                    else:
                        activity, n_targets = self.target_activity_series(tf_name, df_rna, samples)
                        gate_activate = n_targets >= 1 and int(activity.notna().sum()) >= self.min_score_samples
                        if not gate_activate:
                            drop_reason = "no_activate_targets_or_sparse"
                        else:
                            tf_prot = None
                            if df_protein is not None and gate_ensembl:
                                ensembl = str(mem_ens["ENSEMBL_GENE_ID"].iloc[0])
                                tf_prot = self.pipeline._gene_abundance_series(
                                    ensembl, df_protein, samples
                                )
                            hs_beta, _, _ = fit_ols_with_covariate(
                                self.pipeline, activity, hs_score, tf_prot
                            )
                            gate_ols = bool(np.isfinite(hs_beta))
                            if gate_ols:
                                signed_beta = float(hs_beta)
                                drop_reason = "eligible_stage2_3"
                            else:
                                drop_reason = "ols_failed"

                rows.append(
                    {
                        "Cancer": cancer,
                        "Hotspot_ID": hotspot_id,
                        "TF": tf_name,
                        "Hotspot_Type": stage3_type,
                        "gate_catalog": gate_catalog,
                        "gate_nuclear": gate_nuclear,
                        "gate_ensembl": gate_ensembl,
                        "gate_cptac_member_present": gate_cptac_present,
                        "gate_cptac_ge2_members": gate_cptac_pair,
                        "N_Members_In_Matrix": int(len(member_sites)),
                        "N_Score_Samples": int(n_detected),
                        "gate_score_ge_min": gate_score,
                        "N_Activate_Targets": int(n_targets),
                        "gate_activate_ok": gate_activate,
                        "gate_ols_ok": gate_ols,
                        "Signed_Beta": signed_beta,
                        "drop_reason": drop_reason,
                        "measurable_subset": bool(
                            gate_nuclear and gate_ensembl and gate_cptac_present
                        ),
                        "stage3_eligible": bool(gate_ols),
                    }
                )

        detail = pd.DataFrame(rows)
        detail.to_csv(self.output_dir / "funnel" / "cancer_hotspot_gate_detail.csv", index=False)

        # Hotspot-level: nested ever-pass across cancers (nuclear → measurable → …)
        hs_rows: List[Dict[str, object]] = []
        for hotspot_id, g in detail.groupby("Hotspot_ID", sort=False):
            ever_nuclear = bool(g["gate_nuclear"].any())
            ever_measurable = bool(g["measurable_subset"].any())  # already requires nuclear
            ever_score = bool((g["gate_nuclear"] & g["gate_score_ge_min"]).any())
            ever_activate = bool((g["gate_nuclear"] & g["gate_activate_ok"]).any())
            ever_eligible = bool(g["stage3_eligible"].any())
            hs_rows.append(
                {
                    "Hotspot_ID": hotspot_id,
                    "TF": g["TF"].iloc[0],
                    "Hotspot_Type": g["Hotspot_Type"].iloc[0],
                    "n_cancers": int(len(g)),
                    "ever_nuclear": ever_nuclear,
                    "ever_cptac_present": bool(g["gate_cptac_member_present"].any()),
                    "ever_measurable_nested": ever_measurable,
                    "ever_score_ok": ever_score,
                    "ever_activate_ok": ever_activate,
                    "ever_stage3_eligible": ever_eligible,
                    "n_cancers_measurable": int(g["measurable_subset"].sum()),
                    "n_cancers_stage3_eligible": int(g["stage3_eligible"].sum()),
                    "median_signed_beta_eligible": float(
                        g.loc[g["stage3_eligible"], "Signed_Beta"].median()
                    )
                    if g["stage3_eligible"].any()
                    else np.nan,
                }
            )
        hotspot_summary = pd.DataFrame(hs_rows)
        hotspot_summary.to_csv(
            self.output_dir / "funnel" / "hotspot_level_funnel_summary.csv", index=False
        )

        # Type-level counts for report
        summary_rows: List[Dict[str, object]] = []
        for htype, g in hotspot_summary.groupby("Hotspot_Type", sort=False):
            summary_rows.append(
                {
                    "Hotspot_Type": htype,
                    "n_catalog": int(len(g)),
                    "n_nuclear": int(g["ever_nuclear"].sum()),
                    "n_measurable_nested": int(g["ever_measurable_nested"].sum()),
                    "n_score_ok_nested": int(g["ever_score_ok"].sum()),
                    "n_activate_ok_nested": int(g["ever_activate_ok"].sum()),
                    "n_stage3_eligible_any_cancer": int(g["ever_stage3_eligible"].sum()),
                }
            )
        # Cancer×Hotspot eligible counts by type
        cxh = (
            detail.groupby("Hotspot_Type")
            .agg(
                n_cancer_hotspot_rows=("Hotspot_ID", "size"),
                n_measurable_cxh=("measurable_subset", "sum"),
                n_stage3_eligible_cxh=("stage3_eligible", "sum"),
                pct_signed_beta_gt0_eligible=(
                    "Signed_Beta",
                    lambda s: float((s.dropna() > 0).mean()) if s.notna().any() else np.nan,
                ),
            )
            .reset_index()
        )
        type_summary = pd.DataFrame(summary_rows).merge(cxh, on="Hotspot_Type", how="left")
        type_summary.to_csv(self.output_dir / "funnel" / "type_level_funnel_counts.csv", index=False)

        # Predicted_only focused nested step table
        pred = hotspot_summary[hotspot_summary["Hotspot_Type"].eq("Predicted_only")]
        pred_steps = [
            ("catalog", int(len(pred))),
            ("nuclear", int(pred["ever_nuclear"].sum()) if not pred.empty else 0),
            (
                "nuclear_and_measurable_CPTAC",
                int(pred["ever_measurable_nested"].sum()) if not pred.empty else 0,
            ),
            (
                "nuclear_and_score_ge_min",
                int(pred["ever_score_ok"].sum()) if not pred.empty else 0,
            ),
            (
                "nuclear_and_activate_ok",
                int(pred["ever_activate_ok"].sum()) if not pred.empty else 0,
            ),
            (
                "stage3_eligible_any_cancer",
                int(pred["ever_stage3_eligible"].sum()) if not pred.empty else 0,
            ),
        ]
        pred_funnel = pd.DataFrame(pred_steps, columns=["step", "n_hotspots"])
        pred_detail = detail[detail["Hotspot_Type"].eq("Predicted_only")]
        if not pred_detail.empty:
            pred_cxh_steps = [
                ("cancer_x_hotspot_rows", int(len(pred_detail))),
                ("nuclear", int(pred_detail["gate_nuclear"].sum())),
                ("measurable", int(pred_detail["measurable_subset"].sum())),
                ("score_ok", int(pred_detail["gate_score_ge_min"].sum())),
                ("activate_ok", int(pred_detail["gate_activate_ok"].sum())),
                ("stage3_eligible", int(pred_detail["stage3_eligible"].sum())),
            ]
            pred_funnel_cxh = pd.DataFrame(pred_cxh_steps, columns=["step", "n_cancer_x_hotspot"])
        else:
            pred_funnel_cxh = pd.DataFrame(columns=["step", "n_cancer_x_hotspot"])
        pred_funnel.to_csv(self.output_dir / "funnel" / "predicted_only_hotspot_funnel.csv", index=False)
        pred_funnel_cxh.to_csv(
            self.output_dir / "funnel" / "predicted_only_cancer_x_hotspot_funnel.csv", index=False
        )

        # Drop reason counts for Predicted_only
        if not pred_detail.empty:
            reason = (
                pred_detail["drop_reason"]
                .value_counts()
                .rename_axis("drop_reason")
                .reset_index(name="n")
            )
            reason.to_csv(
                self.output_dir / "funnel" / "predicted_only_drop_reasons.csv", index=False
            )

        return detail, type_summary

    def run_stage3_measurable_relaxed(self, min_score_samples: int = MIN_SCORE_SAMPLES_RELAXED) -> pd.DataFrame:
        """Re-run Import×Activate association on nuclear hotspots with relaxed score gate.

        Denominator focus: measurable nuclear Predicted_only (and all types for comparison).
        Does not replace the primary Stage2/3 tables.
        """
        is_nuclear = nuclear_mask(self.hotspots, self.members)
        nuclear_ids = set(self.hotspots.loc[is_nuclear, "hotspot_id"].astype(str))
        rows: List[Dict[str, object]] = []
        old_min = self.min_score_samples
        self.min_score_samples = int(min_score_samples)
        try:
            for cancer in self.config.cancer_types:
                print(f"[Stage3-relaxed/{min_score_samples}] {cancer}", flush=True)
                df_phospho = self.load_phospho(cancer)
                df_rna = self.load_rna(cancer)
                df_protein = self.load_protein(cancer)
                samples = [s for s in df_phospho.columns if s in df_rna.columns]
                if len(samples) < min_score_samples:
                    continue
                for hotspot_id in nuclear_ids:
                    mem = self.members[self.members["hotspot_id"].astype(str).eq(hotspot_id)].copy()
                    mem = mem.dropna(subset=["ENSEMBL_GENE_ID", "POSITION"])
                    mem = mem[mem["ENSEMBL_GENE_ID"].astype(str).str.startswith("ENSG")]
                    if mem.empty:
                        continue
                    hmeta = self.hotspot_meta.loc[hotspot_id]
                    if isinstance(hmeta, pd.DataFrame):
                        hmeta = hmeta.iloc[0]
                    tf_name = str(hmeta.get("gene_name", mem["gene_name"].iloc[0]))
                    stage3_type = str(
                        hmeta.get("stage3_type", map_stage3_type(hmeta.get("hotspot_type")))
                    )
                    ensembl = str(mem["ENSEMBL_GENE_ID"].iloc[0])
                    member_sites = list(
                        dict.fromkeys(
                            [s for s in mem["cptac_site"].astype(str) if s in df_phospho.index]
                        )
                    )
                    if not member_sites:
                        continue
                    hs_score, measured = self.hotspot_score_mean_z(member_sites, df_phospho)
                    hs_score = hs_score.reindex(samples)
                    if int(hs_score.notna().sum()) < min_score_samples:
                        continue
                    activity, n_targets = self.target_activity_series(tf_name, df_rna, samples)
                    if n_targets < 1 or int(activity.notna().sum()) < min_score_samples:
                        continue
                    tf_prot = None
                    if df_protein is not None:
                        tf_prot = self.pipeline._gene_abundance_series(ensembl, df_protein, samples)
                    hs_beta, hs_p, hs_n = fit_ols_with_covariate(
                        self.pipeline, activity, hs_score, tf_prot
                    )
                    if not np.isfinite(hs_beta):
                        continue
                    rows.append(
                        {
                            "Cancer": cancer,
                            "Hotspot_ID": hotspot_id,
                            "TF": tf_name,
                            "Hotspot_Type": stage3_type,
                            "N_Measured_Members": int(len(measured)),
                            "N_Target_Genes": int(n_targets),
                            "N_Score_Samples": int(hs_score.notna().sum()),
                            "N_OLS_Samples": int(hs_n),
                            "Signed_Beta": float(hs_beta),
                            "P_Value": float(hs_p) if np.isfinite(hs_p) else np.nan,
                            "Direction_Correct": bool(hs_beta > 0),
                            "min_score_samples": int(min_score_samples),
                            "subset": "nuclear_measurable_relaxed",
                        }
                    )
        finally:
            self.min_score_samples = old_min

        out = pd.DataFrame(rows)
        if not out.empty:
            out["FDR"] = bh_fdr(out["P_Value"].tolist())
        out.to_csv(
            self.output_dir
            / "stage3"
            / f"stage3_measurable_relaxed_min{min_score_samples}.csv",
            index=False,
        )
        # Class summary
        summary_rows = []
        for htype, g in out.groupby("Hotspot_Type") if not out.empty else []:
            n = len(g)
            n_pos = int((g["Signed_Beta"] > 0).sum())
            try:
                binom_p = float(stats.binomtest(n_pos, n, 0.5, alternative="greater").pvalue)
            except Exception:
                binom_p = float(stats.binom_test(n_pos, n, 0.5, alternative="greater"))
            summary_rows.append(
                {
                    "Hotspot_Type": htype,
                    "n": n,
                    "median_Signed_Beta": float(g["Signed_Beta"].median()),
                    "pct_Signed_Beta_gt0": float(n_pos / n) if n else np.nan,
                    "binom_p_greater_0.5": binom_p,
                    "min_score_samples": int(min_score_samples),
                }
            )
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(
            self.output_dir
            / "stage3"
            / f"stage3_measurable_relaxed_min{min_score_samples}_by_type.csv",
            index=False,
        )
        return out

    def _pick_figure1b_cancer(self, hotspot_id: str, hotspot_df: pd.DataFrame) -> Optional[str]:
        """Pick cancer with most measured members, then most valid pairs, then median rho."""
        sub = hotspot_df[hotspot_df["Hotspot_ID"].astype(str).eq(hotspot_id)].copy()
        if sub.empty:
            return None
        sub = sub[sub["N_Members"] >= 2]
        if sub.empty:
            return None
        sub = sub.sort_values(
            by=["N_Members", "N_Valid_Pairs", "Median_Rho"],
            ascending=[False, False, False],
            na_position="last",
        )
        return str(sub.iloc[0]["Cancer"])

    def _pick_figure1b_cancers(
        self, hotspot_id: str, hotspot_df: pd.DataFrame, max_cancers: int = 2
    ) -> List[str]:
        """Coverage-first cancer, plus optional high-coherence cancer if different."""
        sub = hotspot_df[hotspot_df["Hotspot_ID"].astype(str).eq(hotspot_id)].copy()
        sub = sub[sub["N_Members"] >= 2]
        if sub.empty:
            return []
        # Coverage panel
        cov = sub.sort_values(
            by=["N_Members", "N_Valid_Pairs", "Median_Rho"],
            ascending=[False, False, False],
            na_position="last",
        )
        picked = [str(cov.iloc[0]["Cancer"])]
        # Clear co-phosphorylation panel (spec): highest Median_Rho among usable rows
        coh = sub.dropna(subset=["Median_Rho"]).sort_values(
            by=["Median_Rho", "N_Members", "N_Valid_Pairs"],
            ascending=[False, False, False],
        )
        if not coh.empty:
            c2 = str(coh.iloc[0]["Cancer"])
            med = float(coh.iloc[0]["Median_Rho"])
            if c2 not in picked and med >= 0.2 and max_cancers >= 2:
                picked.append(c2)
        return picked[:max_cancers]

    def plot_figure1b(
        self,
        hotspot_df: pd.DataFrame,
        hotspot_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, object]]:
        """Representative within-hotspot Spearman correlation heatmaps (Figure 1B).

        Selection is by coverage / n_sites / coherence availability — not Import/Export.
        """
        fig_dir = self.output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        ids = list(hotspot_ids) if hotspot_ids else list(DEFAULT_FIGURE1B_HOTSPOT_IDS)
        meta_out: List[Dict[str, object]] = []
        panels: List[Tuple[str, str, pd.DataFrame, List[str], List[str], float]] = []

        for hotspot_id in ids:
            if hotspot_id not in set(self.hotspots["hotspot_id"].astype(str)):
                print(f"[Figure1B] skip missing hotspot_id={hotspot_id}", flush=True)
                continue
            cancers = self._pick_figure1b_cancers(hotspot_id, hotspot_df)
            if not cancers:
                print(f"[Figure1B] no eligible cancer for {hotspot_id}", flush=True)
                continue

            mem = self.members[self.members["hotspot_id"].astype(str).eq(hotspot_id)].copy()
            mem = mem.dropna(subset=["POSITION"]).sort_values("POSITION")
            hmeta = self.hotspot_meta.loc[hotspot_id]
            if isinstance(hmeta, pd.DataFrame):
                hmeta = hmeta.iloc[0]
            label = str(hmeta.get("hotspot_label", hotspot_id))
            htype = str(hmeta.get("stage3_type", ""))

            for cancer in cancers:
                df_phospho = self.load_phospho(cancer)
                site_tokens = []
                tick_labels = []
                evidence_flags = []
                for r in mem.itertuples(index=False):
                    site = str(r.site)
                    cptac = str(getattr(r, "cptac_site", ""))
                    if cptac not in df_phospho.index:
                        continue
                    site_tokens.append(cptac)
                    ev = str(getattr(r, "evidence", ""))
                    tag = (
                        "K"
                        if ev.lower().startswith("known")
                        else ("P" if "predict" in ev.lower() else "?")
                    )
                    tick_labels.append(f"{site} ({tag})")
                    evidence_flags.append(ev)

                if len(site_tokens) < 2:
                    print(
                        f"[Figure1B] {label} {cancer}: <2 measured members",
                        flush=True,
                    )
                    continue

                sub = df_phospho.loc[site_tokens].apply(pd.to_numeric, errors="coerce")
                n = len(site_tokens)
                rho_mat = np.full((n, n), np.nan, dtype=float)
                n_mat = np.zeros((n, n), dtype=int)
                for i in range(n):
                    rho_mat[i, i] = 1.0
                    n_mat[i, i] = int(sub.iloc[i].notna().sum())
                    for j in range(i + 1, n):
                        rho, _, n_ov = spearman_pair(
                            sub.iloc[i], sub.iloc[j], self.min_overlap
                        )
                        n_mat[i, j] = n_ov
                        n_mat[j, i] = n_ov
                        if np.isfinite(rho):
                            rho_mat[i, j] = rho
                            rho_mat[j, i] = rho

                off = rho_mat[np.triu_indices(n, k=1)]
                off = off[np.isfinite(off)]
                med_rho = float(np.median(off)) if len(off) else np.nan
                rho_df = pd.DataFrame(rho_mat, index=tick_labels, columns=tick_labels)
                panels.append((label, cancer, rho_df, tick_labels, evidence_flags, med_rho))

                fig, ax = plt.subplots(figsize=(4.8 + 0.35 * n, 4.2 + 0.3 * n))
                sns.heatmap(
                    rho_df,
                    vmin=-1,
                    vmax=1,
                    center=0,
                    cmap="RdBu_r",
                    annot=True,
                    fmt=".2f",
                    square=True,
                    ax=ax,
                    cbar_kws={"label": "Spearman ρ", "shrink": 0.8},
                    mask=~np.isfinite(rho_mat),
                    linewidths=0.4,
                    linecolor="white",
                )
                title = (
                    f"1B {label} · {cancer}\n"
                    f"{htype} · n_sites={n} · median off-diag ρ={med_rho:.3f}"
                    if np.isfinite(med_rho)
                    else f"1B {label} · {cancer}\n{htype} · n_sites={n}"
                )
                ax.set_title(title, fontsize=11)
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.text(
                    0.0,
                    -0.12,
                    "Tick: site (K=Known, P=Predicted); blank cells: overlap < "
                    f"{self.min_overlap}",
                    transform=ax.transAxes,
                    fontsize=8,
                    color="#555555",
                )
                fig.tight_layout()
                safe = label.replace("/", "-").replace(" ", "_")
                out_stem = fig_dir / f"figure1B_{safe}_{cancer}_correlation_heatmap"
                fig.savefig(f"{out_stem}.png", dpi=200)
                fig.savefig(f"{out_stem}.pdf")
                plt.close(fig)

                rho_df.to_csv(
                    self.output_dir / "stage1" / f"figure1B_{safe}_{cancer}_rho_matrix.csv"
                )
                n_df = pd.DataFrame(n_mat, index=tick_labels, columns=tick_labels)
                n_df.to_csv(
                    self.output_dir
                    / "stage1"
                    / f"figure1B_{safe}_{cancer}_n_overlap_matrix.csv"
                )

                meta_out.append(
                    {
                        "hotspot_id": hotspot_id,
                        "hotspot_label": label,
                        "hotspot_type": htype,
                        "cancer": cancer,
                        "n_sites_measured": n,
                        "median_offdiag_rho": med_rho,
                        "n_valid_offdiag_pairs": int(len(off)),
                        "figure": f"figure1B_{safe}_{cancer}_correlation_heatmap.png",
                    }
                )
                print(
                    f"[Figure1B] {label} {cancer}: n={n}, median_rho={med_rho:.3f}",
                    flush=True,
                )

        # Combined panel (up to 3)
        if len(panels) >= 1:
            use = panels[:3]
            ncols = len(use)
            fig, axes = plt.subplots(1, ncols, figsize=(4.4 * ncols + 1.2, 4.6))
            if ncols == 1:
                axes = [axes]
            for ax, (label, cancer, rho_df, _ticks, _ev, med_rho) in zip(axes, use):
                sns.heatmap(
                    rho_df,
                    vmin=-1,
                    vmax=1,
                    center=0,
                    cmap="RdBu_r",
                    annot=True,
                    fmt=".2f",
                    square=True,
                    ax=ax,
                    cbar=ax is axes[-1],
                    cbar_kws={"label": "Spearman ρ", "shrink": 0.75}
                    if ax is axes[-1]
                    else None,
                    mask=~np.isfinite(rho_df.to_numpy(dtype=float)),
                    linewidths=0.4,
                    linecolor="white",
                )
                ax.set_title(
                    f"{label}\n{cancer} · median ρ={med_rho:.2f}"
                    if np.isfinite(med_rho)
                    else f"{label}\n{cancer}",
                    fontsize=10,
                )
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.tick_params(axis="both", labelsize=8)
            fig.suptitle(
                "Figure 1B · Representative within-hotspot co-phosphorylation",
                fontsize=12,
                y=1.02,
            )
            fig.tight_layout()
            fig.savefig(
                fig_dir / "figure1B_representative_correlation_heatmaps.png",
                dpi=200,
                bbox_inches="tight",
            )
            fig.savefig(
                fig_dir / "figure1B_representative_correlation_heatmaps.pdf",
                bbox_inches="tight",
            )
            plt.close(fig)

        return meta_out

    # ------------------------------------------------------------------ Figures
    def plot_figures(
        self,
        pair_df: pd.DataFrame,
        hotspot_df: pd.DataFrame,
        comparison: pd.DataFrame,
        stage3: pd.DataFrame,
    ) -> Dict[str, object]:
        fig_dir = self.output_dir / "figures"
        stats_out: Dict[str, object] = {}
        sns.set_style("whitegrid")

        # Figure 1A — primary: within vs distance-matched exclude-catalog
        within = pair_df[pair_df["Background_Type"].eq("within_hotspot")]["Spearman_Rho"].dropna()
        bg_path = self.output_dir / "stage1" / "stage1_background_pairs.csv"
        bg_all = pd.read_csv(bg_path) if bg_path.exists() else pd.DataFrame()
        primary_bg_name = "distance_matched_exclude_catalog"
        bg_primary = pd.DataFrame()
        if not bg_all.empty:
            bg_primary = bg_all[bg_all["Background_Type"].eq(primary_bg_name)]
            if bg_primary.empty:
                bg_primary = bg_all[bg_all["Background_Type"].eq("distance_matched_legacy_non_query")]
            if bg_primary.empty:
                bg_primary = bg_all[
                    bg_all["Background_Type"].isin(
                        ["distance_matched_same_protein", "same_protein_non_hotspot_fallback"]
                    )
                ]
        bg_rhos = (
            bg_primary["Spearman_Rho"].dropna().tolist() if not bg_primary.empty else []
        )
        bg_label = (
            str(bg_primary["Background_Type"].iloc[0]) if not bg_primary.empty else "background"
        )

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        plot_df = pd.DataFrame(
            {
                "rho": list(within) + list(bg_rhos),
                "group": (["Within hotspot"] * len(within))
                + ([f"BG: exclude catalog"] * len(bg_rhos)),
            }
        )
        if not plot_df.empty:
            sns.violinplot(
                data=plot_df, x="group", y="rho", inner=None, cut=0, ax=ax, color="#9ecae1"
            )
            sns.boxplot(
                data=plot_df,
                x="group",
                y="rho",
                width=0.25,
                showfliers=False,
                ax=ax,
                color="white",
            )
            sns.stripplot(
                data=plot_df.sample(min(len(plot_df), 4000), random_state=0)
                if len(plot_df) > 4000
                else plot_df,
                x="group",
                y="rho",
                size=2,
                alpha=0.25,
                ax=ax,
                color="#333333",
            )
            med_w = float(np.median(within)) if len(within) else np.nan
            med_b = float(np.median(bg_rhos)) if bg_rhos else np.nan
            p_mw = np.nan
            if len(within) >= 3 and len(bg_rhos) >= 3:
                try:
                    p_mw = float(mannwhitneyu(within, bg_rhos, alternative="greater").pvalue)
                except ValueError:
                    p_mw = np.nan
            stats_out["stage1_median_within_rho"] = med_w
            stats_out["stage1_median_background_rho"] = med_b
            stats_out["stage1_mw_p_greater"] = p_mw
            stats_out["stage1_n_within_pairs"] = int(len(within))
            stats_out["stage1_n_background_pairs"] = int(len(bg_rhos))
            stats_out["stage1_background_definition"] = bg_label
            ax.axhline(0, color="grey", lw=0.8, ls="--")
            ax.set_ylabel("Spearman rho")
            ax.set_xlabel("")
            ax.set_title(
                f"1A Within vs exclude-catalog BG\n"
                f"median within={med_w:.3f}, bg={med_b:.3f}, MW p={p_mw:.2e}\n"
                f"N_within={len(within)}, N_bg={len(bg_rhos)}"
            )
        fig.tight_layout()
        fig.savefig(fig_dir / "figure1A_within_vs_background_rho.png", dpi=200)
        fig.savefig(fig_dir / "figure1A_within_vs_background_rho.pdf")
        plt.close(fig)

        # Figure 1A far-distance control
        far_rhos: List[float] = []
        if not bg_all.empty:
            far_rhos = (
                bg_all[bg_all["Background_Type"].eq("far_distance_exclude_catalog")][
                    "Spearman_Rho"
                ]
                .dropna()
                .tolist()
            )
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        plot_far = pd.DataFrame(
            {
                "rho": list(within) + list(far_rhos),
                "group": (["Within hotspot"] * len(within))
                + ([f"Far BG ({FAR_DIST_MIN}-{FAR_DIST_MAX}aa)"] * len(far_rhos)),
            }
        )
        if not plot_far.empty and far_rhos:
            sns.violinplot(
                data=plot_far, x="group", y="rho", inner=None, cut=0, ax=ax, color="#a1d99b"
            )
            sns.boxplot(
                data=plot_far,
                x="group",
                y="rho",
                width=0.25,
                showfliers=False,
                ax=ax,
                color="white",
            )
            med_w = float(np.median(within)) if len(within) else np.nan
            med_f = float(np.median(far_rhos))
            p_far = np.nan
            if len(within) >= 3 and len(far_rhos) >= 3:
                try:
                    p_far = float(mannwhitneyu(within, far_rhos, alternative="greater").pvalue)
                except ValueError:
                    p_far = np.nan
            stats_out["stage1_median_far_background_rho"] = med_f
            stats_out["stage1_mw_p_greater_vs_far"] = p_far
            stats_out["stage1_n_far_background_pairs"] = int(len(far_rhos))
            ax.axhline(0, color="grey", lw=0.8, ls="--")
            ax.set_ylabel("Spearman rho")
            ax.set_xlabel("")
            ax.set_title(
                f"1A′ Within vs far-distance BG\n"
                f"median within={med_w:.3f}, far={med_f:.3f}, MW p={p_far:.2e}\n"
                f"N_within={len(within)}, N_far={len(far_rhos)}"
            )
        fig.tight_layout()
        fig.savefig(fig_dir / "figure1A_within_vs_far_background_rho.png", dpi=200)
        fig.savefig(fig_dir / "figure1A_within_vs_far_background_rho.pdf")
        plt.close(fig)

        # Legacy background comparison (sensitivity)
        legacy_rhos: List[float] = []
        if not bg_all.empty:
            legacy_rhos = (
                bg_all[bg_all["Background_Type"].eq("distance_matched_legacy_non_query")][
                    "Spearman_Rho"
                ]
                .dropna()
                .tolist()
            )
        if legacy_rhos:
            med_leg = float(np.median(legacy_rhos))
            p_leg = np.nan
            if len(within) >= 3 and len(legacy_rhos) >= 3:
                try:
                    p_leg = float(mannwhitneyu(within, legacy_rhos, alternative="greater").pvalue)
                except ValueError:
                    p_leg = np.nan
            stats_out["stage1_median_legacy_background_rho"] = med_leg
            stats_out["stage1_mw_p_greater_vs_legacy"] = p_leg
            stats_out["stage1_n_legacy_background_pairs"] = int(len(legacy_rhos))

        # Figure 1B representative correlation heatmaps
        fig1b_meta = self.plot_figure1b(hotspot_df)
        if fig1b_meta:
            stats_out["stage1_figure1b"] = fig1b_meta

        # Figure 1C coherence by type
        fig, ax = plt.subplots(figsize=(6, 4.5))
        if not hotspot_df.empty:
            hdf = hotspot_df.dropna(subset=["Median_Rho"]).copy()
            order = ["Known_only", "Mixed", "Predicted_only"]
            present = [t for t in order if t in set(hdf["Hotspot_Type"])]
            sns.violinplot(
                data=hdf,
                x="Hotspot_Type",
                y="Median_Rho",
                order=present,
                inner="box",
                cut=0,
                ax=ax,
            )
            sns.stripplot(
                data=hdf,
                x="Hotspot_Type",
                y="Median_Rho",
                order=present,
                size=2.5,
                alpha=0.35,
                color="black",
                ax=ax,
            )
            ax.axhline(0, color="grey", lw=0.8, ls="--")
            ax.set_ylabel("HotspotCoherence (median rho)")
            ax.set_title("1C Coherence by hotspot type")
            stats_out["stage1_coherence_by_type"] = (
                hdf.groupby("Hotspot_Type")["Median_Rho"].median().to_dict()
            )
        fig.tight_layout()
        fig.savefig(fig_dir / "figure1C_coherence_by_hotspot_type.png", dpi=200)
        fig.savefig(fig_dir / "figure1C_coherence_by_hotspot_type.pdf")
        plt.close(fig)

        # Figure 2A coverage
        fig, ax = plt.subplots(figsize=(5, 5))
        if not comparison.empty:
            x = comparison["Median_Site_Coverage"]
            y = comparison["Hotspot_Coverage"]
            ax.scatter(x, y, s=22, alpha=0.7, c="#2c7fb8", edgecolors="none")
            lims = [0, 1]
            ax.plot(lims, lims, ls="--", color="grey", lw=1)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_xlabel("Median single-site coverage")
            ax.set_ylabel("Hotspot coverage")
            gain = comparison["Coverage_Gain"].dropna()
            above = float((y > x).mean()) if len(comparison) else np.nan
            stats_out["stage2_pct_above_diagonal"] = above
            stats_out["stage2_median_coverage_gain"] = float(gain.median()) if len(gain) else np.nan
            ax.set_title(
                f"2A Coverage\n% above diagonal={above:.1%}, "
                f"median gain={stats_out['stage2_median_coverage_gain']:.3f}"
            )
        fig.tight_layout()
        fig.savefig(fig_dir / "figure2A_coverage_scatter.png", dpi=200)
        fig.savefig(fig_dir / "figure2A_coverage_scatter.pdf")
        plt.close(fig)

        # Figure 2B beta scatter
        fig, ax = plt.subplots(figsize=(5, 5))
        if not comparison.empty:
            x = comparison["Median_Site_Beta"]
            y = comparison["Hotspot_Beta"]
            ax.scatter(x, y, s=22, alpha=0.7, c="#e6550d", edgecolors="none")
            finite = comparison[["Median_Site_Beta", "Hotspot_Beta"]].dropna()
            if not finite.empty:
                lo = float(min(finite.min().min(), 0))
                hi = float(max(finite.max().max(), 0))
                pad = 0.05 * (hi - lo + 1e-6)
                ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--", color="grey", lw=1)
                ax.axhline(0, color="grey", lw=0.8)
                ax.axvline(0, color="grey", lw=0.8)
            ax.set_xlabel("Median single-site beta")
            ax.set_ylabel("Hotspot beta")
            stats_out["stage2_pct_hotspot_beta_gt0"] = float((comparison["Hotspot_Beta"] > 0).mean())
            stats_out["stage2_pct_median_site_beta_gt0"] = float(
                (comparison["Median_Site_Beta"] > 0).mean()
            )
            stats_out["stage2_median_delta_beta"] = float(
                comparison["Delta_Beta"].median()
            ) if comparison["Delta_Beta"].notna().any() else np.nan
            ax.set_title(
                f"2B Hotspot vs site beta\n"
                f"%HS>0={stats_out['stage2_pct_hotspot_beta_gt0']:.1%}, "
                f"%site>0={stats_out['stage2_pct_median_site_beta_gt0']:.1%}"
            )
        fig.tight_layout()
        fig.savefig(fig_dir / "figure2B_beta_scatter.png", dpi=200)
        fig.savefig(fig_dir / "figure2B_beta_scatter.pdf")
        plt.close(fig)

        # Figure 2C direction consistency
        fig, ax = plt.subplots(figsize=(4.5, 4))
        if not comparison.empty:
            hs_prop = float((comparison["Hotspot_Beta"] > 0).mean())
            # site-level: mean of per-hotspot site expected proportions
            site_prop = float(comparison["Site_Expected_Direction_Proportion"].dropna().mean())
            # also aggregate all site betas
            site_path = self.output_dir / "stage2" / "stage2_site_betas.csv"
            if site_path.exists():
                sb = pd.read_csv(site_path)
                if not sb.empty:
                    site_prop_all = float((sb["Site_Beta"] > 0).mean())
                else:
                    site_prop_all = site_prop
            else:
                site_prop_all = site_prop
            props = [site_prop_all, hs_prop]
            labels = ["Individual phosphosite", "Hotspot"]
            ax.bar(labels, props, color=["#fdae6b", "#3182bd"], width=0.55)
            ax.axhline(0.5, color="grey", ls="--", lw=1)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Proportion with expected direction (β>0)")
            for i, v in enumerate(props):
                ax.text(i, v + 0.02, f"{v:.1%}", ha="center")
            stats_out["stage2_direction_site"] = site_prop_all
            stats_out["stage2_direction_hotspot"] = hs_prop
            ax.set_title("2C Directional consistency")
        fig.tight_layout()
        fig.savefig(fig_dir / "figure2C_direction_consistency.png", dpi=200)
        fig.savefig(fig_dir / "figure2C_direction_consistency.pdf")
        plt.close(fig)

        # Figure 3A signed beta violin
        fig, ax = plt.subplots(figsize=(6, 4.5))
        if not stage3.empty:
            order = ["Known_only", "Mixed", "Predicted_only"]
            present = [t for t in order if t in set(stage3["Hotspot_Type"])]
            sns.violinplot(
                data=stage3,
                x="Hotspot_Type",
                y="Signed_Beta",
                hue="Hotspot_Type",
                order=present,
                inner=None,
                cut=0,
                ax=ax,
                palette="Set2",
                legend=False,
            )
            sns.boxplot(
                data=stage3,
                x="Hotspot_Type",
                y="Signed_Beta",
                order=present,
                width=0.25,
                showfliers=False,
                ax=ax,
                color="white",
            )
            sns.stripplot(
                data=stage3,
                x="Hotspot_Type",
                y="Signed_Beta",
                order=present,
                size=3,
                alpha=0.45,
                color="black",
                ax=ax,
            )
            ax.axhline(0, color="grey", lw=0.8, ls="--")
            ax.set_ylabel("Signed beta (Import × Activate)")
            ax.set_title("3A Signed beta by class")
            by_type = {}
            for t, g in stage3.groupby("Hotspot_Type"):
                by_type[t] = {
                    "n": int(len(g)),
                    "median_signed_beta": float(g["Signed_Beta"].median()),
                    "pct_gt0": float((g["Signed_Beta"] > 0).mean()),
                }
            stats_out["stage3_by_type"] = by_type
        fig.tight_layout()
        fig.savefig(fig_dir / "figure3A_signed_beta_by_class.png", dpi=200)
        fig.savefig(fig_dir / "figure3A_signed_beta_by_class.pdf")
        plt.close(fig)

        # Figure 3B expected-direction proportion
        fig, ax = plt.subplots(figsize=(5, 4))
        if not stage3.empty:
            order = ["Known_only", "Mixed", "Predicted_only"]
            present = [t for t in order if t in set(stage3["Hotspot_Type"])]
            props = []
            cis = []
            ns = []
            for t in present:
                g = stage3[stage3["Hotspot_Type"].eq(t)]
                n = len(g)
                k = int((g["Signed_Beta"] > 0).sum())
                p = k / n if n else np.nan
                # Wilson-ish normal approx CI
                if n > 0:
                    se = np.sqrt(p * (1 - p) / n)
                    ci = (max(0, p - 1.96 * se), min(1, p + 1.96 * se))
                    try:
                        binom_p = float(
                            stats.binomtest(k, n, 0.5, alternative="greater").pvalue
                        )
                    except AttributeError:
                        binom_p = float(stats.binom_test(k, n, 0.5, alternative="greater"))
                else:
                    ci = (np.nan, np.nan)
                    binom_p = np.nan
                props.append(p)
                cis.append(ci)
                ns.append(n)
                stats_out.setdefault("stage3_binom_p", {})[t] = binom_p
            x = np.arange(len(present))
            yerr = np.array([[p - ci[0], ci[1] - p] for p, ci in zip(props, cis)]).T
            ax.bar(x, props, color=["#66c2a5", "#fc8d62", "#8da0cb"][: len(present)], width=0.6)
            ax.errorbar(x, props, yerr=yerr, fmt="none", ecolor="black", capsize=4)
            ax.axhline(0.5, color="grey", ls="--", lw=1)
            ax.set_xticks(x)
            ax.set_xticklabels(present)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Proportion SignedBeta > 0")
            for i, (p, n) in enumerate(zip(props, ns)):
                ax.text(i, p + 0.03, f"{p:.1%}\nN={n}", ha="center", fontsize=8)
            ax.set_title("3B Expected-direction proportion")
            stats_out["stage3_pct_signed_beta_gt0"] = {
                t: float(p) for t, p in zip(present, props)
            }
        fig.tight_layout()
        fig.savefig(fig_dir / "figure3B_expected_direction_proportion.png", dpi=200)
        fig.savefig(fig_dir / "figure3B_expected_direction_proportion.pdf")
        plt.close(fig)

        # Figure 3C heatmap
        if not stage3.empty:
            mat = stage3.pivot_table(
                index="Hotspot_ID", columns="Cancer", values="Signed_Beta", aggfunc="first"
            )
            # annotate types
            type_map = (
                stage3.groupby("Hotspot_ID")["Hotspot_Type"].first().reindex(mat.index)
            )
            # order by type then TF-ish id
            order_types = ["Known_only", "Mixed", "Predicted_only"]
            sort_key = type_map.map({t: i for i, t in enumerate(order_types)})
            mat = mat.iloc[sort_key.argsort(kind="mergesort")]
            type_map = type_map.loc[mat.index]
            fig_h = max(6, 0.18 * len(mat) + 2)
            fig, ax = plt.subplots(figsize=(8, fig_h))
            sns.heatmap(
                mat,
                cmap="RdBu_r",
                center=0,
                ax=ax,
                cbar_kws={"label": "Signed beta"},
                xticklabels=True,
                yticklabels=True,
            )
            ax.set_title("3C Pan-cancer signed beta (all eligible)")
            # side text for types — rewrite yticklabels with type prefix
            new_labels = [f"[{type_map.loc[i]}] {i}" for i in mat.index]
            ax.set_yticklabels(new_labels, fontsize=6)
            fig.tight_layout()
            fig.savefig(fig_dir / "figure3C_pancancer_signed_beta_heatmap.png", dpi=200)
            fig.savefig(fig_dir / "figure3C_pancancer_signed_beta_heatmap.pdf")
            plt.close(fig)
            mat.to_csv(self.output_dir / "stage3" / "stage3_signed_beta_matrix.csv")

        # Predicted_only funnel figure
        pred_funnel_path = self.output_dir / "funnel" / "predicted_only_hotspot_funnel.csv"
        pred_cxh_path = (
            self.output_dir / "funnel" / "predicted_only_cancer_x_hotspot_funnel.csv"
        )
        if pred_funnel_path.exists():
            pf = pd.read_csv(pred_funnel_path)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
            axes[0].barh(pf["step"][::-1], pf["n_hotspots"][::-1], color="#6baed6")
            axes[0].set_xlabel("N hotspots")
            axes[0].set_title("Predicted_only attrition (hotspot-level)")
            for i, (step, n) in enumerate(zip(pf["step"][::-1], pf["n_hotspots"][::-1])):
                axes[0].text(n + 0.3, i, str(int(n)), va="center", fontsize=8)
            if pred_cxh_path.exists():
                pc = pd.read_csv(pred_cxh_path)
                axes[1].barh(pc["step"][::-1], pc["n_cancer_x_hotspot"][::-1], color="#fd8d3c")
                axes[1].set_xlabel("N Cancer×Hotspot")
                axes[1].set_title("Predicted_only attrition (Cancer×Hotspot)")
                for i, (step, n) in enumerate(
                    zip(pc["step"][::-1], pc["n_cancer_x_hotspot"][::-1])
                ):
                    axes[1].text(n + 0.5, i, str(int(n)), va="center", fontsize=8)
            else:
                axes[1].axis("off")
            fig.tight_layout()
            fig.savefig(fig_dir / "figure_predicted_only_funnel.png", dpi=200)
            fig.savefig(fig_dir / "figure_predicted_only_funnel.pdf")
            plt.close(fig)
            stats_out["predicted_only_funnel_hotspot"] = {
                str(r.step): int(r.n_hotspots) for r in pf.itertuples(index=False)
            }
            if pred_cxh_path.exists():
                pc = pd.read_csv(pred_cxh_path)
                stats_out["predicted_only_funnel_cxh"] = {
                    str(r.step): int(r.n_cancer_x_hotspot) for r in pc.itertuples(index=False)
                }

        # Relaxed Stage3 class stats if present
        relaxed_path = (
            self.output_dir
            / "stage3"
            / f"stage3_measurable_relaxed_min{MIN_SCORE_SAMPLES_RELAXED}_by_type.csv"
        )
        if relaxed_path.exists():
            rel = pd.read_csv(relaxed_path)
            stats_out["stage3_relaxed_by_type"] = rel.to_dict(orient="records")

        with open(fig_dir / "figure_stats.json", "w") as f:
            json.dump(stats_out, f, indent=2, default=str)
        return stats_out

    def write_summary(
        self,
        hotspot_df: pd.DataFrame,
        comparison: pd.DataFrame,
        stage3: pd.DataFrame,
        fig_stats: Dict[str, object],
    ) -> None:
        n_by_type = (
            self.hotspots["stage3_type"].value_counts().to_dict()
            if "stage3_type" in self.hotspots.columns
            else {}
        )
        lines = [
            "# CPTAC three-stage hotspot validation — PIPELINE SUMMARY",
            "",
            f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`",
            f"- Output root: `{self.output_dir}`",
            f"- Hotspot catalog: mixed/pred-filter d15 + Known_only benchmark",
            f"- Cancers: {', '.join(self.config.cancer_types)}",
            "",
            "## Catalog",
            "",
            f"- Total hotspots analyzed: **{len(self.hotspots)}**",
            f"- By stage3_type: `{json.dumps(n_by_type)}`",
            f"- Candidate Mixed+Predicted: "
            f"**{int((self.hotspots['source_cohort']=='candidate_mixed_pred').sum())}**",
            f"- Known_only benchmark: "
            f"**{int((self.hotspots['source_cohort']=='known_only_benchmark').sum())}**",
            f"- Nuclear (Stage2/3 eligible gate): "
            f"**{int(nuclear_mask(self.hotspots, self.members).sum())}**",
            "",
            "## Stage 1 — co-phosphorylation coherence",
            "",
            f"- Median within-hotspot rho: "
            f"**{fig_stats.get('stage1_median_within_rho', 'NA')}**",
            f"- Median matched-background rho: "
            f"**{fig_stats.get('stage1_median_background_rho', 'NA')}**",
            f"- Mann–Whitney (within > background) p: "
            f"**{fig_stats.get('stage1_mw_p_greater', 'NA')}**",
            f"- N within pairs / N background pairs: "
            f"**{fig_stats.get('stage1_n_within_pairs', 'NA')}** / "
            f"**{fig_stats.get('stage1_n_background_pairs', 'NA')}**",
            f"- Coherence median by type: `{fig_stats.get('stage1_coherence_by_type', {})}`",
            f"- Background definition (primary): "
            f"`{fig_stats.get('stage1_background_definition', 'distance_matched_exclude_catalog')}`",
            f"- Median far-distance BG rho ({FAR_DIST_MIN}-{FAR_DIST_MAX} aa): "
            f"**{fig_stats.get('stage1_median_far_background_rho', 'NA')}** "
            f"(MW p={fig_stats.get('stage1_mw_p_greater_vs_far', 'NA')}; "
            f"N={fig_stats.get('stage1_n_far_background_pairs', 'NA')})",
            f"- Legacy BG (old definition) median rho: "
            f"**{fig_stats.get('stage1_median_legacy_background_rho', 'NA')}** "
            f"(MW p={fig_stats.get('stage1_mw_p_greater_vs_legacy', 'NA')})",
            "",
        ]
        if not hotspot_df.empty:
            lines += [
                f"- Cancer×Hotspot rows with valid coherence: **{hotspot_df['Median_Rho'].notna().sum()}**",
                "",
            ]
        lines += [
            "## Stage 2 — hotspot vs single site (Import × Activate, nuclear)",
            "",
            f"- Eligible Cancer×Hotspot rows: **{len(comparison)}**",
            f"- % coverage above diagonal: "
            f"**{fig_stats.get('stage2_pct_above_diagonal', 'NA')}**",
            f"- Median coverage gain: "
            f"**{fig_stats.get('stage2_median_coverage_gain', 'NA')}**",
            f"- % Hotspot beta > 0: "
            f"**{fig_stats.get('stage2_pct_hotspot_beta_gt0', 'NA')}**",
            f"- % Median site beta > 0: "
            f"**{fig_stats.get('stage2_pct_median_site_beta_gt0', 'NA')}**",
            f"- Median Δβ (hotspot − median site): "
            f"**{fig_stats.get('stage2_median_delta_beta', 'NA')}**",
            f"- Direction consistency site / hotspot: "
            f"**{fig_stats.get('stage2_direction_site', 'NA')}** / "
            f"**{fig_stats.get('stage2_direction_hotspot', 'NA')}**",
            "",
            "## Stage 3 — Known_only / Mixed / Predicted_only",
            "",
            f"- Eligible Cancer×Hotspot rows: **{len(stage3)}**",
            f"- % SignedBeta > 0 by class: "
            f"`{json.dumps(fig_stats.get('stage3_pct_signed_beta_gt0', {}), default=str)}`",
            f"- Per-class detail: `{json.dumps(fig_stats.get('stage3_by_type', {}), default=str)}`",
            f"- Binomial p (p>0.5) by class: "
            f"`{json.dumps(fig_stats.get('stage3_binom_p', {}), default=str)}`",
            f"- Predicted_only funnel (hotspot-level): "
            f"`{json.dumps(fig_stats.get('predicted_only_funnel_hotspot', {}), default=str)}`",
            f"- Predicted_only funnel (Cancer×Hotspot): "
            f"`{json.dumps(fig_stats.get('predicted_only_funnel_cxh', {}), default=str)}`",
            f"- Stage3 measurable/relaxed (min_score={MIN_SCORE_SAMPLES_RELAXED}) by type: "
            f"`{json.dumps(fig_stats.get('stage3_relaxed_by_type', {}), default=str)}`",
            "",
            "## Limitations",
            "",
            "- Stage 1 primary background: same-protein pairs with distance ±3 aa that "
            "**exclude all catalog hotspot members** (stricter than legacy). "
            "Far-distance (40–100 aa) exclude-catalog pairs are a second control. "
            "Coverage/residue matching is still not enforced.",
            "- Protein-abundance residual adjustment is sensitivity-only and skipped when "
            "protein matrix or TF gene is missing.",
            "- Stage 2/3 require ChIP∩CollecTRI activate targets and ≥10 samples with "
            "HotspotScore; many Predicted_only nuclear hotspots drop for sparse CPTAC coverage. "
            "See `funnel/` for attrition. Relaxed Stage3 uses ≥5 score samples as sensitivity.",
            "- Known_only is a biological benchmark recovered from the same clustering rules, "
            "not an independent hold-out validation set.",
            "- Analyses are within-cancer only; no cross-cancer phospho z-scoring.",
            "",
        ]
        (self.output_dir / "PIPELINE_SUMMARY.md").write_text("\n".join(lines))

    def run(self) -> None:
        # Save catalog used
        self.hotspots.to_csv(self.output_dir / "catalog" / "hotspots_all_types.csv", index=False)
        self.members.to_csv(self.output_dir / "catalog" / "hotspot_members_all_types.csv", index=False)
        meta = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "n_hotspots": int(len(self.hotspots)),
            "n_by_type": self.hotspots["stage3_type"].value_counts().to_dict(),
            "n_nuclear": int(nuclear_mask(self.hotspots, self.members).sum()),
            "cancers": list(self.config.cancer_types),
            "min_overlap": self.min_overlap,
            "dist_tol": self.dist_tol,
            "min_score_samples": self.min_score_samples,
        }
        with open(self.output_dir / "catalog" / "catalog_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print("[Run] Stage 1 ...", flush=True)
        pair_df, hotspot_df, _ = self.run_stage1()
        print("[Run] Stage 2/3 ...", flush=True)
        comparison, _, stage3 = self.run_stage2_stage3()
        print("[Run] Predicted_only / type dropout funnel ...", flush=True)
        self.run_dropout_funnel()
        print(
            f"[Run] Stage3 measurable relaxed (min_score={MIN_SCORE_SAMPLES_RELAXED}) ...",
            flush=True,
        )
        self.run_stage3_measurable_relaxed(MIN_SCORE_SAMPLES_RELAXED)
        print("[Run] Figures ...", flush=True)
        fig_stats = self.plot_figures(pair_df, hotspot_df, comparison, stage3)
        self.write_summary(hotspot_df, comparison, stage3, fig_stats)
        print(f"[Done] outputs in {self.output_dir}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hotspot-dir", type=Path, default=DEFAULT_HOTSPOT_DIR)
    p.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--distance", type=int, default=15)
    p.add_argument("--cancer-types", nargs="+", default=None)
    p.add_argument("--min-overlap", type=int, default=MIN_OVERLAP)
    p.add_argument("--dist-tol", type=int, default=DIST_TOL)
    p.add_argument("--min-score-samples", type=int, default=MIN_SCORE_SAMPLES)
    p.add_argument(
        "--skip-known-only",
        action="store_true",
        help="Only analyze Mixed+Predicted candidate hotspots",
    )
    p.add_argument(
        "--figure1b-only",
        action="store_true",
        help="Only regenerate Figure 1B heatmaps using existing stage1_hotspot_level.csv",
    )
    p.add_argument(
        "--figure1b-hotspots",
        nargs="+",
        default=None,
        help="Optional hotspot_id list for Figure 1B (default: STAT3_Y686-T708, HSF1_S292-S326)",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    # Safety: never overwrite mixed_pred_filter / three_level trees
    out = Path(args.output_dir).resolve()
    forbidden_suffixes = (
        "import_target_regulation_hotspot_v11_147pos_d3_platt_mixed_pred_filter",
        "import_target_regulation_hotspot_v11_147pos_d3_platt_mixed_pred_filter_three_level",
    )
    if out.name in forbidden_suffixes or any(s in str(out) and out.name.endswith(s) for s in []):
        raise SystemExit(f"Refusing to write into protected directory: {out}")

    cand_h, cand_m = load_candidate_tables(args.hotspot_dir, args.distance)
    if args.skip_known_only:
        hotspots, members = cand_h, cand_m
    else:
        known_h, known_m = recover_known_only(args.hotspot_dir, args.summary_csv)
        hotspots = pd.concat([cand_h, known_h], ignore_index=True, sort=False)
        members = pd.concat([cand_m, known_m], ignore_index=True, sort=False)

    print(
        f"Catalog: {len(hotspots)} hotspots "
        f"({hotspots['stage3_type'].value_counts().to_dict()})",
        flush=True,
    )

    config = TempoConfig(
        output_dir=str(out),
        cancer_types=list(args.cancer_types) if args.cancer_types else list(DEFAULT_CANCER_LIST),
        directions=["Nuclear Import"],
    )
    runner = ThreeStageHotspotValidation(
        config=config,
        hotspots=hotspots,
        members=members,
        output_dir=out,
        min_overlap=args.min_overlap,
        dist_tol=args.dist_tol,
        min_score_samples=args.min_score_samples,
    )

    if args.figure1b_only:
        hotspot_csv = out / "stage1" / "stage1_hotspot_level.csv"
        if not hotspot_csv.exists():
            raise SystemExit(f"Missing {hotspot_csv}; run full pipeline first.")
        hotspot_df = pd.read_csv(hotspot_csv)
        print("[Run] Figure 1B only ...", flush=True)
        meta = runner.plot_figure1b(hotspot_df, hotspot_ids=args.figure1b_hotspots)
        stats_path = out / "figures" / "figure_stats.json"
        stats: Dict[str, object] = {}
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
        stats["stage1_figure1b"] = meta
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"[Done] Figure 1B written under {out / 'figures'}", flush=True)
        return

    runner.run()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
