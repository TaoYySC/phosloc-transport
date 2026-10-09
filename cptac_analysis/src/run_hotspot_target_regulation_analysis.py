#!/usr/bin/env python3
"""Hotspot-level CPTAC target-regulation scan (parallel to unit-site pipeline).

Activity modes:
  - mean_z / max / direction_weighted: quantitative scores; typically paired with
    phospho_split_mode=median_nonmissing (High/Low by median).
  - any_detected: per-sample indicator — 1.0 if ≥1 hotspot member phosphosite is
    observed (notna) in that sample, else NaN if all members missing. Pair with
    phospho_split_mode=observed_missing so High=detected, Low=undetected (labels
    remain high/low phospho for downstream compatibility).

Does NOT overwrite unit-site results; write to a dedicated --output-dir.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from run_import_target_regulation_analysis import (  # noqa: E402
    DEFAULT_CANCER_LIST,
    EXPRESSION_MODE_UNADJUSTED,
    TargetRegulationBoxplotPipeline,
    TempoConfig,
    VALID_DIRECTIONS,
)

_CPTAC_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _CPTAC_ROOT.parent
# Canonical pipeline: Known/HC-only clustering first, then CPTAC on those hotspots.
DEFAULT_HOTSPOT_DIR = (
    _CPTAC_ROOT
    / "results/import_target_regulation_hotspot_v11_147pos_d3_platt/hotspots"
)
DEFAULT_OUTPUT_DIR = (
    _CPTAC_ROOT
    / "results/import_target_regulation_hotspot_v11_147pos_d3_platt_anchor_only"
)

DIRECTION_CALL_FOR = {
    "Nuclear Import": "nuclear",
    "Nuclear Export": "cytoplasmic",
}
# CPTAC direction gate: per-site nuclear/cyto members, not mean direction_call
IMPORT_GATE_COL = {
    "Nuclear Import": "has_nuclear_member",
    "Nuclear Export": "has_cytoplasmic_member",
}
VALID_ACTIVITY_MODES = ["mean_z", "max", "direction_weighted", "any_detected"]


def zscore_rows(matrix: pd.DataFrame) -> pd.DataFrame:
    """Sample-wise z-score within each site row (nan-aware)."""
    numeric = matrix.apply(pd.to_numeric, errors="coerce")
    mean = numeric.mean(axis=1, skipna=True)
    std = numeric.std(axis=1, skipna=True, ddof=0)
    std = std.replace(0, np.nan)
    return numeric.sub(mean, axis=0).div(std, axis=0)


class HotspotTargetRegulationPipeline(TargetRegulationBoxplotPipeline):
    """Unit-site pipeline with hotspot activity rows injected into phospho matrices."""

    def __init__(
        self,
        config: TempoConfig,
        hotspots: pd.DataFrame,
        members: pd.DataFrame,
        activity_mode: str = "mean_z",
        cluster_distance: int = 10,
    ):
        super().__init__(config)
        if activity_mode not in VALID_ACTIVITY_MODES:
            raise ValueError(f"Unsupported activity_mode: {activity_mode}")
        self.hotspots = hotspots.copy()
        self.members = members.copy()
        self.activity_mode = activity_mode
        self.cluster_distance = int(cluster_distance)
        self._activity_cache: Dict[str, pd.DataFrame] = {}
        self._activity_meta_cache: Dict[str, pd.DataFrame] = {}
        self._base_phospho_cache: Dict[str, pd.DataFrame] = {}

        self.members["ACC_ID"] = self.members["protein_acc"].astype(str)
        self.members["RESIDUE"] = self.members["residue"].astype(str).str.upper()
        self.members["POSITION"] = pd.to_numeric(self.members["position"], errors="coerce").astype("Int64")
        idmap = self.load_idmapping().drop_duplicates("ACC_ID")
        self.members = self.members.merge(idmap, on="ACC_ID", how="left")
        self.members["ENSEMBL_GENE_ID"] = (
            self.members["ENSEMBL_GENE_ID"].astype(str).str.split(".").str[0]
        )
        self.members["cptac_site"] = (
            self.members["ENSEMBL_GENE_ID"]
            + "|"
            + self.members["RESIDUE"]
            + self.members["POSITION"].astype(str)
        )
        self.hotspot_meta = self.hotspots.set_index("hotspot_id", drop=False)

    @staticmethod
    def hotspot_site_key(ensembl: str, hotspot_id: str) -> str:
        return f"{ensembl}|HS_{hotspot_id}"

    @staticmethod
    def parse_hotspot_id_from_site(site: str) -> str:
        s = str(site)
        if "|HS_" in s:
            return s.split("|HS_", 1)[1]
        return s

    def load_base_phospho(self, cancer_type: str) -> pd.DataFrame:
        if cancer_type not in self._base_phospho_cache:
            self._base_phospho_cache[cancer_type] = super().load_phospho(cancer_type)
        return self._base_phospho_cache[cancer_type]

    def load_phospho(self, cancer_type: str) -> pd.DataFrame:
        base = self.load_base_phospho(cancer_type)
        activity, _ = self._ensure_hotspot_activity(cancer_type)
        if activity.empty:
            return base
        # Prefer hotspot rows when keys collide (should not collide).
        combined = pd.concat([base, activity], axis=0)
        return combined[~combined.index.duplicated(keep="last")]

    def _direction_hotspots(self, direction: str) -> pd.DataFrame:
        """Hotspots eligible for a transport direction.

        Gate by has_nuclear_member / has_cytoplasmic_member (pure direction:
        nuclear-only or cytoplasmic-only members). Falls back to legacy
        direction_call == nuclear/cytoplasmic if the new columns are absent.
        """
        gate_col = IMPORT_GATE_COL.get(direction)
        if gate_col and gate_col in self.hotspots.columns:
            mask = self.hotspots[gate_col].astype(bool)
            return self.hotspots.loc[mask].copy()
        call = DIRECTION_CALL_FOR.get(direction)
        if call is None:
            raise ValueError(f"Unsupported direction for hotspot filter: {direction}")
        return self.hotspots[self.hotspots["direction_call"].astype(str).eq(call)].copy()

    def _compute_hotspot_activity_matrix(
        self,
        cancer_type: str,
        df_phospho: pd.DataFrame,
        activity_mode: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        mode = activity_mode or self.activity_mode
        rows: Dict[str, pd.Series] = {}
        meta_rows: List[Dict[str, object]] = []
        samples = list(df_phospho.columns)

        for hotspot_id, mem in self.members.groupby("hotspot_id", sort=False):
            if hotspot_id not in self.hotspot_meta.index:
                continue
            hmeta = self.hotspot_meta.loc[hotspot_id]
            if isinstance(hmeta, pd.DataFrame):
                hmeta = hmeta.iloc[0]

            mem = mem.dropna(subset=["ENSEMBL_GENE_ID", "POSITION"]).copy()
            mem = mem[mem["ENSEMBL_GENE_ID"].astype(str).str.startswith("ENSG")].copy()
            if mem.empty:
                continue

            measured = [s for s in mem["cptac_site"].astype(str) if s in df_phospho.index]
            if not measured:
                continue

            sub = df_phospho.loc[measured].apply(pd.to_numeric, errors="coerce")

            if mode == "any_detected":
                # Detected (≥1 member notna) → 1.0; all missing → NaN.
                # With observed_missing split: high=detected, low=undetected.
                any_obs = sub.notna().any(axis=0)
                score = pd.Series(
                    np.where(any_obs.to_numpy(), 1.0, np.nan),
                    index=any_obs.index,
                    dtype=float,
                )
            else:
                z = zscore_rows(sub)
                if mode == "mean_z":
                    score = z.mean(axis=0, skipna=True)
                elif mode == "max":
                    # Highest FuncTransport among measured members → that site's z
                    mem_meas = mem[mem["cptac_site"].isin(measured)].copy()
                    mem_meas["FuncTransport_score"] = pd.to_numeric(
                        mem_meas["FuncTransport_score"], errors="coerce"
                    )
                    top = mem_meas.sort_values("FuncTransport_score", ascending=False).iloc[0]
                    score = z.loc[str(top["cptac_site"])]
                    if isinstance(score, pd.DataFrame):
                        score = score.iloc[0]
                elif mode == "direction_weighted":
                    mem_meas = mem[mem["cptac_site"].isin(measured)].set_index("cptac_site")
                    signs = np.sign(
                        pd.to_numeric(mem_meas.loc[z.index, "Direction_score"], errors="coerce")
                    ).replace(0, np.nan)
                    if isinstance(signs, pd.DataFrame):
                        signs = signs.iloc[:, 0]
                    weighted = z.mul(signs, axis=0)
                    score = weighted.mean(axis=0, skipna=True)
                else:
                    raise ValueError(mode)

            if isinstance(score, pd.DataFrame):
                score = score.iloc[0]
            if not isinstance(score, pd.Series):
                score = pd.Series(score)
            score = pd.to_numeric(score, errors="coerce").reindex(samples)
            n_obs = int(score.notna().sum())
            n_miss = int(score.isna().sum())
            min_g = int(self.config.min_group_samples)
            if self.config.phospho_split_mode == "observed_missing":
                # Need both detected and undetected arms
                if n_obs < min_g or n_miss < min_g:
                    continue
            elif n_obs < 2 * min_g:
                # Median (or quantile) split needs enough observed samples
                continue

            ensembl = str(mem["ENSEMBL_GENE_ID"].iloc[0])
            site_key = self.hotspot_site_key(ensembl, str(hotspot_id))
            rows[site_key] = score

            # Labeling helpers for known_positive coloring / display
            known_mem = mem[mem["evidence"].astype(str).eq("Known")]
            if not known_mem.empty:
                label_row = known_mem.iloc[0]
            else:
                label_row = mem.sort_values("FuncTransport_score", ascending=False).iloc[0]

            start_pos = int(hmeta["start_position"])
            end_pos = int(hmeta["end_position"])
            span_label = (
                f"{mem.sort_values('position')['site'].iloc[0]}-{mem.sort_values('position')['site'].iloc[-1]}"
                if len(mem) >= 1
                else f"S{start_pos}-S{end_pos}"
            )

            meta_rows.append(
                {
                    "site": site_key,
                    "hotspot_id": str(hotspot_id),
                    "hotspot_label": str(hmeta.get("hotspot_label", hotspot_id)),
                    "hotspot_class": str(hmeta.get("hotspot_class", "")),
                    "direction_call": str(hmeta.get("direction_call", "")),
                    "ACC_ID": str(label_row["ACC_ID"]),
                    "RESIDUE": "",
                    "POSITION": span_label,
                    "ENSEMBL_GENE_ID": ensembl,
                    "tf_gene_id": ensembl,
                    "tf_name": str(hmeta.get("gene_name", "")),
                    "n_members": int(hmeta.get("n_members", len(mem))),
                    "n_measured_members": int(len(measured)),
                    "measured_sites": ";".join(measured),
                    "member_sites": str(hmeta.get("member_sites", "")),
                    "score": float(hmeta.get("mean_Direction_score", np.nan)),
                    "mean_FuncTransport_score": float(
                        hmeta.get("mean_FuncTransport_score", np.nan)
                    ),
                    "activity_mode": mode,
                    "cluster_distance": self.cluster_distance,
                    "has_known": bool(hmeta.get("has_known", False)),
                }
            )

        activity = pd.DataFrame(rows).T if rows else pd.DataFrame(columns=samples)
        meta = pd.DataFrame(meta_rows)
        return activity, meta

    def _ensure_hotspot_activity(
        self, cancer_type: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if cancer_type in self._activity_cache:
            return self._activity_cache[cancer_type], self._activity_meta_cache[cancer_type]
        base = self.load_base_phospho(cancer_type)
        activity, meta = self._compute_hotspot_activity_matrix(cancer_type, base)
        self._activity_cache[cancer_type] = activity
        self._activity_meta_cache[cancer_type] = meta
        return activity, meta

    def analyze_cancer_direction(
        self, cancer_type: str, direction: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print(
            f"Processing hotspot {cancer_type} | {direction} | activity={self.activity_mode} | "
            f"d={self.cluster_distance} | split={self.config.phospho_split_mode}"
        )
        df_phospho = self.load_phospho(cancer_type)
        activity, meta = self._ensure_hotspot_activity(cancer_type)
        if activity.empty or meta.empty:
            print(f"  No measurable hotspots in {cancer_type}")
            return pd.DataFrame(), pd.DataFrame()

        direction_hotspots = set(self._direction_hotspots(direction)["hotspot_id"].astype(str))
        meta = meta[meta["hotspot_id"].astype(str).isin(direction_hotspots)].copy()
        if meta.empty:
            print(f"  No gated hotspots measurable in {cancer_type} for {direction}")
            return pd.DataFrame(), pd.DataFrame()

        sites = meta["site"].astype(str).tolist()
        df_variable = self.find_variable_sites(
            df_phospho=df_phospho,
            sites=sites,
            min_nonzero_ratio=self.config.min_nonzero_ratio,
            max_nonzero_ratio=self.config.max_nonzero_ratio,
            split_mode=self.config.phospho_split_mode,
            min_group_samples=self.config.min_group_samples,
        )
        if df_variable.empty:
            return pd.DataFrame(), pd.DataFrame()

        df_sites = df_variable.merge(meta, on="site", how="left")
        df_sites = df_sites.dropna(subset=["tf_name"]).copy()
        if df_sites.empty:
            return pd.DataFrame(), pd.DataFrame()

        df_rna = self.load_rna(cancer_type)
        df_protein = self.load_protein(cancer_type) if self._needs_protein_matrix() else None

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
                    # Attach hotspot metadata to point rows
                    for row in rows:
                        row["hotspot_id"] = site_row.get("hotspot_id")
                        row["hotspot_label"] = site_row.get("hotspot_label")
                        row["hotspot_class"] = site_row.get("hotspot_class")
                        row["n_measured_members"] = site_row.get("n_measured_members")
                        row["n_members"] = site_row.get("n_members")
                        row["member_sites"] = site_row.get("member_sites")
                        row["activity_mode"] = self.activity_mode
                    summary.update(
                        {
                            "hotspot_id": site_row.get("hotspot_id"),
                            "hotspot_label": site_row.get("hotspot_label"),
                            "hotspot_class": site_row.get("hotspot_class"),
                            "n_measured_members": site_row.get("n_measured_members"),
                            "n_members": site_row.get("n_members"),
                            "member_sites": site_row.get("member_sites"),
                            "measured_sites": site_row.get("measured_sites"),
                            "activity_mode": self.activity_mode,
                            "cluster_distance": self.cluster_distance,
                        }
                    )
                    if expression_mode == EXPRESSION_MODE_UNADJUSTED or site_summary is None:
                        site_summary = summary
                    all_rows.extend(rows)
                except Exception as exc:
                    site = str(site_row.get("site", ""))
                    tf_name = str(site_row.get("tf_name", ""))
                    if site_summary is None:
                        site_summary = {
                            "cancer_type": cancer_type,
                            "direction": direction,
                            "direction_short": self._direction_short(direction),
                            "site": site,
                            "tf_name": tf_name,
                            "hotspot_id": site_row.get("hotspot_id"),
                            "hotspot_class": site_row.get("hotspot_class"),
                            "status": f"error: {exc}",
                            "error_type": type(exc).__name__,
                        }
                    print(
                        f"Warning: skipped {cancer_type} | {expression_mode} | "
                        f"{tf_name} | {site_row.get('hotspot_label')}: {exc}"
                    )
            if site_summary is not None:
                summary_rows.append(site_summary)

        df_points = pd.DataFrame(all_rows)
        df_summary = pd.DataFrame(summary_rows)
        print(
            f"  {cancer_type}: {df_sites['hotspot_id'].nunique()} hotspots scanned; "
            f"{len(df_points)} expression point rows; "
            f"success={(df_summary['status'].eq('success').sum() if not df_summary.empty else 0)}"
        )
        return df_points, df_summary

    def _annotate_display_labels(self, df_points: pd.DataFrame) -> pd.DataFrame:
        if df_points.empty:
            return df_points
        out = df_points.copy()
        if "hotspot_label" in out.columns and out["hotspot_label"].notna().any():
            out["site_label"] = out["hotspot_label"].astype(str)
        else:
            out["site_label"] = self._site_display_label_from_df(out)
        return out

    def export_stratified_tables(self, df_stats: pd.DataFrame, output_dir: Path) -> None:
        if df_stats.empty:
            return
        out = output_dir / "stratified_hotspot_associations"
        out.mkdir(parents=True, exist_ok=True)
        stats = df_stats.copy()
        if "hotspot_class" not in stats.columns and "site" in stats.columns:
            # recover from meta across cancers
            meta_parts = []
            for cancer, meta in self._activity_meta_cache.items():
                m = meta.copy()
                m["cancer_type"] = cancer
                meta_parts.append(m)
            if meta_parts:
                meta_all = pd.concat(meta_parts, ignore_index=True)
                stats = stats.merge(
                    meta_all[
                        [
                            "site",
                            "hotspot_id",
                            "hotspot_label",
                            "hotspot_class",
                            "n_measured_members",
                            "n_members",
                        ]
                    ].drop_duplicates("site"),
                    on="site",
                    how="left",
                )
        stats.to_csv(out / "hotspot_association_stats_all.csv", index=False)
        for cls in ["known_containing", "known_proximal", "known_independent"]:
            sub = stats[stats.get("hotspot_class", pd.Series(dtype=str)).astype(str).eq(cls)]
            sub.to_csv(out / f"hotspot_association_{cls}.csv", index=False)
        # BH-significant activate Import
        if "target_regulation" in stats.columns:
            sig = stats[
                stats["target_regulation"].astype(str).eq("activate")
                & (pd.to_numeric(stats.get("wilcoxon_q_bh"), errors="coerce") < 0.05)
            ].copy()
            sig.to_csv(out / "hotspot_association_activate_bh_significant.csv", index=False)
            if "hotspot_class" in sig.columns:
                sig[sig["hotspot_class"].eq("known_independent")].to_csv(
                    out / "hotspot_association_known_independent_activate_bh_significant.csv",
                    index=False,
                )

    def compute_activity_concordance(
        self,
        significant_keys: pd.DataFrame,
        output_dir: Path,
    ) -> pd.DataFrame:
        """For significant cancer×hotspot pairs, compare max / direction_weighted vs mean_z."""
        if significant_keys.empty:
            return pd.DataFrame()

        rows: List[Dict[str, object]] = []
        for _, key in significant_keys.iterrows():
            cancer = str(key["cancer_type"])
            site = str(key["site"])
            hotspot_id = self.parse_hotspot_id_from_site(site)
            direction_short = str(key.get("direction_short", "Import"))
            direction = self._direction_label(direction_short)

            base = self.load_base_phospho(cancer)
            df_rna = self.load_rna(cancer)
            df_protein = self.load_protein(cancer) if self._needs_protein_matrix() else None

            mean_split = self._get_phospho_high_low_samples(
                site, self.load_phospho(cancer), df_rna, df_protein=df_protein
            )
            if mean_split.get("status") != "success":
                continue

            concordance = {
                "cancer_type": cancer,
                "direction_short": direction_short,
                "site": site,
                "hotspot_id": hotspot_id,
                "hotspot_label": key.get("hotspot_label", key.get("site_label", "")),
                "tf_name": key.get("tf_name", ""),
                "hotspot_class": key.get("hotspot_class", ""),
                "mean_z_delta": key.get("delta_high_minus_low", np.nan),
                "mean_z_q": key.get("wilcoxon_q_bh", np.nan),
                "mean_z_significant": True,
            }

            for alt_mode in ["max", "direction_weighted"]:
                # Compute activity only for this hotspot (full-catalog recompute is too slow).
                mem = self.members[self.members["hotspot_id"].astype(str).eq(hotspot_id)].copy()
                if mem.empty or hotspot_id not in self.hotspot_meta.index:
                    concordance[f"{alt_mode}_status"] = "missing_members"
                    concordance[f"{alt_mode}_same_direction"] = np.nan
                    continue
                # Temporarily restrict member table to this hotspot for activity calc
                prev_members = self.members
                prev_meta = self.hotspot_meta
                try:
                    self.members = mem
                    self.hotspot_meta = self.hotspots[
                        self.hotspots["hotspot_id"].astype(str).eq(hotspot_id)
                    ].set_index("hotspot_id", drop=False)
                    act, meta = self._compute_hotspot_activity_matrix(
                        cancer, base, activity_mode=alt_mode
                    )
                finally:
                    self.members = prev_members
                    self.hotspot_meta = prev_meta

                if site not in act.index:
                    concordance[f"{alt_mode}_status"] = "not_measurable"
                    concordance[f"{alt_mode}_same_direction"] = np.nan
                    continue
                # Temporarily inject alt activity into a synthetic phospho for split
                synth = base.copy()
                synth.loc[site] = act.loc[site]
                split = self._get_phospho_high_low_samples(
                    site, synth, df_rna, df_protein=df_protein
                )
                if split.get("status") != "success":
                    concordance[f"{alt_mode}_status"] = split.get("status")
                    concordance[f"{alt_mode}_same_direction"] = np.nan
                    continue

                # Rebuild expression comparison quickly for activate targets only
                site_meta = meta[meta["site"].eq(site)]
                if site_meta.empty:
                    concordance[f"{alt_mode}_status"] = "missing_meta"
                    continue
                site_row = site_meta.iloc[0].copy()
                # Preserve TF fields from mean_z key if needed
                site_row["tf_name"] = key.get("tf_name", site_row.get("tf_name"))
                chip_cache: Dict[str, pd.DataFrame] = {}
                point_rows, summary = self.build_site_target_expression_rows(
                    cancer_type=cancer,
                    direction=direction,
                    site_row=site_row,
                    df_rna=df_rna,
                    df_phospho=synth,
                    chip_cache=chip_cache,
                    df_protein=df_protein,
                    expression_mode=self._primary_expression_mode(),
                )
                concordance[f"{alt_mode}_status"] = summary.get("status")
                if not point_rows:
                    concordance[f"{alt_mode}_same_direction"] = np.nan
                    continue
                pts = pd.DataFrame(point_rows)
                pts = pts[pts["target_regulation"].astype(str).eq("activate")]
                if pts.empty:
                    concordance[f"{alt_mode}_same_direction"] = np.nan
                    continue
                pts["site_label"] = pts.get("hotspot_label", site)
                stats = self._compare_high_low_by_site(pts)
                if stats.empty:
                    concordance[f"{alt_mode}_same_direction"] = np.nan
                    continue
                delta = float(stats.iloc[0]["delta_high_minus_low"])
                mean_delta = float(pd.to_numeric(key.get("delta_high_minus_low"), errors="coerce"))
                concordance[f"{alt_mode}_delta"] = delta
                concordance[f"{alt_mode}_p"] = float(stats.iloc[0]["wilcoxon_p_expected"])
                concordance[f"{alt_mode}_same_direction"] = bool(
                    np.sign(delta) == np.sign(mean_delta) or (delta == 0 and mean_delta == 0)
                )
                concordance[f"{alt_mode}_significant_raw_p05"] = bool(
                    pd.notna(stats.iloc[0]["wilcoxon_p_expected"])
                    and float(stats.iloc[0]["wilcoxon_p_expected"]) < 0.05
                )

            rows.append(concordance)

        out = pd.DataFrame(rows)
        out_dir = output_dir / "activity_concordance"
        out_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_dir / "significant_hotspot_max_direction_weighted_concordance.csv", index=False)
        if not out.empty:
            summary = {
                "n_significant_pairs": int(len(out)),
                "max_same_direction_frac": float(
                    pd.to_numeric(out.get("max_same_direction"), errors="coerce").mean()
                )
                if "max_same_direction" in out.columns
                else np.nan,
                "direction_weighted_same_direction_frac": float(
                    pd.to_numeric(out.get("direction_weighted_same_direction"), errors="coerce").mean()
                )
                if "direction_weighted_same_direction" in out.columns
                else np.nan,
            }
            (out_dir / "concordance_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
        return out

    def run_scan(self) -> Path:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.activity_mode == "any_detected":
            activity_desc = (
                "any_detected: sample scored 1.0 if ≥1 CPTAC-mapped hotspot member "
                "phosphosite is observed (notna), else NaN if all members missing; "
                "with observed_missing split, high=detected / low=undetected"
            )
        elif self.config.phospho_split_mode == "detected_median":
            activity_desc = (
                "detected_median: Undetected=all members missing; Detected mean_z "
                "median → Low/High. TF residualization for expression fits on "
                "Detected samples only (Undetected get optional out-of-sample residuals)."
            )
        else:
            activity_desc = (
                f"{self.activity_mode} of CPTAC-measured hotspot members; "
                f"split={self.config.phospho_split_mode}"
            )
        run_meta = asdict(self.config)
        run_meta.update(
            {
                "analysis_type": "hotspot",
                "cluster_distance": self.cluster_distance,
                "activity_mode": self.activity_mode,
                "seed_evidence": ["Known", "Cluster", "Predicted", "Predicted;Cluster"],
                "hotspot_stratification": {
                    "known_containing": "contains curated Known member",
                    "known_proximal": "no Known member; min distance to Known on same TF <= 30 aa",
                    "known_independent": "TF has no Known, or min distance to Known > 30 aa",
                },
                "rationale": (
                    "Import CPTAC uses hotspots with pure has_nuclear_member "
                    "(≥1 nuclear AND 0 cytoplasmic members). "
                    f"Activity/grouping: {activity_desc}. "
                    "Downstream still labels groups as high/low phospho."
                ),
                "hotspot_definition": (
                    "Catalog as provided (--hotspot-dir); Import gate=has_nuclear_member; "
                    f"activity={self.activity_mode}; split={self.config.phospho_split_mode}."
                ),
                "import_gate_column": "has_nuclear_member",
            }
        )
        with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(run_meta, f, indent=2)

        print("=" * 60)
        print("Hotspot target regulation scan")
        print("=" * 60)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output directory: {output_dir}")
        print(f"Activity mode: {self.activity_mode}")
        print(f"Cluster distance: {self.cluster_distance}")
        print(f"Cancer types: {', '.join(self.config.cancer_types)}")
        print(f"Directions: {', '.join(self.config.directions)}")

        point_frames: List[pd.DataFrame] = []
        summary_frames: List[pd.DataFrame] = []
        for cancer_type in self.config.cancer_types:
            for direction in self.config.directions:
                df_points, df_summary = self.analyze_cancer_direction(cancer_type, direction)
                if not df_points.empty:
                    point_frames.append(df_points)
                if not df_summary.empty:
                    summary_frames.append(df_summary)

        df_all_points = (
            pd.concat(point_frames, axis=0).reset_index(drop=True) if point_frames else pd.DataFrame()
        )
        df_all_summary = (
            pd.concat(summary_frames, axis=0).reset_index(drop=True) if summary_frames else pd.DataFrame()
        )

        if not df_all_points.empty and "expression_mode" not in df_all_points.columns:
            df_all_points["expression_mode"] = EXPRESSION_MODE_UNADJUSTED

        df_all_points = self._annotate_display_labels(df_all_points)
        df_all_points.to_csv(
            output_dir / "all_target_gene_mean_expression_points_all_modes.csv", index=False
        )
        df_primary_points = self._filter_points_by_expression_mode(
            df_all_points, self._primary_expression_mode()
        )
        df_primary_points = self._annotate_display_labels(df_primary_points)
        df_primary_points.to_csv(
            output_dir / "all_target_gene_mean_expression_points.csv", index=False
        )
        df_all_summary.to_csv(output_dir / "hotspot_level_processing_summary.csv", index=False)
        # Compatibility alias for downstream tools expecting site_level name
        df_all_summary.to_csv(output_dir / "site_level_processing_summary.csv", index=False)

        print(
            f"Primary expression mode: {self._primary_expression_mode()} "
            f"({len(df_primary_points)} point rows)"
        )
        print("Plotting hotspot high/low target expression boxplots...")
        self.plot_all_boxplots(df_primary_points, output_dir)
        print("Plotting merged hotspot high/low boxplots...")
        self.plot_all_merged_boxplots(df_primary_points, output_dir)

        print("Running extended logFC / random / TF-activity analysis...")
        # Disable confounder re-scan of unit phospho for speed; residual already in primary.
        prev_conf = self.config.confounder_analysis
        self.config.confounder_analysis = "none"
        df_all_summary = self.run_target_logfc_activity_random_analysis(
            df_primary_points,
            output_dir,
            df_all_summary=df_all_summary,
            run_confounder=False,
        )
        self.config.confounder_analysis = prev_conf
        df_all_summary.to_csv(output_dir / "hotspot_level_processing_summary.csv", index=False)
        df_all_summary.to_csv(output_dir / "site_level_processing_summary.csv", index=False)

        # Stratified association tables from boxplot stats
        stats_path = (
            output_dir
            / "high_low_phospho_boxplots"
            / "high_low_phospho_comparison_by_site_all.csv"
        )
        if stats_path.exists():
            df_stats = pd.read_csv(stats_path)
            # merge hotspot class from points/summary
            if not df_primary_points.empty and (
                "hotspot_class" not in df_stats.columns or "n_members" not in df_stats.columns
            ):
                map_cols = (
                    df_primary_points[
                        [
                            "site",
                            "hotspot_id",
                            "hotspot_label",
                            "hotspot_class",
                            "n_measured_members",
                            "n_members",
                        ]
                    ]
                    .drop_duplicates("site")
                )
                drop = [
                    c
                    for c in [
                        "hotspot_id",
                        "hotspot_label",
                        "hotspot_class",
                        "n_measured_members",
                        "n_members",
                    ]
                    if c in df_stats.columns
                ]
                df_stats = df_stats.drop(columns=drop).merge(map_cols, on="site", how="left")
                df_stats.to_csv(stats_path, index=False)
            self.export_stratified_tables(df_stats, output_dir)

            sig = df_stats[
                df_stats["target_regulation"].astype(str).eq("activate")
                & (pd.to_numeric(df_stats.get("wilcoxon_q_bh"), errors="coerce") < 0.05)
            ].copy()
            if not sig.empty and self.activity_mode != "any_detected":
                print(f"Computing activity concordance for {len(sig)} significant activate pairs...")
                self.compute_activity_concordance(sig, output_dir)
            elif not sig.empty and self.activity_mode == "any_detected":
                print("Skipping max/direction_weighted concordance (activity=any_detected).")

        print("=" * 60)
        print(f"Done. Outputs under {output_dir}")
        return output_dir / "all_target_gene_mean_expression_points.csv"


def load_hotspot_tables(hotspot_dir: Path, distance: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    hotspots = pd.read_csv(hotspot_dir / f"hotspots_d{distance}.csv")
    members = pd.read_csv(hotspot_dir / f"hotspot_members_d{distance}.csv")
    return hotspots, members


def parse_hotspot_args() -> argparse.Namespace:
    # Reuse unit-site CLI, then add hotspot-specific flags.
    parser = argparse.ArgumentParser(
        description="Hotspot-level CPTAC target regulation analysis (mean-z activity)."
    )
    parser.add_argument("mode", choices=["scan"], nargs="?", default="scan")
    parser.add_argument("--hotspot-dir", type=Path, default=DEFAULT_HOTSPOT_DIR)
    parser.add_argument("--distance", type=int, default=10)
    parser.add_argument(
        "--activity",
        choices=VALID_ACTIVITY_MODES,
        default="mean_z",
        help=(
            "Hotspot activity: mean_z/max/direction_weighted, or any_detected "
            "(pair with --phospho-split-mode observed_missing)"
        ),
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        default=["Nuclear Import"],
        choices=VALID_DIRECTIONS,
    )
    parser.add_argument("--cancer-types", nargs="+", default=DEFAULT_CANCER_LIST)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--linkedomics-base", default=TempoConfig.linkedomics_base)
    parser.add_argument("--chip-dir", default=TempoConfig.chip_dir)
    parser.add_argument("--signed-regulon-path", default=TempoConfig.signed_regulon_path)
    parser.add_argument("--idmapping-path", default=TempoConfig.idmapping_path)
    parser.add_argument("--prediction-output-dir", default=TempoConfig.prediction_output_dir)
    parser.add_argument("--import-prediction-filename", default=TempoConfig.import_prediction_filename)
    parser.add_argument("--export-prediction-filename", default=TempoConfig.export_prediction_filename)
    parser.add_argument("--known-positive-path", default=TempoConfig.known_positive_path)
    parser.add_argument("--ensembl-release", type=int, default=TempoConfig.ensembl_release)
    parser.add_argument("--species", default=TempoConfig.species)
    parser.add_argument("--max-missing-ratio", type=float, default=0.8)
    parser.add_argument("--min-nonzero-ratio", type=float, default=0.05)
    parser.add_argument("--max-nonzero-ratio", type=float, default=0.95)
    parser.add_argument("--chip-threshold", type=float, default=200.0)
    parser.add_argument("--min-chip-sample-frac", type=float, default=0.1)
    parser.add_argument("--chip-top-n", type=int, default=500)
    parser.add_argument("--signed-target-mode", default="chip_intersection")
    parser.add_argument("--phospho-group-frac", type=float, default=0.08)
    parser.add_argument("--phospho-split-mode", default="median_nonmissing")
    parser.add_argument("--min-group-samples", type=int, default=3)
    parser.add_argument("--min-box-points", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--random-iterations", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-random-gene-rows-per-group", type=int, default=50000)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--confounder-analysis", default="none")
    parser.add_argument("--abundance-primary-mode", default="residual")
    parser.add_argument("--phospho-value-mode", default="site_abundance")
    parser.add_argument("--adjustment-covariates", nargs="+", default=["tf_protein"])
    parser.add_argument("--purity-column", default="WES_purity")
    parser.add_argument("--min-samples-for-adjustment", type=int, default=10)
    parser.add_argument("--use-bh-pvalue-correction", action="store_true", default=True)
    parser.add_argument("--no-bh-pvalue-correction", action="store_true")
    parser.add_argument(
        "--test-alternative",
        choices=["directional", "two-sided"],
        default="directional",
        help=(
            "Wilcoxon alternative for High vs Low: directional (one-sided hypothesis) "
            "or two-sided."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_hotspot_args()
    if args.no_bh_pvalue_correction:
        args.use_bh_pvalue_correction = False

    if args.activity == "any_detected" and args.phospho_split_mode != "observed_missing":
        raise SystemExit(
            "activity=any_detected requires --phospho-split-mode observed_missing "
            "(High=≥1 member detected, Low=all members missing)."
        )
    if args.phospho_split_mode == "detected_median" and args.activity != "mean_z":
        print(
            "Warning: detected_median is designed with --activity mean_z "
            "(Detected Low/High from mean_z median; Undetected = all members missing)."
        )

    hotspots, members = load_hotspot_tables(args.hotspot_dir, args.distance)
    print(
        f"Loaded hotspot catalog d={args.distance}: "
        f"{len(hotspots)} hotspots, {len(members)} members"
    )
    print(f"Activity={args.activity} | phospho_split_mode={args.phospho_split_mode} | test_alternative={args.test_alternative}")

    config = TempoConfig(
        linkedomics_base=args.linkedomics_base,
        chip_dir=args.chip_dir,
        signed_regulon_path=args.signed_regulon_path,
        idmapping_path=args.idmapping_path,
        prediction_output_dir=args.prediction_output_dir,
        import_prediction_filename=args.import_prediction_filename,
        export_prediction_filename=args.export_prediction_filename,
        output_dir=args.output_dir,
        known_positive_path=args.known_positive_path,
        ensembl_release=args.ensembl_release,
        species=args.species,
        cancer_types=list(args.cancer_types),
        directions=list(args.directions),
        max_missing_ratio=args.max_missing_ratio,
        min_nonzero_ratio=args.min_nonzero_ratio,
        max_nonzero_ratio=args.max_nonzero_ratio,
        chip_threshold=args.chip_threshold,
        min_chip_sample_frac=args.min_chip_sample_frac,
        chip_top_n=args.chip_top_n,
        signed_target_mode=args.signed_target_mode,
        phospho_group_frac=args.phospho_group_frac,
        phospho_split_mode=args.phospho_split_mode,
        min_group_samples=args.min_group_samples,
        min_box_points=args.min_box_points,
        dpi=args.dpi,
        random_iterations=args.random_iterations,
        random_seed=args.random_seed,
        max_random_gene_rows_per_group=args.max_random_gene_rows_per_group,
        bootstrap_iterations=args.bootstrap_iterations,
        confounder_analysis=args.confounder_analysis,
        abundance_primary_mode=args.abundance_primary_mode,
        phospho_value_mode=args.phospho_value_mode,
        adjustment_covariates=list(args.adjustment_covariates),
        purity_column=args.purity_column,
        min_samples_for_adjustment=args.min_samples_for_adjustment,
        use_bh_pvalue_correction=args.use_bh_pvalue_correction,
        test_alternative=args.test_alternative,
    )

    pipeline = HotspotTargetRegulationPipeline(
        config=config,
        hotspots=hotspots,
        members=members,
        activity_mode=args.activity,
        cluster_distance=args.distance,
    )
    pipeline.run_scan()


if __name__ == "__main__":
    main()
