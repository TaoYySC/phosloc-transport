#!/usr/bin/env python3
"""Build phospho-hotspot catalogs by distance-clustering Known/HC phosphosites.

Definition (anchor-only clustering):
  1) Clustering universe: Known and/or Predicted (HC) phosphosites from the
     FuncTransport summary table only — not unannotated / Cluster-only rows,
     and not every STY on the protein sequence.
     HC evidence ∈ {Predicted, Predicted;Cluster}.
  2) On each protein, single-linkage merge Known/HC sites when adjacent sites
     are within ≤ d aa (primary d=10; sensitivity d=15/30).
  3) Every resulting cluster is a hotspot, including singletons (a Known/HC
     site with no other Known/HC within d aa).
  4) Members = only the Known/HC sites in that cluster (size = 1, 2, 3, …).
  5) Annotate hotspot_class:
       known_containing / known_proximal / known_independent
  6) Nuclear / cytoplasmic gating for CPTAC uses *pure* per-site Localization
     (already screened), NOT mean Direction_score and NOT OR-any:
       has_any_nuclear_member / has_any_cytoplasmic_member = any matching member
       has_nuclear_member = ≥1 nuclear AND 0 cytoplasmic  (pure nuclear)
       has_cytoplasmic_member = ≥1 cytoplasmic AND 0 nuclear (pure cytoplasmic)
       direction_call = nuclear | cytoplasmic | unresolved (mixed / none)
  7) CPTAC activity (separate): mean z-score of all CPTAC-measured member
     phosphosites in the hotspot.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = (
    REPO
    / "import_export/results"
    / "tf_phos_site_FuncTransport_Direction_summary_with_PMID_annotation_v11_147pos_d3_platt.csv"
)
DOMAIN_CSV = REPO / "functional/data/features/domain_features_all_sty.csv"
NLS_CSV = REPO / "functional/data/features/nls_features_all_sty.csv"
NES_CSV = REPO / "functional/data/features/nes_features_all_sty.csv"
MOTIF1433_CSV = REPO / "functional/data/features/1433_features_all_sty.csv"
IDR_CSV = REPO / "functional/data/features/idr_features_all_sty.csv"
DEFAULT_IDMAPPING = (
    REPO / "cptac_analysis/data/source/3.idmapping/HUMAN_9606_idmapping.dat"
)
DEFAULT_LINKEDOMICS = (
    REPO / "cptac_analysis/data/source/1.cpatac/LinkedOmicsKB"
)
DEFAULT_CANCERS = [
    "BRCA", "CCRCC", "COAD", "GBM", "HNSCC",
    "LSCC", "LUAD", "OV", "PDAC", "UCEC",
]

ANNOTATED_EV = {"Known", "Cluster", "Predicted", "Predicted;Cluster"}
KNOWN_EV = {"Known"}
HC_EV = {"Predicted", "Predicted;Cluster"}
PROXIMAL_AA = 30
NUCLEAR_LOCALIZATION = {"Nuclear accumulation"}
CYTOPLASMIC_LOCALIZATION = {"Cytoplasmic redistribution"}
SITE_DIRECTION_THRESHOLD = 0.05


def parse_position(site: object) -> float:
    s = str(site).strip()
    m = re.fullmatch(r"[STY](\d+)", s, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+)", s)
    return float(m.group(1)) if m else np.nan


def parse_residue(site: object) -> str:
    s = str(site).strip().upper()
    if s and s[0] in "STY":
        return s[0]
    return ""


def site_index(acc: str, site: str) -> str:
    return f"{acc}_{site}"


def load_all_tf_sites(summary_path: Path) -> pd.DataFrame:
    """All scored STY sites on TF proteins (clustering universe)."""
    df = pd.read_csv(summary_path)
    df["protein_acc"] = df["protein_acc"].astype(str).str.strip()
    df["gene_name"] = df["gene_name"].astype(str).str.strip()
    df["site"] = df["site"].astype(str).str.strip()
    df["evidence"] = df["evidence"].astype(str).str.strip()
    df["position"] = df["site"].map(parse_position)
    df["residue"] = df["site"].map(parse_residue)
    df["INDEX"] = [site_index(a, s) for a, s in zip(df["protein_acc"], df["site"])]
    df = df.dropna(subset=["position", "residue"]).copy()
    df["position"] = df["position"].astype(int)
    df["FuncTransport_score"] = pd.to_numeric(df["FuncTransport_score"], errors="coerce")
    df["Direction_score"] = pd.to_numeric(df["Direction_score"], errors="coerce")
    return df.reset_index(drop=True)


def load_idmapping(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", names=["ACC_ID", "to", "ENSEMBL_GENE_ID"], dtype=str)
    df = df.query("to == 'Ensembl'").copy()
    df["ENSEMBL_GENE_ID"] = df["ENSEMBL_GENE_ID"].str.split(".").str[0]
    df = df.dropna(subset=["ACC_ID", "ENSEMBL_GENE_ID"]).drop_duplicates()
    return df[["ACC_ID", "ENSEMBL_GENE_ID"]].reset_index(drop=True)


def collect_cptac_observed_indices(
    sites: pd.DataFrame,
    linkedomics_base: Path,
    idmapping_path: Path,
    cancer_types: Sequence[str],
) -> Set[str]:
    """Return INDEX set of summary sites detected in any CPTAC phospho matrix."""
    idmap = load_idmapping(idmapping_path)
    # ACC may map to multiple ENSG; keep all
    acc_to_ensg: Dict[str, List[str]] = (
        idmap.groupby("ACC_ID")["ENSEMBL_GENE_ID"].apply(lambda s: sorted(set(s.astype(str)))).to_dict()
    )

    # Precompute possible CPTAC keys per INDEX
    index_to_keys: Dict[str, List[str]] = {}
    for _, row in sites.iterrows():
        acc = str(row["protein_acc"])
        site = str(row["site"])
        keys = [f"{ensg}|{site}" for ensg in acc_to_ensg.get(acc, [])]
        index_to_keys[str(row["INDEX"])] = keys

    all_keys_needed = {k for keys in index_to_keys.values() for k in keys}
    observed_keys: Set[str] = set()

    for cancer in cancer_types:
        path = (
            linkedomics_base
            / cancer
            / f"{cancer}_phospho_site_abundance_log2_reference_intensity_normalized_Tumor.txt"
        )
        if not path.exists():
            print(f"Warning: missing phospho matrix {path}")
            continue
        # Only need index; faster to read first column
        idx = pd.read_csv(path, sep="\t", usecols=[0], dtype=str).iloc[:, 0]
        parsed = []
        for p in idx:
            parts = str(p).split("|")
            if len(parts) >= 3:
                parsed.append(parts[0].split(".")[0] + "|" + parts[2])
            else:
                parsed.append(str(p))
        cancer_keys = set(parsed)
        hit = all_keys_needed & cancer_keys
        observed_keys |= hit
        print(f"  CPTAC {cancer}: {len(hit)} summary-mapped sites detected")

    observed_index: Set[str] = set()
    for index, keys in index_to_keys.items():
        if any(k in observed_keys for k in keys):
            observed_index.add(index)
    print(f"CPTAC-observed summary sites: {len(observed_index)}")
    return observed_index


def load_region_flags(indices: Sequence[str]) -> pd.DataFrame:
    idx = pd.Index(indices, name="INDEX")
    out = pd.DataFrame(index=idx)
    specs = [
        (DOMAIN_CSV, "MOTIF_Domain_DBD_Inside_Flag", "in_DBD"),
        (NLS_CSV, "MOTIF_NLS_Inside_Flag", "in_NLS"),
        (NES_CSV, "MOTIF_NES_Inside_Flag", "in_NES"),
        (MOTIF1433_CSV, "MOTIF_1433_Inside_Flag", "in_1433"),
    ]
    for path, col, name in specs:
        if not path.exists():
            out[name] = 0
            continue
        feat = pd.read_csv(path, usecols=["INDEX", col])
        feat = feat.drop_duplicates("INDEX").set_index("INDEX")[col]
        out[name] = pd.to_numeric(feat.reindex(idx), errors="coerce").fillna(0).astype(int)
    if IDR_CSV.exists():
        hdr = pd.read_csv(IDR_CSV, nrows=0).columns.tolist()
        idr_col = "MOTIF_IDR_Inside_Flag" if "MOTIF_IDR_Inside_Flag" in hdr else None
        if idr_col:
            feat = pd.read_csv(IDR_CSV, usecols=["INDEX", idr_col]).drop_duplicates("INDEX")
            out["in_IDR"] = (
                pd.to_numeric(feat.set_index("INDEX")[idr_col].reindex(idx), errors="coerce")
                .fillna(0)
                .astype(int)
            )
        else:
            out["in_IDR"] = 0
    else:
        out["in_IDR"] = 0
    return out.reset_index()


def cluster_positions(positions: List[int], distance: int) -> List[List[int]]:
    if not positions:
        return []
    pos = sorted(set(positions))
    groups: List[List[int]] = [[pos[0]]]
    for p in pos[1:]:
        if p - groups[-1][-1] <= distance:
            groups[-1].append(p)
        else:
            groups.append([p])
    return groups


def direction_call(scores: Iterable[float]) -> str:
    """Legacy mean Direction_score call (kept for diagnostics)."""
    vals = [float(x) for x in scores if pd.notna(x)]
    if not vals:
        return "unresolved"
    mean_s = float(np.mean(vals))
    if mean_s > 0.05:
        return "nuclear"
    if mean_s < -0.05:
        return "cytoplasmic"
    return "unresolved"


def pure_direction_call(has_nuclear: bool, has_cytoplasmic: bool) -> str:
    """Pure hotspot direction: all nuclear, all cytoplasmic, or unresolved."""
    if has_nuclear and not has_cytoplasmic:
        return "nuclear"
    if has_cytoplasmic and not has_nuclear:
        return "cytoplasmic"
    return "unresolved"


def top2_mean(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().sort_values(ascending=False)
    if vals.empty:
        return np.nan
    return float(vals.head(2).mean())


def annotate_site_flags(sites: pd.DataFrame) -> pd.DataFrame:
    """Add is_known / is_hc / is_observed / proximal / nuclear-cyto site flags."""
    out = sites.copy()
    known_pos_by_acc = (
        out.loc[out["evidence"].isin(KNOWN_EV)]
        .groupby("protein_acc")["position"]
        .apply(lambda s: sorted(set(int(x) for x in s)))
        .to_dict()
    )

    out["is_known"] = out["evidence"].isin(KNOWN_EV)
    out["is_hc"] = out["evidence"].isin(HC_EV)
    out["is_annotated_seed"] = out["evidence"].isin(ANNOTATED_EV)
    # observed: CPTAC or annotated evidence (Known/Cluster/HC)
    out["is_observed"] = out["cptac_observed"].astype(bool) | out["is_annotated_seed"]
    # Keep anchors: Known or Predicted (HC) only — hotspot = their neighborhood
    out["is_keep_anchor"] = out["is_known"] | out["is_hc"]

    loc = out.get("Localization_annotation", pd.Series("", index=out.index)).astype(str)
    out["is_nuclear_site"] = loc.isin(NUCLEAR_LOCALIZATION)
    out["is_cytoplasmic_site"] = loc.isin(CYTOPLASMIC_LOCALIZATION)
    # Fallback if localization blank but Direction_score already signed
    dir_s = pd.to_numeric(out["Direction_score"], errors="coerce")
    out.loc[~out["is_nuclear_site"] & ~out["is_cytoplasmic_site"] & dir_s.gt(SITE_DIRECTION_THRESHOLD), "is_nuclear_site"] = True
    out.loc[~out["is_nuclear_site"] & ~out["is_cytoplasmic_site"] & dir_s.lt(-SITE_DIRECTION_THRESHOLD), "is_cytoplasmic_site"] = True

    min_dist = []
    for _, row in out.iterrows():
        known_pos = known_pos_by_acc.get(str(row["protein_acc"]), [])
        if not known_pos:
            min_dist.append(np.inf)
        else:
            min_dist.append(min(abs(int(row["position"]) - k) for k in known_pos))
    out["min_dist_to_known"] = min_dist
    # Proximal candidate: near a Known, not itself Known, and observed/HC/Cluster
    # (annotation only; does not keep a hotspot by itself)
    out["is_proximal_candidate"] = (
        (~out["is_known"])
        & out["min_dist_to_known"].le(PROXIMAL_AA)
        & (out["is_hc"] | out["evidence"].eq("Cluster") | out["cptac_observed"].astype(bool))
    )
    out["is_qualifying_anchor"] = out["is_keep_anchor"]
    return out


def annotate_hotspot_class(members: pd.DataFrame, known_pos_by_acc: dict) -> str:
    if members["is_known"].any() if "is_known" in members.columns else members["evidence"].isin(KNOWN_EV).any():
        return "known_containing"
    acc = str(members["protein_acc"].iloc[0])
    known_positions = known_pos_by_acc.get(acc, [])
    if not known_positions:
        return "known_independent"
    member_pos = members["position"].astype(int).tolist()
    min_dist = min(abs(p - k) for p in member_pos for k in known_positions)
    if min_dist <= PROXIMAL_AA:
        return "known_proximal"
    return "known_independent"


def build_hotspots_for_distance(sites: pd.DataFrame, distance: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    known_pos_by_acc = (
        sites.loc[sites["evidence"].isin(KNOWN_EV)]
        .groupby("protein_acc")["position"]
        .apply(lambda s: sorted(set(int(x) for x in s)))
        .to_dict()
    )

    hotspot_rows = []
    member_rows = []
    # Cluster ONLY Known / HC anchors. Unannotated and Cluster-only sites do not
    # enter the clustering universe (and therefore cannot be hotspot members).
    # Singletons are valid hotspots when no other Known/HC lies within `distance`.
    anchors = sites.loc[sites["is_keep_anchor"].astype(bool)].copy()
    for acc, g in anchors.groupby("protein_acc", sort=True):
        g = g.sort_values(["position", "site"]).copy()
        groups = cluster_positions(g["position"].astype(int).tolist(), distance)
        h_idx = 0
        for group_pos in groups:
            mem = g[g["position"].isin(group_pos)].copy()
            if mem.empty:
                continue
            n_observed = int(mem["is_observed"].sum())
            h_idx += 1
            sites_sorted = mem.sort_values("position")["site"].astype(str).tolist()
            start_site = sites_sorted[0]
            end_site = sites_sorted[-1]
            hotspot_id = f"{acc}_H{h_idx}_{start_site}_{end_site}"
            gene = str(mem["gene_name"].iloc[0])
            family = str(mem["TF_family"].iloc[0]) if "TF_family" in mem.columns else ""

            # All members are Known/HC anchors; Direction annotation uses them directly.
            dir_mem = mem
            hclass = annotate_hotspot_class(mem, known_pos_by_acc)
            has_any_nuclear = bool(mem["is_nuclear_site"].any())
            has_any_cyto = bool(mem["is_cytoplasmic_site"].any())
            # CPTAC Import/Export gates: pure direction only (no mixed hotspots).
            has_nuclear = bool(has_any_nuclear and not has_any_cyto)
            has_cyto = bool(has_any_cyto and not has_any_nuclear)
            d_call = pure_direction_call(has_any_nuclear, has_any_cyto)
            mean_dir_call = direction_call(dir_mem["Direction_score"])
            score_mem = mem

            hotspot_rows.append(
                {
                    "hotspot_id": hotspot_id,
                    "hotspot_label": f"{gene}_{start_site}-{end_site}",
                    "protein_acc": acc,
                    "gene_name": gene,
                    "TF_family": family,
                    "n_members": int(len(mem)),
                    "n_observed": n_observed,
                    "n_cptac_observed": int(mem["cptac_observed"].sum()),
                    "n_qualifying_anchor": int(mem["is_keep_anchor"].sum()),
                    "member_sites": ";".join(sites_sorted),
                    "member_indices": ";".join(mem["INDEX"].astype(str).tolist()),
                    "start_position": int(mem["position"].min()),
                    "end_position": int(mem["position"].max()),
                    "span_aa": int(mem["position"].max() - mem["position"].min()),
                    "n_known": int(mem["is_known"].sum()),
                    "n_hc": int(mem["is_hc"].sum()),
                    "n_proximal_candidate": int(mem["is_proximal_candidate"].sum()),
                    "n_cluster": int((mem["evidence"] == "Cluster").sum()),
                    "n_nuclear_sites": int(mem["is_nuclear_site"].sum()),
                    "n_cytoplasmic_sites": int(mem["is_cytoplasmic_site"].sum()),
                    "has_known": bool(mem["is_known"].any()),
                    "has_predicted": bool(mem["is_hc"].any()),
                    "has_proximal_candidate": bool(mem["is_proximal_candidate"].any()),
                    "has_any_nuclear_member": has_any_nuclear,
                    "has_any_cytoplasmic_member": has_any_cyto,
                    "has_nuclear_member": has_nuclear,
                    "has_cytoplasmic_member": has_cyto,
                    "direction_call": d_call,
                    "mean_Direction_call": mean_dir_call,
                    "mean_Direction_score": float(pd.to_numeric(dir_mem["Direction_score"], errors="coerce").mean()),
                    "max_Direction_score": float(pd.to_numeric(dir_mem["Direction_score"], errors="coerce").max()),
                    "mean_FuncTransport_score": float(
                        pd.to_numeric(score_mem["FuncTransport_score"], errors="coerce").mean()
                    ),
                    "max_FuncTransport_score": float(
                        pd.to_numeric(score_mem["FuncTransport_score"], errors="coerce").max()
                    ),
                    "top2_mean_FuncTransport_score": top2_mean(score_mem["FuncTransport_score"]),
                    "top2_mean_Direction_score": top2_mean(dir_mem["Direction_score"]),
                    "hotspot_class": hclass,
                    "cluster_distance": int(distance),
                    "in_DBD": int(mem["in_DBD"].max()) if "in_DBD" in mem else 0,
                    "in_NLS": int(mem["in_NLS"].max()) if "in_NLS" in mem else 0,
                    "in_NES": int(mem["in_NES"].max()) if "in_NES" in mem else 0,
                    "in_1433": int(mem["in_1433"].max()) if "in_1433" in mem else 0,
                    "in_IDR": int(mem["in_IDR"].max()) if "in_IDR" in mem else 0,
                    "Region_union": ";".join(
                        sorted(
                            {
                                r
                                for r in mem.get("Region", pd.Series(dtype=str)).astype(str)
                                if r not in ("—", "-", "nan", "")
                            }
                        )
                    ),
                }
            )
            for _, row in mem.iterrows():
                member_rows.append(
                    {
                        "hotspot_id": hotspot_id,
                        "hotspot_label": f"{gene}_{start_site}-{end_site}",
                        "protein_acc": acc,
                        "gene_name": gene,
                        "INDEX": row["INDEX"],
                        "site": row["site"],
                        "residue": row["residue"],
                        "position": int(row["position"]),
                        "evidence": row["evidence"],
                        "cptac_observed": bool(row["cptac_observed"]),
                        "is_observed": bool(row["is_observed"]),
                        "is_known": bool(row["is_known"]),
                        "is_hc": bool(row["is_hc"]),
                        "is_proximal_candidate": bool(row["is_proximal_candidate"]),
                        "is_qualifying_anchor": bool(row["is_qualifying_anchor"]),
                        "is_nuclear_site": bool(row["is_nuclear_site"]),
                        "is_cytoplasmic_site": bool(row["is_cytoplasmic_site"]),
                        "min_dist_to_known": float(row["min_dist_to_known"])
                        if np.isfinite(row["min_dist_to_known"])
                        else np.nan,
                        "FuncTransport_score": row["FuncTransport_score"],
                        "Direction_score": row["Direction_score"],
                        "Localization_annotation": row.get("Localization_annotation", ""),
                        "Region": row.get("Region", ""),
                        "hotspot_class": hclass,
                        "direction_call": d_call,
                        "has_any_nuclear_member": has_any_nuclear,
                        "has_any_cytoplasmic_member": has_any_cyto,
                        "has_nuclear_member": has_nuclear,
                        "has_cytoplasmic_member": has_cyto,
                        "cluster_distance": int(distance),
                    }
                )

    hotspots = pd.DataFrame(hotspot_rows)
    members = pd.DataFrame(member_rows)
    if not hotspots.empty:
        hotspots = hotspots.sort_values(
            ["gene_name", "start_position", "hotspot_id"]
        ).reset_index(drop=True)
        members = members.sort_values(["hotspot_id", "position"]).reset_index(drop=True)
    return hotspots, members


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO
        / "cptac_analysis/results/import_target_regulation_hotspot_v11_147pos_d3_platt/hotspots",
    )
    parser.add_argument("--distances", type=int, nargs="+", default=[10, 15, 30])
    parser.add_argument("--linkedomics-base", type=Path, default=DEFAULT_LINKEDOMICS)
    parser.add_argument("--idmapping-path", type=Path, default=DEFAULT_IDMAPPING)
    parser.add_argument("--cancer-types", nargs="+", default=DEFAULT_CANCERS)
    parser.add_argument(
        "--skip-cptac-observed",
        action="store_true",
        help="Do not scan CPTAC matrices; observed = annotated evidence only",
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Archive previous catalogs before rewriting under the new anchor-only definition
    if (out_dir / "hotspots_d10.csv").exists():
        from datetime import datetime

        archive = out_dir / f"archive_before_anchor_only_clustering_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        archive.mkdir(parents=True, exist_ok=True)
        for p in list(out_dir.glob("hotspot*.csv")) + list(out_dir.glob("hotspot_build_config.json")):
            if p.is_file():
                p.replace(archive / p.name)
        print(f"Archived previous hotspot catalogs to {archive}")

    sites = load_all_tf_sites(args.summary)
    print(f"Loaded {len(sites)} TF phosphosites from summary (annotation table)")

    if args.skip_cptac_observed:
        sites["cptac_observed"] = False
    else:
        print("Scanning CPTAC phospho matrices for observed sites...")
        observed = collect_cptac_observed_indices(
            sites,
            args.linkedomics_base,
            args.idmapping_path,
            args.cancer_types,
        )
        sites["cptac_observed"] = sites["INDEX"].astype(str).isin(observed)

    region = load_region_flags(sites["INDEX"].astype(str).tolist())
    sites = sites.merge(region, on="INDEX", how="left")
    for c in ["in_DBD", "in_NLS", "in_NES", "in_1433", "in_IDR"]:
        sites[c] = sites[c].fillna(0).astype(int)

    sites = annotate_site_flags(sites)
    n_anchors = int(sites["is_keep_anchor"].sum())
    print(
        "Site flags:",
        f"known={int(sites.is_known.sum())}",
        f"hc={int(sites.is_hc.sum())}",
        f"anchors(Known|HC)={n_anchors}",
        f"proximal_cand={int(sites.is_proximal_candidate.sum())}",
        f"observed={int(sites.is_observed.sum())}",
        f"cptac={int(sites.cptac_observed.sum())}",
    )
    print(f"Clustering universe size: {n_anchors} Known/HC sites")

    summary_stats = {}
    for d in args.distances:
        hotspots, members = build_hotspots_for_distance(sites, d)
        hotspots.to_csv(out_dir / f"hotspots_d{d}.csv", index=False)
        members.to_csv(out_dir / f"hotspot_members_d{d}.csv", index=False)
        size_bins = {}
        if len(hotspots):
            nm = hotspots["n_members"].astype(int)
            size_bins = {
                "1": int((nm == 1).sum()),
                "2": int((nm == 2).sum()),
                "3": int((nm == 3).sum()),
                "≥4": int((nm >= 4).sum()),
            }
        summary_stats[f"d{d}"] = {
            "n_hotspots": int(len(hotspots)),
            "n_members": int(len(members)),
            "n_nuclear": int((hotspots["direction_call"] == "nuclear").sum()) if len(hotspots) else 0,
            "n_cytoplasmic": int((hotspots["direction_call"] == "cytoplasmic").sum()) if len(hotspots) else 0,
            "by_class": hotspots["hotspot_class"].value_counts().to_dict() if len(hotspots) else {},
            "by_n_members_bin": size_bins,
            "median_n_members": float(hotspots["n_members"].median()) if len(hotspots) else np.nan,
            "median_n_observed": float(hotspots["n_observed"].median()) if len(hotspots) else np.nan,
        }
        print(f"d={d}: {len(hotspots)} hotspots, {len(members)} members | size bins {size_bins}")
        if len(hotspots):
            print(hotspots["hotspot_class"].value_counts().to_string())
            print(hotspots["direction_call"].value_counts().to_string())
            print(hotspots["n_members"].value_counts().sort_index().head(12).to_string())

    meta = {
        "definition": (
            "Cluster Known and/or Predicted (HC) phosphosites from the TF summary "
            "table with single-linkage gap≤distance. Every cluster is a hotspot, "
            "including singletons. Members = Known/HC sites only (unannotated and "
            "Cluster-only sites are excluded from clustering). CPTAC Import gating "
            "uses pure has_nuclear_member (≥1 nuclear site AND 0 cytoplasmic), not "
            "OR-any and not mean Direction_score. CPTAC activity = mean z of all "
            "CPTAC-measured member phosphosites. Size strata: n_members = 1 / 2 / 3 / ≥4."
        ),
        "summary_path": str(args.summary),
        "clustering_universe": "Known_or_Predicted_HC_only",
        "observed_definition": (
            "CPTAC-detected in any cohort OR evidence in "
            "{Known, Cluster, Predicted, Predicted;Cluster}"
        ),
        "keep_rule": (
            "cluster Known/HC anchors only; singletons allowed; members = "
            "Known/HC sites in the distance cluster"
        ),
        "import_gate": (
            "has_nuclear_member = pure nuclear "
            "(≥1 Localization=Nuclear accumulation or Direction_score>0.05; "
            "AND no cytoplasmic members)"
        ),
        "export_gate": (
            "has_cytoplasmic_member = pure cytoplasmic "
            "(≥1 Localization=Cytoplasmic redistribution or Direction_score<-0.05; "
            "AND no nuclear members)"
        ),
        "direction_call": "pure nuclear|cytoplasmic|unresolved (mixed hotspots unresolved)",
        "proximal_candidate_aa": PROXIMAL_AA,
        "hc_evidence": sorted(HC_EV),
        "distances": list(args.distances),
        "cptac_observed_scanned": (not args.skip_cptac_observed),
        "cancer_types": list(args.cancer_types),
        "stats": summary_stats,
    }
    (out_dir / "hotspot_build_config.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"Wrote catalogs to {out_dir}")


if __name__ == "__main__":
    main()
