#!/usr/bin/env python3
"""Build filtered hotspots: Mixed evidence + Predicted candidate.

Rules (verified to yield 99 / 48 / 51 on 85 TFs for v11_147pos_d3_platt):
  1) Sites: Known = PMID not empty and not '-';
             Predicted = evidence contains 'Predicted'
             Ordinary observed sites are excluded from clustering.
  2) Within protein_acc, sort by position; break cluster if adjacent gap > 15
     OR span would exceed 40 aa.
  3) Mixed evidence: >=1 Known and >=1 Predicted.
  4) Predicted candidate: 0 Known, >=3 Predicted, mean Predicted
     FuncTransport_score >= 0.6.

Writes both filter summaries and downstream CPTAC inputs:
  - all_clusters_adj15_span40_unfiltered.csv
  - hotspots_adj15_span40_filtered.csv
  - hotspots_d{adj}.csv
  - hotspot_members_d{adj}.csv
  - verification_summary.json / cptac_catalog_config.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = (
    REPO
    / "import_export/data/precomputed/1_transport_classifier_results"
    / "tf_phos_site_FuncTransport_Direction_summary_with_PMID_annotation_v11_147pos_d3_platt.csv"
)
DEFAULT_OUT = (
    REPO
    / "cptac_analysis/results/import_target_regulation_hotspot_v11_147pos_d3_platt"
    / "hotspots_mixed_pred_filter"
)

ADJ_MAX = 15
SPAN_MAX = 40
PRED_MEAN_MIN = 0.6
PRED_MIN_N = 3
PROXIMAL_AA = 15


def parse_position(site: object) -> float:
    m = re.fullmatch(r"[STY](\d+)", str(site).strip(), flags=re.IGNORECASE)
    return float(m.group(1)) if m else np.nan


def load_anchors(summary: Path) -> pd.DataFrame:
    df = pd.read_csv(summary)
    for c in ["protein_acc", "gene_name", "site", "evidence", "PMID"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column in summary: {c}")
        df[c] = df[c].astype(str).str.strip()
    for optional in ["TF_family", "Localization_annotation", "Region", "Annotation_info"]:
        if optional not in df.columns:
            df[optional] = ""
        else:
            df[optional] = df[optional].astype(str)
    df["FuncTransport_score"] = pd.to_numeric(df["FuncTransport_score"], errors="coerce")
    df["Direction_score"] = pd.to_numeric(df.get("Direction_score"), errors="coerce")
    df["position"] = df["site"].map(parse_position)
    df = df.dropna(subset=["position"]).copy()
    df["position"] = df["position"].astype(int)
    df["residue"] = df["site"].astype(str).str[0].str.upper()
    df["INDEX"] = df["protein_acc"].astype(str) + "_" + df["site"].astype(str)
    df["is_known"] = ~df["PMID"].isin(["", "-", "nan", "None", "NaN"])
    df["is_predicted"] = df["evidence"].str.contains("Predicted", regex=False)
    df["is_hc"] = df["is_predicted"].astype(bool)
    df["is_cluster_ev"] = df["evidence"].str.contains("Cluster", regex=False)
    loc = df["Localization_annotation"].astype(str)
    df["is_nuclear_site"] = loc.eq("Nuclear accumulation")
    df["is_cytoplasmic_site"] = loc.str.contains("Cytoplasmic", na=False)
    df["cptac_observed"] = False
    df["is_observed"] = True
    return df.loc[df["is_known"] | df["is_predicted"]].copy()


def cluster_sorted(g: pd.DataFrame, adj_max: int, span_max: int) -> list[pd.DataFrame]:
    g = g.sort_values(["position", "site"]).reset_index(drop=True)
    if g.empty:
        return []
    clusters: list[list[int]] = []
    cur = [0]
    for i in range(1, len(g)):
        gap = int(g.at[i, "position"] - g.at[cur[-1], "position"])
        span = int(g.at[i, "position"] - g.at[cur[0], "position"])
        if gap > adj_max or span > span_max:
            clusters.append(cur)
            cur = [i]
        else:
            cur.append(i)
    clusters.append(cur)
    return [g.iloc[ix].copy() for ix in clusters]


def classify(mem: pd.DataFrame) -> str | None:
    n_known = int(mem["is_known"].sum())
    n_pred = int(mem["is_predicted"].sum())
    pred_scores = mem.loc[mem["is_predicted"], "FuncTransport_score"]
    mean_pred = float(pred_scores.mean()) if len(pred_scores) else np.nan
    if n_known >= 1 and n_pred >= 1:
        return "Mixed evidence"
    if (
        n_known == 0
        and n_pred >= PRED_MIN_N
        and pd.notna(mean_pred)
        and mean_pred >= PRED_MEAN_MIN
    ):
        return "Predicted candidate"
    return None


def annotate_hotspot_class(mem: pd.DataFrame, known_pos_by_acc: dict) -> str:
    if bool(mem["is_known"].any()):
        return "known_containing"
    acc = str(mem["protein_acc"].iloc[0])
    known_positions = known_pos_by_acc.get(acc, [])
    if not known_positions:
        return "known_independent"
    member_pos = mem["position"].astype(int).tolist()
    min_dist = min(abs(p - k) for p in member_pos for k in known_positions)
    if min_dist <= PROXIMAL_AA:
        return "known_proximal"
    return "known_independent"


def top2_mean(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna().sort_values(ascending=False)
    if vals.empty:
        return float("nan")
    return float(vals.head(2).mean())


def pure_direction_call(has_any_n: bool, has_any_c: bool) -> str:
    if has_any_n and not has_any_c:
        return "nuclear"
    if has_any_c and not has_any_n:
        return "cytoplasmic"
    return "unresolved"


def region_flags(mem: pd.DataFrame) -> dict:
    region = mem.get("Region", pd.Series(dtype=str)).astype(str)
    joined = ";".join(region.tolist()).upper()
    return {
        "in_DBD": int("DBD" in joined),
        "in_NLS": int("NLS" in joined),
        "in_NES": int("NES" in joined),
        "in_1433": int("14-3-3" in joined or "1433" in joined),
        "in_IDR": int("IDR" in joined),
        "Region_union": ";".join(
            sorted({r for r in region if r not in ("—", "-", "nan", "", "None")})
        ),
    }


def build_catalog(
    anchors: pd.DataFrame, adj_max: int, span_max: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return all_clusters, kept_summary, hotspots_d, hotspot_members_d."""
    known_pos_by_acc = (
        anchors.loc[anchors["is_known"]]
        .groupby("protein_acc")["position"]
        .apply(lambda s: sorted(set(int(x) for x in s)))
        .to_dict()
    )

    summary_rows = []
    hotspot_rows = []
    member_rows = []

    for acc, g in anchors.groupby("protein_acc", sort=True):
        h_idx = 0
        for mem in cluster_sorted(g, adj_max, span_max):
            sites = mem.sort_values("position")["site"].tolist()
            start, end = int(mem["position"].min()), int(mem["position"].max())
            pos = sorted(mem["position"].unique())
            pred_scores = mem.loc[mem["is_predicted"], "FuncTransport_score"]
            htype = classify(mem)
            summary_rows.append(
                {
                    "protein_acc": acc,
                    "gene_name": str(mem["gene_name"].iloc[0]),
                    "start_site": sites[0],
                    "end_site": sites[-1],
                    "start_position": start,
                    "end_position": end,
                    "span_aa": end - start,
                    "n_sites": int(len(mem)),
                    "n_known": int(mem["is_known"].sum()),
                    "n_predicted": int(mem["is_predicted"].sum()),
                    "n_cluster": int(mem["is_cluster_ev"].sum()),
                    "mean_predicted_FuncTransport": float(pred_scores.mean())
                    if len(pred_scores)
                    else np.nan,
                    "max_FuncTransport": float(mem["FuncTransport_score"].max()),
                    "member_sites": ";".join(sites),
                    "max_adjacent_gap": int(max(np.diff(pos))) if len(pos) > 1 else 0,
                    "hotspot_type": htype,
                }
            )
            if htype is None:
                continue

            h_idx += 1
            gene = str(mem["gene_name"].iloc[0])
            family = str(mem["TF_family"].iloc[0]) if "TF_family" in mem.columns else ""
            start_site, end_site = sites[0], sites[-1]
            hotspot_id = f"{acc}_H{h_idx}_{start_site}_{end_site}"
            hotspot_label = f"{gene}_{start_site}-{end_site}"
            hclass = annotate_hotspot_class(mem, known_pos_by_acc)

            # Annotate proximal distance for members
            known_positions = known_pos_by_acc.get(acc, [])
            min_dists = []
            for p in mem["position"].astype(int):
                if not known_positions:
                    min_dists.append(np.nan)
                else:
                    min_dists.append(float(min(abs(p - k) for k in known_positions)))
            mem = mem.copy()
            mem["min_dist_to_known"] = min_dists
            mem["is_proximal_candidate"] = (
                (~mem["is_known"])
                & mem["min_dist_to_known"].le(PROXIMAL_AA)
                & mem["is_predicted"]
            )
            mem["is_qualifying_anchor"] = True

            has_any_nuclear = bool(mem["is_nuclear_site"].any())
            has_any_cyto = bool(mem["is_cytoplasmic_site"].any())
            # OR-style gates here; pure gates applied by reannotate_hotspot_pure_direction.py
            has_nuclear = has_any_nuclear
            has_cyto = has_any_cyto
            d_call = pure_direction_call(has_any_nuclear, has_any_cyto)
            flags = region_flags(mem)

            hotspot_rows.append(
                {
                    "hotspot_id": hotspot_id,
                    "hotspot_label": hotspot_label,
                    "protein_acc": acc,
                    "gene_name": gene,
                    "TF_family": family,
                    "n_members": int(len(mem)),
                    "n_observed": int(len(mem)),
                    "n_cptac_observed": 0,
                    "n_qualifying_anchor": int(len(mem)),
                    "member_sites": ";".join(sites),
                    "member_indices": ";".join(mem["INDEX"].astype(str).tolist()),
                    "start_position": start,
                    "end_position": end,
                    "span_aa": end - start,
                    "n_known": int(mem["is_known"].sum()),
                    "n_hc": int(mem["is_hc"].sum()),
                    "n_proximal_candidate": int(mem["is_proximal_candidate"].sum()),
                    "n_cluster": int(mem["is_cluster_ev"].sum()),
                    "n_nuclear_sites": int(mem["is_nuclear_site"].sum()),
                    "n_cytoplasmic_sites": int(mem["is_cytoplasmic_site"].sum()),
                    "has_known": bool(mem["is_known"].any()),
                    "has_predicted": bool(mem["is_predicted"].any()),
                    "has_proximal_candidate": bool(mem["is_proximal_candidate"].any()),
                    "has_nuclear_member": has_nuclear,
                    "has_cytoplasmic_member": has_cyto,
                    "direction_call": d_call,
                    "mean_Direction_score": float(
                        pd.to_numeric(mem["Direction_score"], errors="coerce").mean()
                    ),
                    "max_Direction_score": float(
                        pd.to_numeric(mem["Direction_score"], errors="coerce").max()
                    ),
                    "mean_FuncTransport_score": float(
                        pd.to_numeric(mem["FuncTransport_score"], errors="coerce").mean()
                    ),
                    "max_FuncTransport_score": float(
                        pd.to_numeric(mem["FuncTransport_score"], errors="coerce").max()
                    ),
                    "top2_mean_FuncTransport_score": top2_mean(mem["FuncTransport_score"]),
                    "top2_mean_Direction_score": top2_mean(mem["Direction_score"]),
                    "hotspot_class": hclass,
                    "hotspot_type": htype,
                    "cluster_distance": int(adj_max),
                    "cluster_span_max": int(span_max),
                    **flags,
                }
            )
            for _, row in mem.iterrows():
                member_rows.append(
                    {
                        "hotspot_id": hotspot_id,
                        "hotspot_label": hotspot_label,
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
                        "is_qualifying_anchor": True,
                        "is_nuclear_site": bool(row["is_nuclear_site"]),
                        "is_cytoplasmic_site": bool(row["is_cytoplasmic_site"]),
                        "min_dist_to_known": row["min_dist_to_known"],
                        "FuncTransport_score": row["FuncTransport_score"],
                        "Direction_score": row["Direction_score"],
                        "Localization_annotation": row.get("Localization_annotation", ""),
                        "Region": row.get("Region", ""),
                        "hotspot_class": hclass,
                        "hotspot_type": htype,
                        "direction_call": d_call,
                        "has_nuclear_member": has_nuclear,
                        "has_cytoplasmic_member": has_cyto,
                        "cluster_distance": int(adj_max),
                    }
                )

    all_hs = pd.DataFrame(summary_rows)
    kept = all_hs[all_hs["hotspot_type"].notna()].copy()
    kept = kept.sort_values(["hotspot_type", "gene_name", "start_position"]).reset_index(drop=True)
    hotspots = pd.DataFrame(hotspot_rows)
    members = pd.DataFrame(member_rows)
    if not hotspots.empty:
        hotspots = hotspots.sort_values(["hotspot_type", "gene_name", "start_position"]).reset_index(
            drop=True
        )
    return all_hs, kept, hotspots, members


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--adj-max", type=int, default=ADJ_MAX)
    parser.add_argument("--span-max", type=int, default=SPAN_MAX)
    args = parser.parse_args()

    anchors = load_anchors(args.summary)
    all_hs, kept, hotspots, members = build_catalog(anchors, args.adj_max, args.span_max)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_hs.to_csv(args.output_dir / "all_clusters_adj15_span40_unfiltered.csv", index=False)
    kept.to_csv(args.output_dir / "hotspots_adj15_span40_filtered.csv", index=False)

    dtag = int(args.adj_max)
    hotspots.to_csv(args.output_dir / f"hotspots_d{dtag}.csv", index=False)
    members.to_csv(args.output_dir / f"hotspot_members_d{dtag}.csv", index=False)

    obs = {
        "total": int(len(kept)),
        "Mixed evidence": int((kept["hotspot_type"] == "Mixed evidence").sum()),
        "Predicted candidate": int((kept["hotspot_type"] == "Predicted candidate").sum()),
        "n_TF": int(kept["gene_name"].nunique()) if len(kept) else 0,
        "n_members_rows": int(len(members)),
    }
    meta = {
        "summary": str(args.summary),
        "adj_max": args.adj_max,
        "span_max": args.span_max,
        "observed": obs,
        "claimed_reference": {
            "total": 99,
            "Mixed evidence": 48,
            "Predicted candidate": 51,
            "n_TF": 85,
        },
        "downstream_files": [
            f"hotspots_d{dtag}.csv",
            f"hotspot_members_d{dtag}.csv",
        ],
    }
    (args.output_dir / "verification_summary.json").write_text(json.dumps(meta, indent=2) + "\n")
    config = {
        "definition": "mixed_pred_filter_adj15_span40",
        "n_hotspots": obs["total"],
        "type_counts": {
            "Predicted candidate": obs["Predicted candidate"],
            "Mixed evidence": obs["Mixed evidence"],
        },
        "n_TF": obs["n_TF"],
        "n_nuclear_gate": int(hotspots["has_nuclear_member"].astype(bool).sum())
        if len(hotspots)
        else 0,
        "distance_tag": dtag,
        "span_max": int(args.span_max),
    }
    (args.output_dir / "cptac_catalog_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(obs, indent=2))
    print(f"Wrote {args.output_dir}")
    print(f"  hotspots_d{dtag}.csv ({len(hotspots)} rows)")
    print(f"  hotspot_members_d{dtag}.csv ({len(members)} rows)")


if __name__ == "__main__":
    main()
