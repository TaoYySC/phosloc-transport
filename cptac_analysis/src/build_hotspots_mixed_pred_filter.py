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
    / "import_export/results"
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


def parse_position(site: object) -> float:
    m = re.fullmatch(r"[STY](\d+)", str(site).strip(), flags=re.IGNORECASE)
    return float(m.group(1)) if m else np.nan


def load_anchors(summary: Path) -> pd.DataFrame:
    df = pd.read_csv(summary)
    for c in ["protein_acc", "gene_name", "site", "evidence", "PMID"]:
        df[c] = df[c].astype(str).str.strip()
    df["FuncTransport_score"] = pd.to_numeric(df["FuncTransport_score"], errors="coerce")
    df["position"] = df["site"].map(parse_position)
    df = df.dropna(subset=["position"]).copy()
    df["position"] = df["position"].astype(int)
    df["is_known"] = ~df["PMID"].isin(["", "-", "nan", "None", "NaN"])
    df["is_predicted"] = df["evidence"].str.contains("Predicted", regex=False)
    df["is_cluster_ev"] = df["evidence"].str.contains("Cluster", regex=False)
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


def build_catalog(anchors: pd.DataFrame, adj_max: int, span_max: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for acc, g in anchors.groupby("protein_acc", sort=True):
        for mem in cluster_sorted(g, adj_max, span_max):
            sites = mem.sort_values("position")["site"].tolist()
            start, end = int(mem["position"].min()), int(mem["position"].max())
            pos = sorted(mem["position"].unique())
            pred_scores = mem.loc[mem["is_predicted"], "FuncTransport_score"]
            rows.append(
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
                    "hotspot_type": classify(mem),
                }
            )
    all_hs = pd.DataFrame(rows)
    kept = all_hs[all_hs["hotspot_type"].notna()].copy()
    kept = kept.sort_values(["hotspot_type", "gene_name", "start_position"]).reset_index(drop=True)
    return all_hs, kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--adj-max", type=int, default=ADJ_MAX)
    parser.add_argument("--span-max", type=int, default=SPAN_MAX)
    args = parser.parse_args()

    anchors = load_anchors(args.summary)
    all_hs, kept = build_catalog(anchors, args.adj_max, args.span_max)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_hs.to_csv(args.output_dir / "all_clusters_adj15_span40_unfiltered.csv", index=False)
    kept.to_csv(args.output_dir / "hotspots_adj15_span40_filtered.csv", index=False)

    obs = {
        "total": int(len(kept)),
        "Mixed evidence": int((kept["hotspot_type"] == "Mixed evidence").sum()),
        "Predicted candidate": int((kept["hotspot_type"] == "Predicted candidate").sum()),
        "n_TF": int(kept["gene_name"].nunique()),
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
    }
    (args.output_dir / "verification_summary.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(obs, indent=2))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
