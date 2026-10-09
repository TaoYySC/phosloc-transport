#!/usr/bin/env python3
"""Re-annotate an existing hotspot catalog with pure nuclear/cytoplasmic gates.

Copies source catalog files, then sets:
  has_any_*_member = OR of site flags (diagnostic)
  has_nuclear_member = ≥1 nuclear AND 0 cytoplasmic
  has_cytoplasmic_member = ≥1 cytoplasmic AND 0 nuclear
  direction_call = nuclear | cytoplasmic | unresolved
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SRC = (
    REPO
    / "cptac_analysis/results/import_target_regulation_hotspot_v11_147pos_d3_platt"
    / "hotspots_mixed_pred_filter"
)
DEFAULT_OUT = (
    REPO
    / "cptac_analysis/results/import_target_regulation_hotspot_v11_147pos_d3_platt"
    / "hotspots_mixed_pred_filter_pure_direction"
)


def pure_call(has_any_n: bool, has_any_c: bool) -> str:
    if has_any_n and not has_any_c:
        return "nuclear"
    if has_any_c and not has_any_n:
        return "cytoplasmic"
    return "unresolved"


def reannotate(hotspots: pd.DataFrame, members: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hs = hotspots.copy()
    mem = members.copy()

    # Prefer explicit site flags; fall back to Localization_annotation.
    if "is_nuclear_site" not in mem.columns:
        loc = mem.get("Localization_annotation", pd.Series("", index=mem.index)).astype(str)
        mem["is_nuclear_site"] = loc.eq("Nuclear accumulation")
        mem["is_cytoplasmic_site"] = loc.str.contains("Cytoplasmic", na=False)

    agg = (
        mem.groupby("hotspot_id", sort=False)
        .agg(
            n_nuclear_sites=("is_nuclear_site", lambda s: int(s.astype(bool).sum())),
            n_cytoplasmic_sites=("is_cytoplasmic_site", lambda s: int(s.astype(bool).sum())),
            has_any_nuclear_member=("is_nuclear_site", lambda s: bool(s.astype(bool).any())),
            has_any_cytoplasmic_member=("is_cytoplasmic_site", lambda s: bool(s.astype(bool).any())),
        )
        .reset_index()
    )
    agg["has_nuclear_member"] = agg["has_any_nuclear_member"] & ~agg["has_any_cytoplasmic_member"]
    agg["has_cytoplasmic_member"] = agg["has_any_cytoplasmic_member"] & ~agg["has_any_nuclear_member"]
    agg["direction_call"] = [
        pure_call(bool(n), bool(c))
        for n, c in zip(agg["has_any_nuclear_member"], agg["has_any_cytoplasmic_member"])
    ]

    drop_cols = [
        c
        for c in [
            "n_nuclear_sites",
            "n_cytoplasmic_sites",
            "has_any_nuclear_member",
            "has_any_cytoplasmic_member",
            "has_nuclear_member",
            "has_cytoplasmic_member",
            "direction_call",
        ]
        if c in hs.columns
    ]
    hs = hs.drop(columns=drop_cols).merge(agg, on="hotspot_id", how="left")

    mem_drop = [
        c
        for c in [
            "has_any_nuclear_member",
            "has_any_cytoplasmic_member",
            "has_nuclear_member",
            "has_cytoplasmic_member",
            "direction_call",
        ]
        if c in mem.columns
    ]
    mem = mem.drop(columns=mem_drop).merge(
        agg[
            [
                "hotspot_id",
                "has_any_nuclear_member",
                "has_any_cytoplasmic_member",
                "has_nuclear_member",
                "has_cytoplasmic_member",
                "direction_call",
            ]
        ],
        on="hotspot_id",
        how="left",
    )
    return hs, mem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-dir", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--distance", type=int, default=15)
    args = parser.parse_args()

    src = args.src_dir
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # Copy ancillary filter/verification files unchanged.
    for name in [
        "all_clusters_adj15_span40_unfiltered.csv",
        "hotspots_adj15_span40_filtered.csv",
        "verification_summary.json",
    ]:
        src_f = src / name
        if src_f.exists():
            shutil.copy2(src_f, out / name)

    hs_path = src / f"hotspots_d{args.distance}.csv"
    mem_path = src / f"hotspot_members_d{args.distance}.csv"
    hs, mem = reannotate(pd.read_csv(hs_path), pd.read_csv(mem_path))
    hs.to_csv(out / f"hotspots_d{args.distance}.csv", index=False)
    mem.to_csv(out / f"hotspot_members_d{args.distance}.csv", index=False)

    n_imp = int(hs["has_nuclear_member"].astype(bool).sum())
    n_exp = int(hs["has_cytoplasmic_member"].astype(bool).sum())
    n_unres = int((hs["direction_call"].astype(str) == "unresolved").sum())
    meta = {
        "source_catalog": str(src),
        "direction_gate": "pure",
        "definition": (
            "Same Mixed+Predicted candidate membership as source; "
            "Import gate = pure nuclear (any nuclear, zero cytoplasmic); "
            "Export gate = pure cytoplasmic (any cytoplasmic, zero nuclear)."
        ),
        "n_hotspots": int(len(hs)),
        "type_counts": hs["hotspot_type"].value_counts().to_dict()
        if "hotspot_type" in hs.columns
        else {},
        "n_TF": int(hs["gene_name"].nunique()) if "gene_name" in hs.columns else None,
        "n_nuclear_gate_pure": n_imp,
        "n_cytoplasmic_gate_pure": n_exp,
        "n_unresolved_mixed_or_none": n_unres,
        "direction_call_counts": hs["direction_call"].value_counts().to_dict(),
        "distance_tag": args.distance,
    }
    (out / "cptac_catalog_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
