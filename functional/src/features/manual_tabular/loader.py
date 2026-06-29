from pathlib import Path
import pandas as pd


FEATURE_DIR = Path(__file__).resolve().parents[3] / "data" / "features"

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


def load_all_feature_tables():
    tables = {}
    for family_name, file_name in FEATURE_TABLE_FILES.items():
        tables[family_name] = pd.read_csv(FEATURE_DIR / file_name)
    return tables


class ManualTabularLoader:
    def __init__(self, index_col="INDEX"):
        self.index_col = index_col

    def merge_all(self, sample_df: pd.DataFrame) -> pd.DataFrame:
        out_df = sample_df.copy()
        out_df[self.index_col] = out_df[self.index_col].astype(str).str.strip()

        feature_tables = load_all_feature_tables()

        for family_name, feat_df in feature_tables.items():
            feat_df = feat_df.copy()

            if self.index_col not in feat_df.columns:
                raise ValueError(
                    f"Feature table '{family_name}' missing index column '{self.index_col}'"
                )

            feat_df[self.index_col] = feat_df[self.index_col].astype(str).str.strip()
            feat_df = feat_df.drop_duplicates(subset=[self.index_col])

            before_cols = set(out_df.columns)
            out_df = out_df.merge(feat_df, on=self.index_col, how="left")
            added_cols = [c for c in out_df.columns if c not in before_cols]

            print(
                f"[INFO] merged family={family_name} "
                f"table_rows={len(feat_df)} added_cols={len(added_cols)}"
            )

        return out_df