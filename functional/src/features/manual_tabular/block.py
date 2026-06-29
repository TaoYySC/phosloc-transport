import pandas as pd

from src.features.manual_tabular.loader import ManualTabularLoader
from src.features.manual_tabular.assembler import TabularFeatureAssembler


class ManualTabularBlock:
    def __init__(
        self,
        families,
        index_col="INDEX",
        return_dataframe=False,
        **assembler_kwargs,
    ):
        self.loader = ManualTabularLoader(index_col=index_col)
        self.assembler = TabularFeatureAssembler(
            feature_families=families,
            return_dataframe=return_dataframe,
            **assembler_kwargs,
        )

    def attach_features(self, sample_df: pd.DataFrame) -> pd.DataFrame:
        return self.loader.merge_all(sample_df)

    def fit_transform(self, train_df: pd.DataFrame, y=None):
        return self.assembler.fit_transform(train_df, y=y)

    def transform(self, df: pd.DataFrame):
        return self.assembler.transform(df)

    def get_feature_names_out(self):
        return self.assembler.get_feature_names_out()

    @property
    def feature_info(self):
        return self.assembler.feature_info