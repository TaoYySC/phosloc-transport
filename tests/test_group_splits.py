import importlib.util
import unittest
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    pd = None


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name, relative_path):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


functional_split = None
import_export_split = None

HAS_SPLIT_DEPS = (
    pd is not None
    and importlib.util.find_spec("numpy") is not None
    and importlib.util.find_spec("sklearn") is not None
)

if HAS_SPLIT_DEPS:
    functional_split = load_module("functional_split_under_test", "functional/src/data/split.py")
    import_export_split = load_module("import_export_split_under_test", "import_export/src/data/split.py")


def assert_group_disjoint(test_case, df, train_idx, test_idx, group_col):
    train_groups = set(df.iloc[train_idx][group_col].astype(str))
    test_groups = set(df.iloc[test_idx][group_col].astype(str))
    test_case.assertFalse(train_groups & test_groups)


@unittest.skipIf(not HAS_SPLIT_DEPS, "pandas/numpy/scikit-learn dependencies are not installed")
class GroupSplitTests(unittest.TestCase):
    def test_functional_outer_cv_splits_keep_groups_disjoint(self):
        df = pd.DataFrame(
            {
                "ACC_ID": [f"P{i}" for i in range(12)],
                "LABEL": [0, 1] * 6,
            }
        )

        splits = list(
            functional_split.build_outer_cv_splits(
                df,
                n_splits=3,
                seed=42,
                group_col="ACC_ID",
                label_col="LABEL",
                stratify=True,
            )
        )

        self.assertEqual(len(splits), 3)
        for _, train_idx, test_idx in splits:
            assert_group_disjoint(self, df, train_idx, test_idx, "ACC_ID")

    def test_import_export_group_kfold_keeps_clusters_disjoint(self):
        df = pd.DataFrame(
            {
                "Cluster_ID": [f"C{i}" for i in range(12)],
                "LABEL": [0, 1] * 6,
            }
        )

        splits = import_export_split.build_group_kfold_splits(
            df,
            n_splits=3,
            group_col="Cluster_ID",
            label_col="LABEL",
            seed=42,
            stratify=True,
        )

        self.assertEqual(len(splits), 3)
        for train_idx, test_idx in splits:
            assert_group_disjoint(self, df, train_idx, test_idx, "Cluster_ID")


if __name__ == "__main__":
    unittest.main()
