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


functional_utils = None
import_export_utils = None

if pd is not None and importlib.util.find_spec("numpy") is not None:
    functional_utils = load_module("functional_utils_under_test", "functional/src/utils.py")
    import_export_utils = load_module("import_export_utils_under_test", "import_export/src/utils.py")


@unittest.skipIf(pd is None or functional_utils is None, "pandas/numpy dependencies are not installed")
class FunctionalPreprocessingTests(unittest.TestCase):
    def test_standardize_sample_table_builds_index(self):
        df = pd.DataFrame(
            {
                "ACC_ID": [" P12345 ", "Q8XYZ1"],
                "POSITION": [" 7 ", 12],
            }
        )

        out = functional_utils.standardize_sample_table(df)

        self.assertEqual(out["INDEX"].tolist(), ["P12345_7", "Q8XYZ1_12"])
        self.assertEqual(out["ACC_ID"].tolist(), ["P12345", "Q8XYZ1"])
        self.assertEqual(out["POSITION"].tolist(), [7, 12])

    def test_drop_invalid_site_rows_keeps_only_sty_sites(self):
        df = pd.DataFrame(
            {
                "INDEX": ["P1_S2", "P2_A1", "P3_T1"],
                "ACC_ID": ["P1", "P2", "P3"],
                "POSITION": [2, 1, 1],
                "FULL_SEQUENCE": ["ASD", "ACD", "TAA"],
            }
        )

        out = functional_utils.drop_invalid_site_rows(df, enforce_sty=True)

        self.assertEqual(out["INDEX"].tolist(), ["P1_S2", "P3_T1"])


@unittest.skipIf(pd is None or import_export_utils is None, "pandas/numpy dependencies are not installed")
class ImportExportPreprocessingTests(unittest.TestCase):
    def test_standardize_sample_table_builds_index(self):
        df = pd.DataFrame(
            {
                "ACC_ID": [" P12345 ", "Q8XYZ1"],
                "POSITION": [" 7 ", 12],
            }
        )

        out = import_export_utils.standardize_sample_table(df)

        self.assertEqual(out["INDEX"].tolist(), ["P12345_7", "Q8XYZ1_12"])
        self.assertEqual(out["ACC_ID"].tolist(), ["P12345", "Q8XYZ1"])
        self.assertEqual(out["POSITION"].tolist(), [7, 12])

    def test_drop_invalid_site_rows_removes_bad_positions_and_non_sty(self):
        df = pd.DataFrame(
            {
                "INDEX": ["P1_S2", "P2_A1", "P3_T1", "P4_Y9"],
                "ACC_ID": ["P1", "P2", "P3", "P4"],
                "POSITION": [2, 1, 1, 9],
                "FULL_SEQUENCE": ["ASD", "ACD", "TAA", "YAA"],
            }
        )

        out = import_export_utils.drop_invalid_site_rows(df, enforce_sty=True)

        self.assertEqual(out["INDEX"].tolist(), ["P1_S2", "P3_T1"])


if __name__ == "__main__":
    unittest.main()
