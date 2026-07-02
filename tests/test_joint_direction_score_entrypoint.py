import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HAS_JOINT_SCORE_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ["numpy", "pandas", "matplotlib"]
)


def load_joint_score_module():
    module_path = REPO_ROOT / "import_export/scripts/calculate_joint_direction_score.py"
    spec = importlib.util.spec_from_file_location("joint_direction_score_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(not HAS_JOINT_SCORE_DEPS, "numpy/pandas/matplotlib dependencies are not installed")
class JointDirectionScoreEntrypointTests(unittest.TestCase):
    def test_import_exposes_parse_args_without_running_main(self):
        module = load_joint_score_module()

        args = module.parse_args(
            [
                "--functional_score_threshold",
                "0.75",
                "--min_vote",
                "5",
                "--merge_mode",
                "keep_all_fill_zero",
            ]
        )

        self.assertEqual(args.functional_score_threshold, 0.75)
        self.assertEqual(args.min_vote, 5)
        self.assertEqual(args.merge_mode, "keep_all_fill_zero")
        self.assertEqual(module.format_threshold_tag(0.75), "gt0p75")


if __name__ == "__main__":
    unittest.main()
