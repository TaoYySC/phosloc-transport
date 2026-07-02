# Panels: Supplementary Figure 2(c,d,e)
"""Command-line wrapper for functional validation-score plots."""

from pathlib import Path
import runpy
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


if __name__ == "__main__":
    runpy.run_module("src.plot_functional_validation_scores", run_name="__main__")
