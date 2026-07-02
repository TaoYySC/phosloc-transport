# Panels: Figure 1(c,d,e); Supplementary Figure 1(a,b)
"""Command-line wrapper for dataset-description plots."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plot_dataset_description import main


if __name__ == "__main__":
    main()
