# Panels: Supplementary Figure 4(a,b)
"""Command-line wrapper for Import/Export score-distribution panels."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plot_import_export_score_distribution import main


if __name__ == "__main__":
    main()
