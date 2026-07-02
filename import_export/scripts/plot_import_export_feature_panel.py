# Panels: Figure 3(d); Supplementary Figure 4(d)
"""Command-line wrapper for the Import/Export selected-feature panel."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plot_import_export_feature_panel import main


if __name__ == "__main__":
    main()
