#!/usr/bin/env python3
"""Command-line wrapper for recompute_site_wilcoxon_min_targets10."""

from pathlib import Path
import sys

_CPTAC_ROOT = Path(__file__).resolve().parent.parent
if str(_CPTAC_ROOT) not in sys.path:
    sys.path.insert(0, str(_CPTAC_ROOT))

from src.recompute_site_wilcoxon_min_targets10 import main


if __name__ == "__main__":
    main()
