#!/usr/bin/env python3
"""Command-line wrapper for per-site across-cancer CPTAC boxplots."""

from pathlib import Path
import sys

_CPTAC_ROOT = Path(__file__).resolve().parent.parent
if str(_CPTAC_ROOT) not in sys.path:
    sys.path.insert(0, str(_CPTAC_ROOT))

from src.plot_phosphosite_across_cancers import main


if __name__ == "__main__":
    main()
