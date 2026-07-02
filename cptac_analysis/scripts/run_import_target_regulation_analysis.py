#!/usr/bin/env python3
"""Command-line wrapper for the CPTAC target-regulation analysis pipeline."""

from pathlib import Path
import sys

_CPTAC_ROOT = Path(__file__).resolve().parent.parent
if str(_CPTAC_ROOT) not in sys.path:
    sys.path.insert(0, str(_CPTAC_ROOT))

from src.run_import_target_regulation_analysis import main


if __name__ == "__main__":
    main()
