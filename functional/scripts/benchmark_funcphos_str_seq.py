# Panels: Figure 2(c)
"""Command-line wrapper for FuncPhos benchmark plots."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark_funcphos_str_seq import main


if __name__ == "__main__":
    main()
