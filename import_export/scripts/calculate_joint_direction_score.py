# Panels: Figure 3(c)
"""Command-line wrapper for joint direction-score calculation."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calculate_joint_direction_score import main


if __name__ == "__main__":
    main()
