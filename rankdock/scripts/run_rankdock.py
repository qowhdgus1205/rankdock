#!/usr/bin/env python3
"""Run the RankDock active-learning loop."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from active_learning import parse_args, run_bo


if __name__ == "__main__":
    run_bo(parse_args())
