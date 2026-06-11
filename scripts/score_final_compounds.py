#!/usr/bin/env python3
"""CLI wrapper for rankdock.score_final."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rankdock.score_final import main


if __name__ == "__main__":
    raise SystemExit(main())
