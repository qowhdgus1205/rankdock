#!/usr/bin/env python3
"""CLI wrapper for rankdock.retrieval."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval import main


if __name__ == "__main__":
    main()
