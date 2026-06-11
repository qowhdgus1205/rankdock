#!/usr/bin/env python3
"""CLI wrapper for rankdock.data.embeddings."""

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    runpy.run_module("rankdock.data.embeddings", run_name="__main__")
