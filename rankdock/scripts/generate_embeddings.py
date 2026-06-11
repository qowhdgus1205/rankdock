#!/usr/bin/env python3
"""CLI wrapper for data.embeddings."""

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    runpy.run_module("data.embeddings", run_name="__main__")
