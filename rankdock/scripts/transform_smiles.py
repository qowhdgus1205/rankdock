#!/usr/bin/env python3
"""CLI wrapper for rankdock.pdbqt."""

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    runpy.run_module("pdbqt", run_name="__main__")
