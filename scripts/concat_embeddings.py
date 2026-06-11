#!/usr/bin/env python3
"""Concatenate graph and SMILES embedding parts."""

import argparse
import sys
from pathlib import Path


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rankdock.data.embeddings import concat_embedding

    parser = argparse.ArgumentParser(description="Batch-wise concatenate graph and SMILES embeddings")
    parser.add_argument("--graph_dir", required=True, help="Directory containing graph part_*.npy files")
    parser.add_argument("--smiles_dir", required=True, help="Directory containing SMILES part_*.npy files")
    parser.add_argument("--output_dir", required=True, help="Directory to save combined part_*.npy files")
    args = parser.parse_args()

    concat_embedding(args.graph_dir, args.smiles_dir, args.output_dir)
