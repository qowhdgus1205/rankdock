import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def load_top_sets(scores_csv):
    scores = pd.read_csv(scores_csv, usecols=["Score"])["Score"].to_numpy(np.float32)
    n = len(scores)
    specs = {
        "top1000": 1000,
        "top5000": 5000,
        "top1pct": int(np.ceil(n * 0.01)),
    }
    out = {}
    for name, k in specs.items():
        kth = np.partition(scores, k - 1)[k - 1]
        ids = np.flatnonzero(scores <= kth).astype(np.int64)
        out[name] = {
            "k": int(k),
            "cutoff": float(kth),
            "ids": set(ids.tolist()),
            "with_ties": int(ids.shape[0]),
        }
    return out, n


def summarize_run(run_dir, top_sets):
    rows = []
    files = sorted(glob.glob(os.path.join(run_dir, "round_cumulative", "round_*_cumulative.csv*")))
    for path in files:
        stem = Path(path).name
        round_no = int(stem.split("_")[1])
        df = pd.read_csv(path, usecols=["sample_id"])
        selected = set(df["sample_id"].astype(np.int64).tolist())
        row = {
            "round": round_no,
            "cumulative_selected": int(len(selected)),
        }
        for name, payload in top_sets.items():
            hits = len(selected.intersection(payload["ids"]))
            row[f"{name}_hits"] = int(hits)
            row[f"{name}_ratio_vs_k"] = float(hits / payload["k"])
            row[f"{name}_ratio_vs_ties"] = float(hits / payload["with_ties"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("round")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_csv", default="data/EnamineHTS_scores.csv")
    parser.add_argument("--root", default="results/cumulative/enamine2m")
    parser.add_argument("--out_dir", default="outputs/enamine2m_lsh_init0p2/retrieval_summary")
    args = parser.parse_args()

    top_sets, n_total = load_top_sets(args.scores_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "scores_csv": args.scores_csv,
        "n_total": int(n_total),
        "top_sets": {
            name: {k: v for k, v in payload.items() if k != "ids"}
            for name, payload in top_sets.items()
        },
    }
    (out_dir / "topk_cutoffs.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    all_frames = []
    for run_path in sorted(Path(args.root).iterdir()):
        if not run_path.is_dir():
            continue
        if not (run_path / "round_cumulative").is_dir():
            continue
        df = summarize_run(str(run_path), top_sets)
        if df.empty:
            continue
        df.insert(0, "run", run_path.name)
        df.to_csv(out_dir / f"{run_path.name}_round_retrieval.csv", index=False)
        all_frames.append(df)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined.to_csv(out_dir / "all_round_retrieval.csv", index=False)

        lines = []
        for run, sub in combined.groupby("run", sort=True):
            lines.append(f"## {run}")
            lines.append(sub.to_string(index=False))
            lines.append("")
        (out_dir / "all_round_retrieval.txt").write_text("\n".join(lines), encoding="utf-8")
        print(out_dir / "all_round_retrieval.csv")
        print(out_dir / "all_round_retrieval.txt")
    else:
        print("No completed round_cumulative files found.")


if __name__ == "__main__":
    main()
