#!/usr/bin/env python3
"""Train from saved cumulative rounds and score each run's final compounds.

This evaluates whether the model trained on round-wise cumulative labels ranks
the final extracted compounds consistently with their true docking ranks.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from rankdock.active_learning import (
        PartMemmap,
        RankModel,
        predict_m2_raw_sharded,
        prepare_dataloaders_ddp,
        prepare_regression_dataloaders_ddp,
        train_pairwise_ddp,
        train_rankdnn_ddp,
        train_regression_ddp,
    )
except ModuleNotFoundError:
    from active_learning import (
        PartMemmap,
        RankModel,
        predict_m2_raw_sharded,
        prepare_dataloaders_ddp,
        prepare_regression_dataloaders_ddp,
        train_pairwise_ddp,
        train_rankdnn_ddp,
        train_regression_ddp,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def infer_run_metadata(run_name: str) -> dict:
    parts = run_name.split("_")
    model = parts[0]
    if model not in {"rf", "mlp", "triplet", "pairwise"}:
        raise ValueError(f"Cannot infer model from run dir name: {run_name}")
    acquisition = parts[1] if len(parts) > 1 else "unknown"
    seed = 2025
    for part in parts:
        if part.startswith("seed"):
            try:
                seed = int(part.replace("seed", ""))
            except ValueError:
                seed = 2025
    variant = "rerun" if "rerun" in parts else ("fixed" if "fixed" in parts else "default")
    return {"model": model, "acquisition": acquisition, "seed": seed, "variant": variant}


def load_library_index(csv_path: str) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(csv_path, usecols=["SMILES", "Score"])
    df.insert(0, "sample_id", np.arange(len(df), dtype=np.int64))
    score = df["Score"].to_numpy(np.float32, copy=False)
    order = np.argsort(score, kind="mergesort")
    true_rank = np.empty(len(score), dtype=np.int64)
    true_rank[order] = np.arange(1, len(score) + 1, dtype=np.int64)
    return df, true_rank


def attach_sample_ids(final_df: pd.DataFrame, library_df: pd.DataFrame) -> pd.DataFrame:
    if "sample_id" in final_df.columns:
        return final_df.copy()

    left = final_df.copy()
    left["_dup_i"] = left.groupby(["SMILES", "Score"], sort=False).cumcount()
    right = library_df.copy()
    right["_dup_i"] = right.groupby(["SMILES", "Score"], sort=False).cumcount()
    merged = left.merge(
        right[["SMILES", "Score", "_dup_i", "sample_id"]],
        on=["SMILES", "Score", "_dup_i"],
        how="left",
        validate="one_to_one",
    ).drop(columns=["_dup_i"])
    if merged["sample_id"].isna().any():
        missing = int(merged["sample_id"].isna().sum())
        raise ValueError(f"Failed to map {missing} final compounds to library sample_id")
    merged["sample_id"] = merged["sample_id"].astype(np.int64)
    return merged


def add_embeddings(df: pd.DataFrame, feats: PartMemmap) -> pd.DataFrame:
    out = df.copy()
    ids = out["sample_id"].to_numpy(np.int64, copy=False)
    emb = feats.get_batch(ids).astype(np.float32, copy=False)
    out["combined_embedding"] = list(emb)
    return out


def train_model(model_name: str, train_df: pd.DataFrame, feats: PartMemmap, args: argparse.Namespace):
    train_df = add_embeddings(train_df, feats)
    x_dim = feats.dim
    device = torch.device(args.device)

    if model_name == "rf":
        from sklearn.ensemble import RandomForestRegressor

        x_train = np.vstack(train_df["combined_embedding"].values).astype(np.float32, copy=False)
        y_train = train_df["Score"].to_numpy(np.float32, copy=False)
        model = RandomForestRegressor(
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth,
            min_samples_split=args.rf_min_samples_split,
            min_samples_leaf=args.rf_min_samples_leaf,
            random_state=args.seed,
            n_jobs=args.rf_n_jobs,
        )
        model.fit(x_train, y_train)
        return model

    if model_name == "mlp":
        model = RankModel(x_dim, num_layers=args.m2_layers, dropout=args.m2_dropout).to(device)
        loaders = prepare_regression_dataloaders_ddp(
            train_df,
            args.m2_train_bs,
            args.m2_split,
            rank=0,
            world=1,
            num_workers=args.num_workers,
            seed=args.seed,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=args.prefetch_factor,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.m2_lr, weight_decay=args.m2_wd)
        train_regression_ddp(
            model,
            loaders[0],
            loaders[1],
            optimizer,
            device,
            epochs=args.m2_epochs,
            patience=args.m2_patience,
            use_amp=False,
            rank=0,
            train_sampler=loaders[2],
        )
        return model

    model = RankModel(x_dim, num_layers=args.m2_layers, dropout=args.m2_dropout).to(device)
    loaders = prepare_dataloaders_ddp(
        train_df,
        args.m2_train_bs,
        args.m2_split,
        rank=0,
        world=1,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=args.prefetch_factor,
        semi_hard_pos_window=args.semi_hard_pos_window,
        semi_hard_neg_window=args.semi_hard_neg_window,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.m2_lr, weight_decay=args.m2_wd)
    if model_name == "pairwise":
        train_pairwise_ddp(
            model,
            loaders[0],
            loaders[1],
            optimizer,
            device,
            epochs=args.m2_epochs,
            patience=args.m2_patience,
            use_amp=False,
            rank=0,
            train_sampler=loaders[2],
        )
    elif model_name == "triplet":
        train_rankdnn_ddp(
            model,
            loaders[0],
            loaders[1],
            optimizer,
            device,
            margin=args.margin,
            lambda_rank=args.lambda_rank,
            epochs=args.m2_epochs,
            patience=args.m2_patience,
            use_amp=False,
            rank=0,
            train_sampler=loaders[2],
        )
    else:
        raise ValueError(model_name)
    return model


def predict_final(model_name: str, model, final_ids: np.ndarray, feats: PartMemmap, args: argparse.Namespace) -> np.ndarray:
    if model_name == "rf":
        pred = np.empty(len(final_ids), dtype=np.float32)
        for start in range(0, len(final_ids), args.m2_pred_bs):
            end = min(start + args.m2_pred_bs, len(final_ids))
            xb = feats.get_batch(final_ids[start:end]).astype(np.float32, copy=False)
            pred[start:end] = model.predict(xb).astype(np.float32, copy=False)
        return pred

    _, pred = predict_m2_raw_sharded(
        model,
        feats,
        final_ids,
        rank=0,
        world=1,
        batch_size=args.m2_pred_bs,
        device=args.device,
    )
    return pred.astype(np.float32, copy=False)


def ndcg_binary(relevance: np.ndarray, k: int, total_relevant: int | None = None) -> float:
    k = min(k, len(relevance))
    if k <= 0:
        return float("nan")
    rel = relevance[:k].astype(np.float64, copy=False)
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
    dcg = float(np.sum(rel * discounts))
    if total_relevant is None:
        total_relevant = int(np.sum(relevance))
    ideal_hits = int(min(total_relevant, k))
    if ideal_hits <= 0:
        return float("nan")
    idcg = float(np.sum(discounts[:ideal_hits]))
    return dcg / idcg


def summarize_scored(
    df: pd.DataFrame,
    top1000_cutoff_rank: int,
    top5000_cutoff_rank: int,
    top1pct_cutoff_rank: int,
) -> dict:
    pred_ordered = df.sort_values("pred_score", ascending=True, kind="mergesort")
    true_rank = pred_ordered["true_rank_global"].to_numpy(np.float64, copy=False)
    pred_rank = np.arange(1, len(pred_ordered) + 1, dtype=np.float64)
    tmp = pd.DataFrame({"pred_rank": pred_rank, "true_rank": true_rank})
    rel1000 = (true_rank <= top1000_cutoff_rank).astype(np.int8)
    rel5000 = (true_rank <= top5000_cutoff_rank).astype(np.int8)
    rel1pct = (true_rank <= top1pct_cutoff_rank).astype(np.int8)
    hit1000 = int(np.sum(rel1000[: min(1000, len(rel1000))]))
    hit5000 = int(np.sum(rel5000[: min(5000, len(rel5000))]))
    hit1pct = int(np.sum(rel1pct[: min(top1pct_cutoff_rank, len(rel1pct))]))
    return {
        "n_final": len(df),
        "spearman_pred_vs_true_rank": float(tmp["pred_rank"].corr(tmp["true_rank"], method="spearman")),
        "kendall_pred_vs_true_rank": float(tmp["pred_rank"].corr(tmp["true_rank"], method="kendall")),
        "mean_true_rank_top1000_pred": float(np.mean(true_rank[: min(1000, len(true_rank))])),
        "median_true_rank_top1000_pred": float(np.median(true_rank[: min(1000, len(true_rank))])),
        "hits_top1000_true_top1000": hit1000,
        "hits_top5000_true_top5000": hit5000,
        "hits_top1pct_true_top1pct": hit1pct,
        "recovery_top1000_true_top1000": hit1000 / float(top1000_cutoff_rank),
        "recovery_top5000_true_top5000": hit5000 / float(top5000_cutoff_rank),
        "recovery_top1pct_true_top1pct": hit1pct / float(top1pct_cutoff_rank),
        "ndcg_top1000_true_top1000": ndcg_binary(rel1000, 1000, total_relevant=top1000_cutoff_rank),
        "ndcg_top5000_true_top5000": ndcg_binary(rel5000, 5000, total_relevant=top5000_cutoff_rank),
        "ndcg_top1pct_true_top1pct": ndcg_binary(rel1pct, top1pct_cutoff_rank, total_relevant=top1pct_cutoff_rank),
        "ndcg_top1pct_within_recovered_hits": ndcg_binary(
            rel1pct,
            top1pct_cutoff_rank,
            total_relevant=None,
        ),
    }


def write_text_summary(summary_df: pd.DataFrame, path: Path) -> None:
    cols = [
        "run",
        "model",
        "acquisition",
        "seed",
        "variant",
        "round",
        "spearman_pred_vs_true_rank",
        "kendall_pred_vs_true_rank",
        "mean_true_rank_top1000_pred",
        "median_true_rank_top1000_pred",
        "hits_top1000_true_top1000",
        "hits_top5000_true_top5000",
        "hits_top1pct_true_top1pct",
        "recovery_top1000_true_top1000",
        "recovery_top5000_true_top5000",
        "recovery_top1pct_true_top1pct",
        "ndcg_top1000_true_top1000",
        "ndcg_top5000_true_top5000",
        "ndcg_top1pct_true_top1pct",
        "ndcg_top1pct_within_recovered_hits",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write(summary_df[cols].to_string(index=False))
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bo_root", default="outputs/enamine2m_lsh_init0p2/bo")
    parser.add_argument("--smiles_csv", default="data/EnamineHTS_scores.csv")
    parser.add_argument("--emb_dir", default="output/combined_embeddings/2M")
    parser.add_argument("--out_dir", default="outputs/enamine2m_lsh_init0p2/final_set_round_model_scores")
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--rounds", nargs="*", type=int, default=list(range(0, 11)))
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=3)
    parser.add_argument("--m2_layers", type=int, default=2)
    parser.add_argument("--m2_dropout", type=float, default=0.3)
    parser.add_argument("--m2_lr", type=float, default=1e-3)
    parser.add_argument("--m2_wd", type=float, default=1e-4)
    parser.add_argument("--m2_train_bs", type=int, default=4096)
    parser.add_argument("--m2_pred_bs", type=int, default=4096)
    parser.add_argument("--m2_epochs", type=int, default=200)
    parser.add_argument("--m2_patience", type=int, default=50)
    parser.add_argument("--m2_split", type=float, default=0.9)
    parser.add_argument("--semi_hard_pos_window", type=int, default=0)
    parser.add_argument("--semi_hard_neg_window", type=int, default=0)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--lambda_rank", type=float, default=0.01)
    parser.add_argument("--rf_n_estimators", type=int, default=200)
    parser.add_argument("--rf_max_depth", type=int, default=None)
    parser.add_argument("--rf_min_samples_split", type=int, default=2)
    parser.add_argument("--rf_min_samples_leaf", type=int, default=1)
    parser.add_argument("--rf_n_jobs", type=int, default=32)
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    set_seed(args.seed)

    bo_root = Path(args.bo_root)
    out_dir = Path(args.out_dir)
    pred_root = out_dir / "predictions"
    pred_root.mkdir(parents=True, exist_ok=True)
    incremental_summary_csv = out_dir / "final_compound_prediction_vs_true_rank_summary_incremental.csv"
    if not incremental_summary_csv.exists():
        pd.DataFrame().to_csv(incremental_summary_csv, index=False)

    library_df, true_rank_global = load_library_index(args.smiles_csv)
    top1pct_cutoff_rank = max(1, int(len(library_df) * 0.01))
    feats = PartMemmap(args.emb_dir)

    run_dirs = [bo_root / r for r in args.runs] if args.runs else sorted(p for p in bo_root.iterdir() if p.is_dir())
    required_incremental_cols = {
        "run",
        "round",
        "model",
        "acquisition",
        "recovery_top1pct_true_top1pct",
        "ndcg_top1pct_true_top1pct",
    }
    summary_rows = []
    done_keys = set()
    if incremental_summary_csv.exists() and incremental_summary_csv.stat().st_size > 1:
        old_summary = pd.read_csv(incremental_summary_csv)
        if required_incremental_cols.issubset(old_summary.columns):
            summary_rows = old_summary.to_dict("records")
            done_keys = {(str(r["run"]), int(r["round"])) for r in summary_rows}
        else:
            incremental_summary_csv.write_text("", encoding="utf-8")
    for run_dir in run_dirs:
        final_csv = run_dir / "top1pct.csv"
        cum_dir = run_dir / "round_cumulative"
        if not final_csv.exists() or not cum_dir.exists():
            continue

        run_name = run_dir.name
        meta = infer_run_metadata(run_name)
        model_name = meta["model"]
        print(f"[RUN] {run_name} | model={model_name} | acq={meta['acquisition']}", flush=True)

        final_df = attach_sample_ids(pd.read_csv(final_csv), library_df)
        final_ids = final_df["sample_id"].to_numpy(np.int64, copy=False)
        final_df["true_rank_global"] = true_rank_global[final_ids]
        final_df = final_df.sort_values("true_rank_global", kind="mergesort").reset_index(drop=True)
        final_df["true_rank_within_final"] = np.arange(1, len(final_df) + 1, dtype=np.int64)
        final_df = final_df.sort_values("sample_id", kind="mergesort").reset_index(drop=True)

        for rd in args.rounds:
            key = (run_name, int(rd))
            if key in done_keys:
                print(f"[DONE-SKIP] {run_name} round={rd:02d}", flush=True)
                continue
            cumulative_csv = cum_dir / f"round_{rd:02d}_cumulative.csv"
            if not cumulative_csv.exists():
                continue
            out_csv = pred_root / run_name / f"round_{rd:02d}_final_compound_predictions.csv"
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            if out_csv.exists():
                print(f"[SKIP] {out_csv}", flush=True)
                scored = pd.read_csv(out_csv)
            else:
                train_df = pd.read_csv(cumulative_csv)
                if "sample_id" not in train_df.columns:
                    train_df = attach_sample_ids(train_df, library_df)
                print(
                    f"[TRAIN] {run_name} round={rd:02d} | n_train={len(train_df)} | n_final={len(final_df)}",
                    flush=True,
                )
                model = train_model(model_name, train_df, feats, args)
                pred = predict_final(model_name, model, final_ids, feats, args)
                scored = final_df.copy()
                scored["source_run"] = run_name
                scored["model"] = model_name
                scored["round"] = rd
                scored["pred_score"] = pred
                pred_order = np.argsort(pred, kind="mergesort")
                pred_rank = np.empty(len(pred_order), dtype=np.int64)
                pred_rank[pred_order] = np.arange(1, len(pred_order) + 1, dtype=np.int64)
                scored["predicted_rank_within_final"] = pred_rank
                scored["rank_error_within_final"] = scored["predicted_rank_within_final"] - scored["true_rank_within_final"]
                scored = scored.sort_values("predicted_rank_within_final", kind="mergesort")
                scored.to_csv(out_csv, index=False)
                print(f"[SAVE] {out_csv}", flush=True)

            row = summarize_scored(
                scored,
                top1000_cutoff_rank=1000,
                top5000_cutoff_rank=5000,
                top1pct_cutoff_rank=top1pct_cutoff_rank,
            )
            row.update({"run": run_name, "round": rd, **meta})
            summary_rows.append(row)
            done_keys.add(key)
            row_df = pd.DataFrame([row])
            if incremental_summary_csv.exists() and incremental_summary_csv.stat().st_size > 1:
                row_df.to_csv(incremental_summary_csv, mode="a", header=False, index=False)
            else:
                row_df.to_csv(incremental_summary_csv, index=False)

    summary = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "final_compound_prediction_vs_true_rank_summary.csv"
    summary_txt = out_dir / "final_compound_prediction_vs_true_rank_summary.txt"
    summary.to_csv(summary_csv, index=False)
    write_text_summary(summary.sort_values(["run", "round"]), summary_txt)
    print(f"[DONE] {summary_csv}", flush=True)
    print(f"[DONE] {summary_txt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
