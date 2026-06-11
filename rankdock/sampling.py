import argparse
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter
import random

def Apply_lsh(csv_path, embedding_dir, input_dim=1280, output_dim=32, seed=None):
    class HyperplaneLSH:
        def __init__(self, input_dim, output_dim):
            self.hyperplanes = np.random.randn(output_dim, input_dim)

        def compute_hash(self, vectors):
            return (np.dot(vectors, self.hyperplanes.T) > 0).astype(int)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    print("Loading input CSV...")
    df = pd.read_csv(csv_path).reset_index(drop=True)
    df["sample_id"] = np.arange(len(df), dtype=np.int64)
    total_n = len(df)

    print("Initializing LSH...")
    lsh = HyperplaneLSH(input_dim=input_dim, output_dim=output_dim)

    print("Processing embeddings in batches...")
    embedding_parts = sorted(glob.glob(os.path.join(embedding_dir, "part_*.npy")))
    all_hashes = []
    indices_by_part = []

    row_start = 0
    for part_path in tqdm(embedding_parts, desc="Embedding Batches"):
        emb = np.load(part_path, mmap_mode="r")
        part_len = emb.shape[0]
        hash_vectors = lsh.compute_hash(emb)
        all_hashes.extend(map(tuple, hash_vectors))
        indices_by_part.append((row_start, row_start + part_len, emb))  # store start, end, emb
        row_start += part_len

    df['hash'] = all_hashes

    print(f"✅ Total Samples: {len(df)}")
    print(f"✅ Total Unique Hash Buckets: {len(set(all_hashes))}")

    return df

def load_selected_embeddings(embedding_dir, selected_indices):
    embedding_parts = sorted(glob.glob(os.path.join(embedding_dir, "part_*.npy")))
    if not embedding_parts:
        raise FileNotFoundError(f"No part_*.npy found in {embedding_dir}")

    ordered_positions = list(enumerate(selected_indices))
    positions_by_index = defaultdict(list)
    for output_pos, row_idx in ordered_positions:
        positions_by_index[int(row_idx)].append(output_pos)

    first_part = np.load(embedding_parts[0], mmap_mode="r")
    emb_dim = first_part.shape[1]
    selected_embeddings = np.empty((len(selected_indices), emb_dim), dtype=first_part.dtype)

    row_start = 0
    remaining = set(positions_by_index.keys())
    for part_path in tqdm(embedding_parts, desc="Collecting Selected Embeddings"):
        emb = np.load(part_path, mmap_mode="r")
        row_end = row_start + emb.shape[0]
        part_indices = sorted(idx for idx in remaining if row_start <= idx < row_end)

        for global_idx in part_indices:
            local_idx = global_idx - row_start
            for output_pos in positions_by_index[global_idx]:
                selected_embeddings[output_pos] = emb[local_idx]

        remaining.difference_update(part_indices)
        row_start = row_end

        if not remaining:
            break

    if remaining:
        raise IndexError(f"Some selected indices exceed embedding rows: {sorted(remaining)[:10]}")

    return selected_embeddings

def save_selected_embeddings(embedding_dir, selected_indices, output_npy_path):
    selected_embeddings = load_selected_embeddings(embedding_dir, selected_indices)
    os.makedirs(os.path.dirname(output_npy_path) or ".", exist_ok=True)
    np.save(output_npy_path, selected_embeddings)
    print(f"✅ Selected embeddings saved to: {output_npy_path}")

def select_initial_samples(df, num_samples, strategy='uniform', output_csv_path=None, output_embedding_path=None, embedding_dir=None, seed=None):
    print(f"\n📦 Selecting {num_samples} samples using strategy: {strategy}")
    if seed is not None:
        random.seed(seed)

    hash_to_samples = defaultdict(list)
    for idx, row in df.iterrows():
        h = tuple(row['hash'])
        hash_to_samples[h].append(idx)

    selected_indices = []
    bucket_keys = list(hash_to_samples.keys())
    total_buckets = len(bucket_keys)

    per_bucket = num_samples // total_buckets
    remainder = num_samples % total_buckets

    # 1️⃣ 가능한 만큼만 균등 추출
    for i, h in enumerate(bucket_keys):
        samples = hash_to_samples[h]
        k = per_bucket + (1 if i < remainder else 0)
        if len(samples) <= k:
            selected_indices.extend(samples)
        else:
            selected_indices.extend(random.sample(samples, k))

    # 2️⃣ 부족할 경우 랜덤 샘플링으로 보충
    missing = num_samples - len(selected_indices)
    if missing > 0:
        print(f"🔄 {missing}개 부족 — 전체에서 무작위로 보충합니다.")
        all_available = list(set(df.index) - set(selected_indices))
        selected_indices.extend(random.sample(all_available, min(missing, len(all_available))))

    df_selected = df.loc[selected_indices]
    if output_csv_path:
        df_selected.to_csv(output_csv_path, index=False)
        print(f"✅ Selected samples saved to: {output_csv_path}")
    if output_embedding_path:
        if embedding_dir is None:
            raise ValueError("embedding_dir is required to save selected embeddings.")
        save_selected_embeddings(embedding_dir, selected_indices, output_embedding_path)

    return df_selected

def select_random_samples(df, num_samples=1, output_csv_path=None, output_embedding_path=None, embedding_dir=None, seed=None):
    print(f"\n🎲 Selecting {num_samples} random samples")
    if seed is not None:
        random.seed(seed)

    num_samples = min(num_samples, len(df))
    selected_indices = random.sample(list(df.index), num_samples)
    df_selected = df.loc[selected_indices]

    if output_csv_path:
        df_selected.to_csv(output_csv_path, index=False)
        print(f"✅ Random samples saved to: {output_csv_path}")
    if output_embedding_path:
        if embedding_dir is None:
            raise ValueError("embedding_dir is required to save random sample embeddings.")
        save_selected_embeddings(embedding_dir, selected_indices, output_embedding_path)

    return df_selected

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSH-based initial sampling using embedding parts")
    parser.add_argument("--csv_path", required=True, help="Path to SMILES CSV")
    parser.add_argument("--embedding_dir", default=None, help="Directory containing part_*.npy embeddings")
    parser.add_argument("--input_dim", type=int, default=1280)
    parser.add_argument("--output_dim", type=int, default=32)
    parser.add_argument("--sample_ratio", type=float, default=None,
                        help="Ratio of total samples to select (e.g., 0.001 = 0.1%%)")
    parser.add_argument("--output_csv", default="./outputs/initial_selected_samples.csv")
    parser.add_argument("--output_embedding_npy", default=None,
                        help="Optional path to save embeddings for LSH-selected samples")
    parser.add_argument("--random_output_csv", default=None,
                        help="Optional path to save additional randomly selected samples")
    parser.add_argument("--random_output_embedding_npy", default=None,
                        help="Optional path to save embeddings for randomly selected samples")
    parser.add_argument("--random_count", type=int, default=1,
                        help="Number of random samples to save when --random_output_csv is set")
    parser.add_argument("--random_ratio", type=float, default=None,
                        help="Ratio of total samples to randomly select (e.g., 0.001 = 0.1%%)")
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    run_lsh_sampling = args.embedding_dir is not None and args.sample_ratio is not None
    run_random_sampling = args.random_output_csv is not None

    if not run_lsh_sampling and not run_random_sampling:
        parser.error("Specify either LSH sampling (--embedding_dir and --sample_ratio) or random sampling (--random_output_csv).")

    if args.random_ratio is not None and args.random_ratio < 0:
        parser.error("--random_ratio must be >= 0.")

    if args.random_count < 0:
        parser.error("--random_count must be >= 0.")

    if (args.output_embedding_npy or args.random_output_embedding_npy) and args.embedding_dir is None:
        parser.error("--embedding_dir is required when saving selected embeddings.")

    if run_lsh_sampling:
        df = Apply_lsh(
            csv_path=args.csv_path,
            embedding_dir=args.embedding_dir,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            seed=args.seed
        )

        total_count = len(df)
        num_samples = int(total_count * args.sample_ratio)
        print(f"\n📊 Total: {total_count} compounds → Sampling: {num_samples} compounds ({args.sample_ratio * 100:.2f}%)")

        select_initial_samples(
            df=df,
            num_samples=num_samples,
            strategy='uniform',
            output_csv_path=args.output_csv,
            output_embedding_path=args.output_embedding_npy,
            embedding_dir=args.embedding_dir,
            seed=args.seed
        )
    else:
        print("Loading input CSV for random sampling...")
        df = pd.read_csv(args.csv_path).reset_index(drop=True)
        df["sample_id"] = np.arange(len(df), dtype=np.int64)

    if run_random_sampling:
        random_num_samples = args.random_count
        if args.random_ratio is not None:
            random_num_samples = int(len(df) * args.random_ratio)
            print(f"\n📊 Total: {len(df)} compounds → Random sampling: {random_num_samples} compounds ({args.random_ratio * 100:.2f}%)")

        select_random_samples(
            df=df,
            num_samples=random_num_samples,
            output_csv_path=args.random_output_csv,
            output_embedding_path=args.random_output_embedding_npy,
            embedding_dir=args.embedding_dir,
            seed=args.seed
        )
