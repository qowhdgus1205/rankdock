import os, glob, math, random
import numpy as np
import pandas as pd

# ----------------------------
# 0) Memory-safe part provider
# ----------------------------
class PartMemmap:
    def __init__(self, parts_dir: str, pattern: str = "part_*.npy"):
        self.parts_paths = sorted(glob.glob(os.path.join(parts_dir, pattern)))
        if not self.parts_paths:
            raise FileNotFoundError(f"No parts in {parts_dir} with pattern {pattern}")
        self.parts = [np.load(p, mmap_mode="r") for p in self.parts_paths]
        self.sizes = [a.shape[0] for a in self.parts]
        self.dim = self.parts[0].shape[1]
        for i,a in enumerate(self.parts):
            if a.shape[1] != self.dim:
                raise ValueError(f"Dim mismatch in {self.parts_paths[i]}: {a.shape[1]} vs {self.dim}")
        self.cum = np.cumsum([0] + self.sizes)
        self.total = int(self.cum[-1])

    def get_batch(self, indices: np.ndarray) -> np.ndarray:
        idx = np.asarray(indices, dtype=np.int64)
        out = np.empty((len(idx), self.dim), dtype=np.float32)
        part_ids = np.searchsorted(self.cum, idx, side="right") - 1
        for p, arr in enumerate(self.parts):
            m = (part_ids == p)
            if not m.any():
                continue
            sel = idx[m] - self.cum[p]
            out[m] = arr[sel].astype(np.float32, copy=False)
        return out

# ----------------------------
# 1) CSV: read specific rows without loading whole file
# ----------------------------
def read_smiles_by_row_indices(csv_path, indices, smiles_col="SMILES", chunksize=1_000_000):
    """
    indices: 0-based row indices in the CSV after header (i.e., df.reset_index(drop=True) index)
    Returns dict: {row_index: smiles}
    """
    idx_set = set(int(i) for i in indices)
    out = {}
    start = 0
    usecols = [smiles_col]
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        end = start + len(chunk)
        # intersection within this chunk range
        hit = [i for i in idx_set if start <= i < end]
        if hit:
            rel = np.array(hit, dtype=np.int64) - start
            smiles = chunk.iloc[rel][smiles_col].astype(str).tolist()
            for i, smi in zip(hit, smiles):
                out[int(i)] = smi
            idx_set.difference_update(hit)
            if not idx_set:
                break
        start = end
    if idx_set:
        raise RuntimeError(f"Some indices were not found in CSV (out of range?): {sorted(list(idx_set))[:10]} ...")
    return out

# ----------------------------
# 2) ChemBERTa re-embedding (CSV alignment check)
# ----------------------------
def chemberta_embed(smiles_list, device="cuda"):
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    mdl = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1").to(device)
    mdl.eval()

    # tokenize
    t = tok(smiles_list, return_tensors="pt", padding=True, truncation=True)
    t = {k: v.to(device) for k, v in t.items()}
    with torch.no_grad():
        out = mdl(**t).last_hidden_state.mean(dim=1).detach().cpu().numpy().astype(np.float32)
    return out

def cosine_sim(a, b, eps=1e-12):
    # a,b: (K,D)
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + eps)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + eps)
    return np.sum(an * bn, axis=1)

# ----------------------------
# 3) Main checks
# ----------------------------
def check_alignment(
    csv_path,
    smiles_dir,
    graph_dir,
    combined_dir,
    smiles_col="SMILES",
    pattern="part_*.npy",
    K_internal=5000,
    K_csv=256,
    seed=2025,
    chunksize=1_000_000,
    device="cuda",
):
    random.seed(seed); np.random.seed(seed)

    # Load memmaps
    mm_smiles = PartMemmap(smiles_dir, pattern=pattern)
    mm_graph  = PartMemmap(graph_dir, pattern=pattern)
    mm_comb   = PartMemmap(combined_dir, pattern=pattern)

    # Get CSV row count without loading all
    # (fast: just count lines) - safe for huge file
    import subprocess
    wc = subprocess.check_output(["bash", "-lc", f"python -c \"import sys; p='{csv_path}';"
                                                 f"import os; "
                                                 f"import gzip; "
                                                 f"f=open(p,'rb'); "
                                                 f"n=sum(1 for _ in f)-1; "
                                                 f"print(n)\""])
    N_csv = int(wc.decode().strip())

    print(f"[N] CSV rows = {N_csv:,}")
    print(f"[N] smiles emb total = {mm_smiles.total:,}  dim={mm_smiles.dim}")
    print(f"[N] graph  emb total = {mm_graph.total:,}   dim={mm_graph.dim}")
    print(f"[N] comb   emb total = {mm_comb.total:,}    dim={mm_comb.dim}")

    # A1) size checks
    ok_sizes = (mm_smiles.total == N_csv) and (mm_graph.total == N_csv) and (mm_comb.total == N_csv)
    if not ok_sizes:
        print("\n[FAIL] Row count mismatch. This alone implies NOT aligned with CSV.")
        return

    # A2) combined = concat(smiles, graph) check
    # (sample K_internal indices)
    K_internal = min(K_internal, N_csv)
    idx = np.random.choice(N_csv, size=K_internal, replace=False).astype(np.int64)

    g = mm_graph.get_batch(idx)
    s = mm_smiles.get_batch(idx)
    c = mm_comb.get_batch(idx)

    cg = np.concatenate([g, s], axis=1)  # NOTE: if your combined order is [smiles, graph], swap here
    # compute max abs diff
    mad = np.max(np.abs(cg - c))
    meanad = float(np.mean(np.abs(cg - c)))
    print(f"\n[Internal] combined == concat(graph,smiles) ?  max|diff|={mad:.6g}  mean|diff|={meanad:.6g}")
    if mad > 1e-4:
        print("[WARN] combined does NOT match concat(graph,smiles).")
        print("       -> If your combined is concat(smiles,graph), swap the order in code and re-check.")
        # Try swapped order
        cs = np.concatenate([s, g], axis=1)
        mad2 = np.max(np.abs(cs - c))
        meanad2 = float(np.mean(np.abs(cs - c)))
        print(f"[Internal-swapped] combined == concat(smiles,graph) ? max|diff|={mad2:.6g} mean|diff|={meanad2:.6g}")
        if min(mad, mad2) > 1e-4:
            print("[FAIL] combined is not a clean concat of the two sources at same row indices.")
            return
        else:
            print("[OK] combined matches concat(smiles,graph). Use that order going forward.")
    else:
        print("[OK] combined matches concat(graph,smiles) at sampled rows.")

    if K_csv <= 0:
        print("\n[CSV alignment - ChemBERTa]")
        print("  [SKIP] K_csv <= 0; skipped ChemBERTa re-embedding check.")
        print("\nDone.")
        return

    # B) CSV alignment check using ChemBERTa re-embedding
    # Pick K_csv indices, read SMILES from CSV by chunks, re-embed, compare cosine similarity
    K_csv = min(K_csv, N_csv)
    idx2 = np.random.choice(N_csv, size=K_csv, replace=False).astype(np.int64)
    smi_map = read_smiles_by_row_indices(csv_path, idx2, smiles_col=smiles_col, chunksize=chunksize)
    smi_list = [smi_map[int(i)] for i in idx2]

    # stored chemberta part: we assume smiles_dir corresponds to ChemBERTa embeddings
    stored = mm_smiles.get_batch(idx2)

    # re-embed
    # (batch re-embed to avoid OOM)
    import torch
    dev = device
    if dev == "cuda" and not torch.cuda.is_available():
        dev = "cpu"

    # chunk for embedding
    re = []
    bs = 64
    for i in range(0, len(smi_list), bs):
        re.append(chemberta_embed(smi_list[i:i+bs], device=dev))
    re = np.vstack(re).astype(np.float32)

    cos = cosine_sim(stored, re)
    print("\n[CSV alignment - ChemBERTa]")
    print(f"  cosine: mean={cos.mean():.6f}  min={cos.min():.6f}  p50={np.median(cos):.6f}  p10={np.quantile(cos,0.1):.6f}")
    # heuristic thresholds
    # aligned -> 대부분 0.99~1.0 근처
    # misaligned -> 0.xx (대개 0.0~0.3 근처)
    if np.quantile(cos, 0.1) > 0.95:
        print("  [PASS] ChemBERTa embeddings are aligned with CSV SMILES order (high confidence).")
    else:
        print("  [FAIL] ChemBERTa embeddings do NOT look aligned with CSV order.")
        print("         -> Most common cause: invalid SMILES were dropped during embedding generation.")
        print("         -> Fix: save idx alongside embeddings or keep placeholders for invalid rows.")

    print("\nDone.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--smiles_dir", required=True)
    ap.add_argument("--graph_dir", required=True)
    ap.add_argument("--combined_dir", required=True)
    ap.add_argument("--smiles_col", default="SMILES")
    ap.add_argument("--pattern", default="part_*.npy")
    ap.add_argument("--K_internal", type=int, default=5000)
    ap.add_argument("--K_csv", type=int, default=256)
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    check_alignment(
        csv_path=args.csv,
        smiles_dir=args.smiles_dir,
        graph_dir=args.graph_dir,
        combined_dir=args.combined_dir,
        smiles_col=args.smiles_col,
        pattern=args.pattern,
        K_internal=args.K_internal,
        K_csv=args.K_csv,
        chunksize=args.chunksize,
        device=args.device,
    )
