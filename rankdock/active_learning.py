import os
import glob
import math
import random
import argparse
import json
from datetime import timedelta
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, random_split, DistributedSampler, TensorDataset

from torch.amp import autocast, GradScaler

import gpytorch
try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except Exception:
    Chem = None
    MurckoScaffold = None

# =========================
# 0) DDP setup / utils
# =========================
def ddp_setup(timeout_minutes: int = 10):
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")

    # speed-focused defaults for large-scale training
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)

    if world > 1:
        dist.init_process_group(
            backend="nccl" if use_cuda else "gloo",
            init_method="env://",
            timeout=timedelta(minutes=max(1, int(timeout_minutes))),
        )
    return local_rank, rank, world

def ddp_cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def is_rank0(rank: int) -> bool:
    return rank == 0

def bcast_object(obj, src=0):
    if not dist.is_initialized():
        return obj
    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]

def dist_barrier():
    if dist.is_initialized():
        dist.barrier()

def dist_device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
    return torch.device("cpu")

def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model

def shard_indices(indices: np.ndarray, rank: int, world: int) -> np.ndarray:
    # interleaved sharding
    return indices[rank::world]

def allgather_variable_1d_int64(local_arr: np.ndarray, world: int):
    """
    Gather variable-length 1D int64 arrays from all ranks.
    Returns concatenated array on all ranks.
    """
    if world <= 1 or not dist.is_initialized():
        return local_arr.astype(np.int64, copy=False)
    dev = dist_device()
    local_n = torch.tensor([local_arr.shape[0]], device=dev, dtype=torch.int64)
    sizes = [torch.zeros_like(local_n) for _ in range(world)]
    dist.all_gather(sizes, local_n)
    sizes = [int(s.item()) for s in sizes]
    max_n = max(sizes) if sizes else 0

    pad = np.full((max_n,), -1, dtype=np.int64)
    pad[:local_arr.shape[0]] = local_arr
    t = torch.from_numpy(pad).to(dev, non_blocking=True)

    gathered = [torch.empty_like(t) for _ in range(world)]
    dist.all_gather(gathered, t)

    out = []
    for i in range(world):
        n = sizes[i]
        if n > 0:
            out.append(gathered[i][:n].cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0,), dtype=np.int64)

def allgather_variable_1d_float32(local_arr: np.ndarray, world: int):
    """
    Gather variable-length 1D float32 arrays from all ranks.
    Returns concatenated array on all ranks.
    """
    if world <= 1 or not dist.is_initialized():
        return local_arr.astype(np.float32, copy=False)
    dev = dist_device()
    local_n = torch.tensor([local_arr.shape[0]], device=dev, dtype=torch.int64)
    sizes = [torch.zeros_like(local_n) for _ in range(world)]
    dist.all_gather(sizes, local_n)
    sizes = [int(s.item()) for s in sizes]
    max_n = max(sizes) if sizes else 0

    pad = np.zeros((max_n,), dtype=np.float32)
    pad[:local_arr.shape[0]] = local_arr
    t = torch.from_numpy(pad).to(dev, non_blocking=True)

    gathered = [torch.empty_like(t) for _ in range(world)]
    dist.all_gather(gathered, t)

    out = []
    for i in range(world):
        n = sizes[i]
        if n > 0:
            out.append(gathered[i][:n].cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0,), dtype=np.float32)

def gather_pairs_ids_vals(local_ids: np.ndarray, local_vals: np.ndarray, world: int):
    """
    Gather (ids, vals) variable-length pairs from all ranks.
    Return all_ids, all_vals concatenated (on all ranks).
    """
    all_ids = allgather_variable_1d_int64(local_ids.astype(np.int64, copy=False), world)
    all_vals = allgather_variable_1d_float32(local_vals.astype(np.float32, copy=False), world)
    return all_ids, all_vals

# =========================
# 1) Memmap feature provider
# =========================
class PartMemmap:
    def __init__(self, parts_dir: str, pattern: str = "part_*.npy"):
        self.parts_paths = sorted(glob.glob(os.path.join(parts_dir, pattern)))
        if not self.parts_paths:
            raise FileNotFoundError(f"No .npy parts found in {parts_dir} with pattern {pattern}")
        self.parts = [np.load(p, mmap_mode="r") for p in self.parts_paths]  # (Ni, D)
        self.sizes = [arr.shape[0] for arr in self.parts]
        self.dim = self.parts[0].shape[1]
        self.cum = np.cumsum([0] + self.sizes)  # len = n_parts+1
        self.total = int(self.cum[-1])

        for i, arr in enumerate(self.parts):
            if arr.shape[1] != self.dim:
                raise ValueError(f"Dim mismatch at part {i}: got {arr.shape[1]}, expect {self.dim}")

    def get_batch(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        out = np.empty((indices.shape[0], self.dim), dtype=self.parts[0].dtype)
        part_ids = np.searchsorted(self.cum, indices, side="right") - 1

        for p in range(len(self.parts)):
            mask = (part_ids == p)
            if not mask.any():
                continue
            sel = indices[mask] - self.cum[p]
            out[mask] = self.parts[p][sel]
        return out

# =========================
# 2) Your TripletDataset + collate_fn (same behavior)
# =========================
class TripletDataset(Dataset):
    def __init__(self, embeddings, docking_scores, max_tries: int = 20,
                 semi_hard_pos_window: int = 0, semi_hard_neg_window: int = 0):
        self.embeddings = embeddings
        self.docking_scores = docking_scores
        self.max_tries = max_tries
        self.semi_hard_pos_window = int(max(0, semi_hard_pos_window))
        self.semi_hard_neg_window = int(max(0, semi_hard_neg_window))
        # precompute rank order for O(1) sampling
        self.order = np.argsort(self.docking_scores)
        self.rank = np.empty_like(self.order)
        self.rank[self.order] = np.arange(len(self.order))

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        q_idx = int(idx)
        n_total = len(self.docking_scores)

        # O(1) sampling using rank order (lower score = better)
        r = int(self.rank[q_idx])
        if r > 0:
            if self.semi_hard_pos_window > 0:
                lo = max(0, r - self.semi_hard_pos_window)
                hi = r
                p_idx = int(self.order[random.randrange(lo, hi)])
            else:
                p_idx = int(self.order[random.randrange(0, r)])
        else:
            p_idx = q_idx
        if r < n_total - 1:
            if self.semi_hard_neg_window > 0:
                lo = r + 1
                hi = min(n_total, r + 1 + self.semi_hard_neg_window)
                n_idx = int(self.order[random.randrange(lo, hi)])
            else:
                n_idx = int(self.order[random.randrange(r + 1, n_total)])
        else:
            n_idx = q_idx

        # deterministic fallback (guarantee valid)
        if (p_idx == q_idx) or (n_idx == q_idx) or (p_idx == n_idx):
            p_idx = (q_idx + 1) % n_total
            n_idx = (q_idx + 2) % n_total
            if p_idx == n_idx:
                n_idx = (q_idx + 3) % n_total

        q_embedding = torch.from_numpy(self.embeddings[q_idx].astype(np.float32, copy=False))
        p_embedding = torch.from_numpy(self.embeddings[p_idx].astype(np.float32, copy=False))
        n_embedding = torch.from_numpy(self.embeddings[n_idx].astype(np.float32, copy=False))
        return q_embedding.clone(), p_embedding.clone(), n_embedding.clone()

def collate_fn(batch):
    query_batch = torch.stack([b[0] for b in batch])
    positive_batch = torch.stack([b[1] for b in batch])
    negative_batch = torch.stack([b[2] for b in batch])
    return query_batch, positive_batch, negative_batch

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def _autocast_ctx(device, enabled):
    device_type = device.type if isinstance(device, torch.device) else str(device)
    return autocast(device_type=device_type, dtype=torch.float16, enabled=enabled)

def _loader_perf_kwargs(num_workers: int, pin_memory: bool, persistent_workers: bool, prefetch_factor: int):
    kwargs = {
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None and int(prefetch_factor) > 0:
            kwargs["prefetch_factor"] = int(prefetch_factor)
    return kwargs

def prepare_dataloaders_ddp(df, batch_size, split_ratio, rank, world, num_workers=4, seed=2025,
                            pin_memory=True, persistent_workers=True, prefetch_factor=3,
                            semi_hard_pos_window: int = 0, semi_hard_neg_window: int = 0):
    # faster than .tolist()
    embeddings = np.vstack(df["combined_embedding"].values).astype(np.float32, copy=False)
    scores = df["Score"].to_numpy(np.float32, copy=False)

    dataset = TripletDataset(
        embeddings, scores,
        semi_hard_pos_window=semi_hard_pos_window,
        semi_hard_neg_window=semi_hard_neg_window
    )

    n = len(dataset)
    train_size = int(split_ratio * n)
    val_size = n - train_size

    g = torch.Generator()
    g.manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=g)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    val_sampler   = DistributedSampler(val_dataset,   num_replicas=world, rank=rank, shuffle=False, drop_last=False)


    loader_kwargs = _loader_perf_kwargs(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        **loader_kwargs,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        collate_fn=collate_fn,
        **loader_kwargs,
    )
    return train_loader, val_loader, train_sampler

# =========================
# 3) RankModel (your architecture)
# =========================
class RankModel(nn.Module):
    def __init__(self, input_dim, num_layers=3, dropout=0.4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()

        current_dim = input_dim
        next_dim = None

        for i in range(num_layers):
            if i == 0:
                next_dim = input_dim * 2
            else:
                next_dim = max(int(next_dim / 2), 256)

            self.layers.append(nn.Linear(current_dim, next_dim))
            self.layers.append(nn.LeakyReLU())
            self.layers.append(nn.Dropout(dropout))

            if current_dim != next_dim:
                self.skip_layers.append(nn.Linear(current_dim, next_dim))
            else:
                self.skip_layers.append(nn.Identity())

            current_dim = next_dim

        self.regressor = nn.Linear(current_dim, 1)

    def encode(self, x):
        residual = x
        for i in range(len(self.skip_layers)):
            idx = i * 3
            linear = self.layers[idx]
            act = self.layers[idx + 1]
            drop = self.layers[idx + 2]

            x = linear(x)
            x = act(x)
            x = drop(x)

            x = x + self.skip_layers[i](residual)
            residual = x
        return x

    def forward(self, x):
        x = self.encode(x)
        return self.regressor(x)

# =========================
# 4) RankDNN loss (KEEP)
# =========================
def rankdnn_loss(q, p, n, margin=0.1, lambda_rank=0.1):
    # Explicitly enforce the desired ordering: p < q < n.
    loss_pq = torch.relu(margin + (p - q))
    loss_qn = torch.relu(margin + (q - n))
    order_loss = (loss_pq + loss_qn).mean()

    # Enforce an explicit global separation: (n - p) should be >= 2 * margin.
    desired_gap = 2.0 * margin
    gap_loss = F.softplus(desired_gap - (n - p)).mean()
    return order_loss + lambda_rank * gap_loss

def pairwise_loss(q, p, n):
    # enforce p < q < n with logistic loss
    return F.softplus(p - q).mean() + F.softplus(q - n).mean()

def forward_triplet_scores(model, q, p, n):
    x = torch.cat([q, p, n], dim=0)
    y = model(x).squeeze(-1)
    b = q.shape[0]
    q_s, p_s, n_s = y.split((b, b, b), dim=0)
    return q_s, p_s, n_s

# =========================
# 5) Train M2 with AMP+DDP (handle None batch)
# =========================
def train_rankdnn_ddp(
    model_ddp, train_loader, val_loader, optimizer, device,
    margin=0.3, lambda_rank=0.01,
    epochs=500, early_stop=True, patience=60,
    use_amp=True, rank=0, train_sampler=None
):
    """
    DDP-safe training:
    - All ranks run the same number of steps per epoch (no None batches).
    - Validation metrics are synchronized across ranks.
    - Early stopping decision is broadcast from rank0 so all ranks break together.
    - Best checkpoint is stored on rank0 and broadcast to all ranks at end.
    """
    scaler = GradScaler(enabled=use_amp)

    best_val = float("inf")
    best_state = None
    bad = 0

    world = dist.get_world_size() if dist.is_initialized() else 1

    for ep in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(ep)

        # -----------------
        # train
        # -----------------
        model_ddp.train()
        tr_loss = 0.0
        tr_steps = 0

        for q, p, n in train_loader:
            q = q.to(device, non_blocking=True)
            p = p.to(device, non_blocking=True)
            n = n.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with _autocast_ctx(device, use_amp):
                q_s, p_s, n_s = forward_triplet_scores(model_ddp, q, p, n)
                loss = rankdnn_loss(q_s, p_s, n_s, margin=margin, lambda_rank=lambda_rank)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tr_loss += float(loss.detach().cpu())
            tr_steps += 1

        avg_tr = tr_loss / max(1, tr_steps)

        # -----------------
        # val
        # -----------------
        model_ddp.eval()
        va_loss = 0.0
        va_steps = 0

        with torch.no_grad():
            for q, p, n in val_loader:
                q = q.to(device, non_blocking=True)
                p = p.to(device, non_blocking=True)
                n = n.to(device, non_blocking=True)

                with _autocast_ctx(device, use_amp):
                    q_s, p_s, n_s = forward_triplet_scores(model_ddp, q, p, n)
                    loss = rankdnn_loss(q_s, p_s, n_s, margin=margin, lambda_rank=lambda_rank)

                va_loss += float(loss.detach().cpu())
                va_steps += 1

        avg_va = va_loss / max(1, va_steps)

        # -----------------
        # sync metrics across ranks (important for consistent stopping)
        # -----------------
        if dist.is_initialized():
            t = torch.tensor([avg_tr, avg_va], device=device, dtype=torch.float32)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t /= world
            avg_tr = float(t[0].item())
            avg_va = float(t[1].item())

        if rank == 0 and (ep % 10 == 0 or ep == epochs - 1):
            print(f"[M2] ep {ep+1:4d} | train {avg_tr:.6f} | val {avg_va:.6f}")

        # -----------------
        # early stopping: decide on rank0 and broadcast stop flag
        # -----------------
        stop = False
        improved = False

        if early_stop:
            if avg_va < best_val:
                best_val = avg_va
                improved = True
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    stop = True

            # rank0 saves checkpoint when improved
            if improved and rank == 0:
                best_state = {k: v.detach().cpu() for k, v in unwrap_model(model_ddp).state_dict().items()}

            # broadcast stop decision so all ranks break together
            if dist.is_initialized():
                st = torch.tensor([1 if stop else 0], device=device, dtype=torch.int32)
                dist.broadcast(st, src=0)
                stop = bool(st.item())

        if stop:
            if rank == 0:
                print(f"\n[M2] Early stopping at ep {ep+1}")
            break

    # -----------------
    # broadcast best_state to all ranks and restore
    # -----------------
    if dist.is_initialized():
        best_state = bcast_object(best_state, src=0)
        dist_barrier()

    if best_state is not None:
        unwrap_model(model_ddp).load_state_dict(best_state, strict=True)

    dist_barrier()

    if rank == 0 and best_state is not None:
        print(f"\n[M2] restored best val={best_val:.4f}")

def train_pairwise_ddp(
    model_ddp, train_loader, val_loader, optimizer, device,
    epochs=500, early_stop=True, patience=60,
    use_amp=True, rank=0, train_sampler=None
):
    scaler = GradScaler(enabled=use_amp)
    best_val = float("inf")
    best_state = None
    bad = 0
    world = dist.get_world_size() if dist.is_initialized() else 1

    for ep in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(ep)

        model_ddp.train()
        tr_loss = 0.0
        tr_steps = 0
        for q, p, n in train_loader:
            q = q.to(device, non_blocking=True)
            p = p.to(device, non_blocking=True)
            n = n.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with _autocast_ctx(device, use_amp):
                q_s, p_s, n_s = forward_triplet_scores(model_ddp, q, p, n)
                loss = pairwise_loss(q_s, p_s, n_s)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tr_loss += float(loss.detach().cpu())
            tr_steps += 1

        # -----------------
        # validate
        # -----------------
        model_ddp.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for q, p, n in val_loader:
                q = q.to(device, non_blocking=True)
                p = p.to(device, non_blocking=True)
                n = n.to(device, non_blocking=True)
                with _autocast_ctx(device, use_amp):
                    q_s, p_s, n_s = forward_triplet_scores(model_ddp, q, p, n)
                    loss = pairwise_loss(q_s, p_s, n_s)
                val_loss += float(loss.detach().cpu())
                val_steps += 1

        if dist.is_initialized():
            t = torch.tensor([tr_loss, tr_steps, val_loss, val_steps], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            tr_loss, tr_steps, val_loss, val_steps = t.tolist()

        tr_avg = tr_loss / max(tr_steps, 1)
        val_avg = val_loss / max(val_steps, 1)

        if rank == 0 and (ep % 10 == 0 or ep == epochs - 1):
            print(f"[M2] ep {ep+1:4d} | train {tr_avg:.6f} | val {val_avg:.6f}")

        if val_avg < best_val - 1e-6:
            best_val = val_avg
            if rank == 0:
                best_state = {k: v.detach().cpu() for k, v in unwrap_model(model_ddp).state_dict().items()}
            bad = 0
        else:
            bad += 1

        stop = False
        if early_stop and bad >= patience:
            stop = True
        if dist.is_initialized():
            st = torch.tensor([1 if stop else 0], device=device, dtype=torch.int32)
            dist.broadcast(st, src=0)
            stop = bool(st.item())
        if stop:
            if rank == 0:
                print(f"\n[M2] Early stopping at ep {ep+1}")
            break

    if dist.is_initialized():
        best_state = bcast_object(best_state, src=0)
        dist_barrier()
    if best_state is not None:
        unwrap_model(model_ddp).load_state_dict(best_state, strict=True)
    dist_barrier()
    if rank == 0 and best_state is not None:
        print(f"\n[M2] restored best val={best_val:.4f}")

def prepare_regression_dataloaders_ddp(df, batch_size, split_ratio, rank, world, num_workers=4, seed=2025,
                                       pin_memory=True, persistent_workers=True, prefetch_factor=3):
    embeddings = np.vstack(df["combined_embedding"].values).astype(np.float32, copy=False)
    scores = df["Score"].to_numpy(np.float32, copy=False)

    dataset = TensorDataset(
        torch.from_numpy(embeddings),
        torch.from_numpy(scores)
    )

    n = len(dataset)
    train_size = int(split_ratio * n)
    val_size = n - train_size

    g = torch.Generator()
    g.manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=g)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    val_sampler   = DistributedSampler(val_dataset,   num_replicas=world, rank=rank, shuffle=False, drop_last=False)

    loader_kwargs = _loader_perf_kwargs(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=train_sampler, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        sampler=val_sampler, **loader_kwargs,
    )
    return train_loader, val_loader, train_sampler

def train_regression_ddp(
    model_ddp, train_loader, val_loader, optimizer, device,
    epochs=200, early_stop=True, patience=50,
    use_amp=True, rank=0, train_sampler=None
):
    scaler = GradScaler(enabled=use_amp)
    best_val = float("inf")
    best_state = None
    bad = 0
    world = dist.get_world_size() if dist.is_initialized() else 1

    for ep in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(ep)

        model_ddp.train()
        tr_loss = 0.0
        tr_steps = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).unsqueeze(-1)

            optimizer.zero_grad(set_to_none=True)
            with _autocast_ctx(device, use_amp):
                pred = model_ddp(xb)
                loss = F.mse_loss(pred, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tr_loss += float(loss.detach().cpu())
            tr_steps += 1

        model_ddp.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).unsqueeze(-1)
                with _autocast_ctx(device, use_amp):
                    pred = model_ddp(xb)
                    loss = F.mse_loss(pred, yb)
                val_loss += float(loss.detach().cpu())
                val_steps += 1

        if dist.is_initialized():
            t = torch.tensor([tr_loss, tr_steps, val_loss, val_steps], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            tr_loss, tr_steps, val_loss, val_steps = t.tolist()

        tr_avg = tr_loss / max(tr_steps, 1)
        val_avg = val_loss / max(val_steps, 1)

        if rank == 0 and (ep % 10 == 0 or ep == epochs - 1):
            print(f"[M2] ep {ep+1:4d} | train {tr_avg:.6f} | val {val_avg:.6f}")

        if val_avg < best_val - 1e-6:
            best_val = val_avg
            if rank == 0:
                best_state = {k: v.detach().cpu() for k, v in unwrap_model(model_ddp).state_dict().items()}
            bad = 0
        else:
            bad += 1

        stop = False
        if early_stop and bad >= patience:
            stop = True
        if dist.is_initialized():
            st = torch.tensor([1 if stop else 0], device=device, dtype=torch.int32)
            dist.broadcast(st, src=0)
            stop = bool(st.item())
        if stop:
            if rank == 0:
                print(f"\n[M2] Early stopping at ep {ep+1}")
            break

    if dist.is_initialized():
        best_state = bcast_object(best_state, src=0)
        dist_barrier()
    if best_state is not None:
        unwrap_model(model_ddp).load_state_dict(best_state, strict=True)
    dist_barrier()
    if rank == 0 and best_state is not None:
        print(f"\n[M2] restored best val={best_val:.4f}")
# =========================
# 6) Sharded prediction: M2 raw
# =========================
@torch.no_grad()
def predict_m2_raw_sharded(model, feats: PartMemmap, all_indices: np.ndarray, rank, world,
                           batch_size=4096, device="cuda"):
    model.eval()
    local_ids = shard_indices(all_indices, rank, world)
    local_out = np.empty((len(local_ids),), dtype=np.float32)

    for s in range(0, len(local_ids), batch_size):
        e = min(s + batch_size, len(local_ids))
        ids = local_ids[s:e]
        xb = feats.get_batch(ids).astype(np.float32, copy=False)
        xb = torch.from_numpy(xb).to(device, non_blocking=True)
        y = model(xb).squeeze(-1).float().detach().cpu().numpy().astype(np.float32)
        local_out[s:e] = y

    return local_ids, local_out

# =========================
# 7) SVGP (DDP train + sharded sigma predict)
# =========================
def standardize_x(X):
    mu = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-12
    return (X - mu) / std, mu, std

def apply_x_std(X, mu, std):
    return (X - mu) / (std + 1e-12)

def standardize_y(y):
    mu = float(np.mean(y))
    std = float(np.std(y) + 1e-12)
    return (y - mu) / std, mu, std

class SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, kernel="matern", nu=1.5):
        q = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        strat = gpytorch.variational.VariationalStrategy(self, inducing_points, q, learn_inducing_locations=True)
        super().__init__(strat)
        self.mean_module = gpytorch.means.ConstantMean()
        base = gpytorch.kernels.MaternKernel(nu=nu) if kernel == "matern" else gpytorch.kernels.RBFKernel()
        self.covar_module = gpytorch.kernels.ScaleKernel(base)

    def forward(self, x):
        mean = self.mean_module(x)
        cov = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, cov)

def fit_svgp_ddp(
    X_obs, y_obs, device, rank, world,
    M=1024, batch_size=4096, iters=1000, lr=0.01,
    kernel="matern", nu=1.5, seed=2025, use_amp=True, print_every=100,
    num_workers=2, pin_memory=True, persistent_workers=False, prefetch_factor=2
):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    Xs, x_mu, x_std = standardize_x(X_obs.astype(np.float32))
    ys, y_mu, y_std = standardize_y(y_obs.astype(np.float32))

    # inducing points
    if Xs.shape[0] <= M:
        Z_np = Xs
    else:
        Z_np = Xs[np.random.choice(Xs.shape[0], M, replace=False)]
    Z = torch.tensor(Z_np, dtype=torch.float32, device=device)

    model = SVGPModel(Z, kernel=kernel, nu=nu).to(device)
    lik = gpytorch.likelihoods.GaussianLikelihood().to(device)

    if world > 1:
        ddp_kwargs = {
            "find_unused_parameters": False,
            "broadcast_buffers": False,
        }
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [device.index]
            ddp_kwargs["output_device"] = device.index
        model_ddp = DDP(model, **ddp_kwargs)
        lik_ddp = DDP(lik, **ddp_kwargs)
    else:
        model_ddp = model
        lik_ddp = lik

    model_ddp.train(); lik_ddp.train()

    train_x = torch.from_numpy(Xs).to(dtype=torch.float32)
    train_y = torch.from_numpy(ys).to(dtype=torch.float32)
    ds = TensorDataset(train_x, train_y)

    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, drop_last=False)
    svgp_loader_kwargs = _loader_perf_kwargs(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    dl = DataLoader(ds, batch_size=batch_size, sampler=sampler, **svgp_loader_kwargs)

    opt = torch.optim.Adam(
        [{'params': model_ddp.parameters()}, {'params': lik_ddp.parameters()}],
        lr=lr
    )

    elbo = gpytorch.mlls.VariationalELBO(unwrap_model(lik_ddp), unwrap_model(model_ddp), num_data=train_y.numel())
    scaler = GradScaler(enabled=use_amp)

    for it in range(1, iters + 1):
        sampler.set_epoch(it)
        total = 0.0

        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with _autocast_ctx(device, use_amp):
                out = model_ddp(xb)
                loss = -elbo(out, yb)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total += float(loss.detach().cpu())

        if rank == 0 and (it == 1 or it % print_every == 0 or it == iters):
            print(f"[SVGP] iter {it:4d}/{iters} loss_sum={total:.4f}")

    dist_barrier()
    model = unwrap_model(model_ddp)
    lik = unwrap_model(lik_ddp)
    model.eval(); lik.eval()
    return model, lik, x_mu, x_std, y_mu, y_std

@torch.no_grad()
def svgp_sigma_sharded(model, likelihood, feats: PartMemmap, all_indices: np.ndarray,
                       x_mu, x_std, device, rank, world, batch_size=8192):
    model.eval(); likelihood.eval()
    local_ids = shard_indices(all_indices, rank, world)
    local_sig = np.empty((len(local_ids),), dtype=np.float32)

    for s in range(0, len(local_ids), batch_size):
        e = min(s + batch_size, len(local_ids))
        ids = local_ids[s:e]
        Xb = feats.get_batch(ids).astype(np.float32, copy=False)
        Xb = apply_x_std(Xb, x_mu, x_std)
        xb = torch.tensor(Xb, dtype=torch.float32, device=device).contiguous()

        with gpytorch.settings.fast_pred_var(True), gpytorch.settings.max_preconditioner_size(0):
            f_pred = model(xb)
        var_b = f_pred.variance.detach().cpu().numpy()
        local_sig[s:e] = np.sqrt(np.maximum(var_b, 1e-12)).astype(np.float32)

    return local_ids, local_sig

@torch.no_grad()
def svgp_mu_sigma_sharded(model, likelihood, feats: PartMemmap, all_indices: np.ndarray,
                          x_mu, x_std, device, rank, world, batch_size=8192):
    model.eval(); likelihood.eval()
    local_ids = shard_indices(all_indices, rank, world)
    local_mu = np.empty((len(local_ids),), dtype=np.float32)
    local_sig = np.empty((len(local_ids),), dtype=np.float32)

    for s in range(0, len(local_ids), batch_size):
        e = min(s + batch_size, len(local_ids))
        ids = local_ids[s:e]
        Xb = feats.get_batch(ids).astype(np.float32, copy=False)
        Xb = apply_x_std(Xb, x_mu, x_std)
        xb = torch.tensor(Xb, dtype=torch.float32, device=device).contiguous()

        with gpytorch.settings.fast_pred_var(True), gpytorch.settings.max_preconditioner_size(0):
            f_pred = model(xb)
        mu_b = f_pred.mean.detach().cpu().numpy().astype(np.float32)
        var_b = f_pred.variance.detach().cpu().numpy()
        local_mu[s:e] = mu_b
        local_sig[s:e] = np.sqrt(np.maximum(var_b, 1e-12)).astype(np.float32)

    return local_ids, local_mu, local_sig

# =========================
# 8) Acquisition
# =========================
def _phi(z):
    return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z * z)

def _Phi(z):
    z = torch.as_tensor(np.asarray(z), dtype=torch.float32)
    return (0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))).cpu().numpy()

def apply_x_std_np(X, mu, std, eps=1e-12):
    return (X - mu) / (std + eps)

def apply_y_std_np(y, mu, std, eps=1e-12):
    return (y - mu) / (std + eps)

def min_max_scale_np(preds, min_val, max_val, eps=1e-12):
    preds = np.asarray(preds, dtype=np.float32)
    pmin, pmax = float(preds.min()), float(preds.max())
    s01 = (preds - pmin) / ((pmax - pmin) + eps)
    return s01 * (max_val - min_val) + min_val

def acq_scores(mu_std, sig_std, kind="lcb", f_best_std=None, kappa=2.0, xi=0.0, minimize=True,
               sigma_floor=0.05, z_clip=8.0):
    mu = np.asarray(mu_std, dtype=np.float32)
    s = np.asarray(sig_std, dtype=np.float32)

    if kind in ("poi", "eoi"):
        s = np.maximum(s, sigma_floor)
    else:
        s = s + 1e-12

    if kind == "greedy":
        return -mu if minimize else mu
    if kind == "lcb":
        return -(mu - kappa * s) if minimize else (mu + kappa * s)
    if kind == "ucb":
        return -(mu + kappa * s) if minimize else (mu + kappa * s)

    if kind in ("poi", "eoi"):
        if f_best_std is None:
            raise ValueError("poi/eoi need f_best_std")
        imp = (f_best_std - mu - xi) if minimize else (mu - f_best_std - xi)
        z = imp / (s + 1e-12)
        z = np.clip(z, -z_clip, z_clip)
        if kind == "poi":
            return _Phi(z)
        return imp * _Phi(z) + s * _phi(z)

    raise ValueError(f"unknown acq kind: {kind}")

def calibrate_mu_for_ei_pi(mu_pool_raw, obs_pred, obs_y,
                           do_calibrate=True, do_minmax=True,
                           minmax_range="observed",
                           global_y_min=None, global_y_max=None):
    mu = mu_pool_raw.astype(np.float32)
    if do_calibrate and (np.std(obs_pred) >= 1e-8):
        a, b = np.polyfit(obs_pred.astype(np.float32), obs_y.astype(np.float32), deg=1)
        mu = a * mu + b
    if do_minmax:
        if minmax_range == "global":
            if global_y_min is None or global_y_max is None:
                raise ValueError("global_y_min/global_y_max required when minmax_range='global'")
            y_lo, y_hi = float(global_y_min), float(global_y_max)
        else:
            y_lo, y_hi = float(np.min(obs_y)), float(np.max(obs_y))
        mu = min_max_scale_np(mu, y_lo, y_hi)
    return mu

@torch.no_grad()
def svgp_predict_mu_sigma_ids(
    model, likelihood, feats: PartMemmap, ids: np.ndarray,
    x_mu, x_std, device="cuda", batch_size=65536,
    use_likelihood=True, fast_pred_var=False
):
    model.eval(); likelihood.eval()
    mu_all = np.empty((len(ids),), dtype=np.float32)
    std_all = np.empty((len(ids),), dtype=np.float32)

    for s in range(0, len(ids), batch_size):
        e = min(s + batch_size, len(ids))
        sel = ids[s:e]
        xb = feats.get_batch(sel).astype(np.float32, copy=False)
        xb = apply_x_std_np(xb, x_mu, x_std)
        xb = torch.from_numpy(xb).to(device, non_blocking=True)
        with gpytorch.settings.fast_pred_var(fast_pred_var), gpytorch.settings.max_preconditioner_size(0):
            if use_likelihood:
                pred = likelihood(model(xb))
                mu_b = pred.mean.detach().cpu().numpy()
                var_b = pred.variance.detach().cpu().numpy()
            else:
                f_pred = model(xb)
                mu_b = f_pred.mean.detach().cpu().numpy()
                var_b = f_pred.variance.detach().cpu().numpy()
        mu_all[s:e] = mu_b.astype(np.float32, copy=False)
        std_all[s:e] = np.sqrt(np.maximum(var_b, 1e-12)).astype(np.float32, copy=False)
    return mu_all, std_all

def build_mixed_candidates(
    mu_pool, remaining_ids, *,
    M, feats, device, model_svgp, lik_svgp, x_mu, x_std,
    frac_mu=0.6, frac_sig=0.3, frac_rnd=0.1,
    sigma_batch_size=65536,
    use_likelihood_sigma=True,
    fast_pred_var=False,
    seed=2025
):
    rng = np.random.default_rng(seed)
    N = len(mu_pool)
    M = int(min(M, N))
    if M <= 0:
        return np.array([], dtype=np.int64)

    m_mu = max(1, int(M * frac_mu))
    m_sig = max(1, int(M * frac_sig))
    m_rnd = max(0, M - m_mu - m_sig)

    idx_mu = np.argpartition(mu_pool, m_mu - 1)[:m_mu]
    idx_rnd = rng.choice(N, size=m_rnd, replace=False) if m_rnd > 0 else np.array([], dtype=np.int64)

    base = np.unique(np.concatenate([idx_mu, idx_rnd]))
    base_ids = remaining_ids[base]

    _, sig_base = svgp_predict_mu_sigma_ids(
        model_svgp, lik_svgp,
        feats, base_ids,
        x_mu, x_std,
        device=device,
        batch_size=sigma_batch_size,
        use_likelihood=use_likelihood_sigma,
        fast_pred_var=fast_pred_var
    )

    if base.size <= m_sig:
        idx_sig = base
    else:
        local = np.argpartition(-sig_base, m_sig - 1)[:m_sig]
        idx_sig = base[local]

    cand = np.unique(np.concatenate([idx_mu, idx_sig, idx_rnd]))
    if cand.size > M:
        cand = cand[:M]
    if cand.size < M:
        need = M - cand.size
        rest = np.setdiff1d(np.arange(N), cand, assume_unique=False)
        if rest.size > 0 and need > 0:
            extra = rng.choice(rest, size=min(need, rest.size), replace=False)
            cand = np.unique(np.concatenate([cand, extra]))
            if cand.size > M:
                cand = cand[:M]
    return cand.astype(np.int64)

def safe_scaffold_ratio(smiles_list):
    if Chem is None or MurckoScaffold is None:
        return 0, 0, 0.0
    n_valid = 0
    scaffolds = set()
    for smi in smiles_list:
        if not isinstance(smi, str):
            continue
        smi = smi.strip()
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None:
            continue
        scaf_smi = Chem.MolToSmiles(scaf)
        if not scaf_smi:
            continue
        n_valid += 1
        scaffolds.add(scaf_smi)
    n_unique = len(scaffolds)
    ratio = (n_unique / n_valid) if n_valid > 0 else 0.0
    return n_valid, n_unique, ratio

def eval_topk_overlap_hits(scores, preds, ks, minimize=True):
    scores = np.asarray(scores)
    preds = np.asarray(preds)
    N = len(scores)
    rows = []
    for k in ks:
        k = int(k)
        if k <= 0 or k > N:
            continue
        if minimize:
            true_idx = np.argpartition(scores, k - 1)[:k]
            pred_idx = np.argpartition(preds, k - 1)[:k]
        else:
            true_idx = np.argpartition(-scores, k - 1)[:k]
            pred_idx = np.argpartition(-preds, k - 1)[:k]
        overlap = len(set(true_idx.tolist()) & set(pred_idx.tolist()))
        rows.append((k, overlap, overlap / k))
    return rows

def eval_topk_scaffold_diversity(smiles, preds, ks, minimize=True):
    smiles = np.asarray(smiles)
    preds = np.asarray(preds)
    N = len(smiles)
    rows = []
    for k in ks:
        k = int(k)
        if k <= 0 or k > N:
            continue
        if minimize:
            top_idx = np.argpartition(preds, k - 1)[:k]
        else:
            top_idx = np.argpartition(-preds, k - 1)[:k]
        top_smiles = smiles[top_idx].tolist()
        n_valid, n_unique, ratio = safe_scaffold_ratio(top_smiles)
        rows.append((k, n_valid, n_unique, ratio))
    return rows


def compute_enrichment_factor(n_active_found, n_selected, n_active_total, n_total):
    if n_selected <= 0 or n_total <= 0 or n_active_total <= 0:
        return 0.0
    selected_rate = float(n_active_found) / float(n_selected)
    base_rate = float(n_active_total) / float(n_total)
    if base_rate <= 0.0:
        return 0.0
    return selected_rate / base_rate

def select_topk_with_temperature(order_best_first: np.ndarray, k: int, pool_mul: float, temp: float, seed: int) -> np.ndarray:
    """
    Temperature sampling within a top-ranked pool.
    - pool_mul=1.0 or temp<=0 -> deterministic top-k.
    - Larger pool_mul/temp -> more stochastic picks (still top-biased).
    """
    n = len(order_best_first)
    if k <= 0 or n == 0:
        return np.zeros((0,), dtype=np.int64)

    k = min(k, n)
    pool_mul = float(max(1.0, pool_mul))
    temp = float(temp)
    pool_k = min(n, max(k, int(math.ceil(k * pool_mul))))
    pool = order_best_first[:pool_k]

    if pool_k == k or temp <= 0.0:
        return pool[:k]

    ranks = np.arange(pool_k, dtype=np.float64)
    logits = -ranks / (max(1e-8, temp) * float(pool_k))
    logits -= np.max(logits)
    probs = np.exp(logits)
    probs /= np.sum(probs)

    rng = np.random.default_rng(seed)
    chosen_local = rng.choice(pool_k, size=k, replace=False, p=probs)
    return pool[chosen_local.astype(np.int64, copy=False)]

# =========================
# 9) Main: end-to-end DDP BO
# =========================
def run_bo(args):
    local_rank, rank, world = ddp_setup(timeout_minutes=args.ddp_timeout_min)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)

    if rank == 0:
        df_pool = pd.read_csv(args.smiles_csv)
        if "Score" not in df_pool.columns or "SMILES" not in df_pool.columns:
            raise ValueError("pool csv must contain columns: Score, SMILES")
        df_pool = df_pool.reset_index(drop=True)
        pool_scores = df_pool["Score"].to_numpy(np.float32, copy=False)
        pool_smiles = df_pool["SMILES"].to_numpy()
        pool_labels = df_pool["label"].to_numpy() if "label" in df_pool.columns else None
        n_total = len(df_pool)
        n_active_total = int(pd.Series(pool_labels).astype(float).eq(1).sum()) if pool_labels is not None else 0
    else:
        df_pool = None
        pool_scores = None
        pool_smiles = None
        pool_labels = None
        n_total = None
        n_active_total = 0

    n_total = bcast_object(n_total, src=0)
    n_active_total = bcast_object(n_active_total, src=0)

    feats = PartMemmap(args.emb_dir, pattern=args.emb_pattern)
    if feats.total != n_total:
        raise ValueError(f"Embedding rows ({feats.total}) != SMILES rows ({n_total}).")

    df_init = pd.read_csv(args.init_csv).copy()
    if "sample_id" not in df_init.columns:
        raise ValueError("init csv must contain sample_id")

    init_ids = df_init["sample_id"].to_numpy(np.int64)
    if "Score" not in df_init.columns:
        init_scores = pool_scores[init_ids] if rank == 0 else None
        init_scores = bcast_object(init_scores, src=0)
        df_init["Score"] = np.asarray(init_scores, dtype=np.float32)
    if "SMILES" not in df_init.columns:
        if rank == 0:
            df_init["SMILES"] = pool_smiles[init_ids]
        else:
            df_init["SMILES"] = ""
    if "label" not in df_init.columns and pool_labels is not None:
        init_labels = pool_labels[init_ids] if rank == 0 else None
        init_labels = bcast_object(init_labels, src=0)
        df_init["label"] = init_labels

    init_feats = feats.get_batch(init_ids)
    df_init["combined_embedding"] = [row.copy() for row in init_feats]
    df_init["selection_round"] = 0
    df_labeled = df_init.reset_index(drop=True)

    selected_set = set(df_labeled["sample_id"].tolist())
    remaining_mask = np.ones(int(n_total), dtype=bool)
    remaining_mask[init_ids] = False
    remaining_ids = np.flatnonzero(remaining_mask).astype(np.int64, copy=False)

    select_per_round = int(math.ceil(n_total * args.select_frac))

    if rank == 0:
        print(f"[BO] world={world} | total={n_total:,} | init={len(df_labeled):,} | remaining={len(remaining_ids):,}")
        print(f"[BO] rounds={args.rounds} | select/round={select_per_round:,} ({args.select_frac*100:.3f}%) | acq={args.acq}")
        if Chem is None or MurckoScaffold is None:
            print("[BO] RDKit not available: scaffold diversity will be 0.")
        if args.eval_retrieval:
            print(f"=== Retrieval + Diversity (Predicted TopK) === {args.acq.upper()}")

    # precompute evaluation ks
    top1p_k = max(1, int(math.ceil(n_total * 0.01)))
    eval_ks = [1000, 5000, top1p_k]

    # SVGP (fit once)
    model_svgp = None
    lik_svgp = None
    x_mu = x_std = y_mu = y_std = None
    f_best_std = None

    if args.acq != "greedy":
        X_obs = np.vstack(df_labeled["combined_embedding"].values).astype(np.float32, copy=False)
        y_obs = df_labeled["Score"].to_numpy(np.float32, copy=False)
        if rank == 0:
            print("\n[BO] Fit SVGP (DDP)...")
        model_svgp, lik_svgp, x_mu, x_std, y_mu, y_std = fit_svgp_ddp(
            X_obs, y_obs,
            device=device, rank=rank, world=world,
            M=args.svgp_M, batch_size=args.svgp_batch, iters=args.svgp_iters, lr=args.svgp_lr,
            kernel="matern", nu=1.5, seed=args.seed, use_amp=bool(args.svgp_amp), print_every=args.svgp_print_every,
            num_workers=args.svgp_num_workers,
            pin_memory=bool(args.pin_memory),
            persistent_workers=bool(args.persistent_workers),
            prefetch_factor=args.prefetch_factor
        )
        f_best_std = (float(np.min(y_obs)) - y_mu) / (y_std + 1e-12)

    input_dim = int(len(df_labeled.iloc[0]["combined_embedding"]))
    last_m2_state = None
    last_rf_model = None
    metrics_payload = {
        "config": {
            "smiles_csv": args.smiles_csv,
            "emb_dir": args.emb_dir,
            "init_csv": args.init_csv,
            "rounds": int(args.rounds),
            "select_frac": float(args.select_frac),
            "acq": args.acq,
            "m2_model": args.m2_model,
            "final_top_frac": float(args.final_top_frac),
            "seed": int(args.seed),
        },
        "round_metrics": [],
        "final_metrics": None,
    }

    def save_round_selection_csv(df_round, path):
        out_cols = ["selection_round", "sample_id", "SMILES", "Score"]
        if "label" in df_round.columns:
            out_cols.append("label")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df_round.loc[:, out_cols].to_csv(path, index=False)

    def save_prediction_ranking_csv(ids, preds, path, round_no=None, top_n=None):
        ids = np.asarray(ids, dtype=np.int64)
        preds = np.asarray(preds, dtype=np.float32)
        if ids.size == 0:
            out = pd.DataFrame(columns=["predicted_rank", "sample_id", "SMILES", "Score", "pred_score"])
        else:
            n = ids.size if top_n is None else min(int(top_n), ids.size)
            order = np.argsort(preds)[:n]  # lower predicted docking score is better
            ranked_ids = ids[order]
            out = pd.DataFrame({
                "predicted_rank": np.arange(1, n + 1, dtype=np.int64),
                "sample_id": ranked_ids,
                "SMILES": pool_smiles[ranked_ids],
                "Score": pool_scores[ranked_ids],
                "pred_score": preds[order],
            })
            if round_no is not None:
                out.insert(0, "round", int(round_no))
            if pool_labels is not None:
                out["label"] = pool_labels[ranked_ids]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        out.to_csv(path, index=False)

    if rank == 0:
        if args.round_selected_dir and args.save_initial_selection:
            round0_path = os.path.join(args.round_selected_dir, "round_00_initial.csv")
            save_round_selection_csv(df_labeled, round0_path)
            print(f"[BO] Saved initial selections to: {round0_path}")
        if args.round_cumulative_dir:
            cumulative0_path = os.path.join(args.round_cumulative_dir, "round_00_cumulative.csv")
            save_round_selection_csv(df_labeled, cumulative0_path)
            print(f"[BO] Saved round 0 cumulative selections to: {cumulative0_path}")

    def train_torch_model(current_df):
        m2 = RankModel(input_dim=input_dim, num_layers=args.m2_layers, dropout=args.m2_dropout).to(device)
        m2.apply(init_weights)
        if not args.no_compile:
            try:
                m2 = torch.compile(m2, mode="reduce-overhead")
            except Exception:
                pass
        if world > 1:
            ddp_kwargs = {
                "find_unused_parameters": False,
                "broadcast_buffers": False,
            }
            if device.type == "cuda":
                ddp_kwargs["device_ids"] = [local_rank]
                ddp_kwargs["output_device"] = local_rank
            ddp_m2 = DDP(m2, **ddp_kwargs)
        else:
            ddp_m2 = m2
        opt = torch.optim.AdamW(ddp_m2.parameters(), lr=args.m2_lr, weight_decay=args.m2_wd)

        if args.m2_model == "mlp":
            train_loader, val_loader, train_sampler = prepare_regression_dataloaders_ddp(
                current_df,
                batch_size=args.m2_train_bs,
                split_ratio=args.m2_split,
                rank=rank, world=world,
                num_workers=args.num_workers,
                seed=args.seed,
                pin_memory=bool(args.pin_memory),
                persistent_workers=bool(args.persistent_workers),
                prefetch_factor=args.prefetch_factor
            )
            train_regression_ddp(
                ddp_m2, train_loader, val_loader, opt, device,
                epochs=args.m2_epochs,
                early_stop=True, patience=args.m2_patience,
                use_amp=device.type == "cuda", rank=rank, train_sampler=train_sampler
            )
        elif args.m2_model == "pairwise":
            pos_win = args.semi_hard_pos_window
            neg_win = args.semi_hard_neg_window
            if args.semi_hard_pos_frac > 0:
                pos_win = max(pos_win, int(len(current_df) * args.semi_hard_pos_frac))
            if args.semi_hard_neg_frac > 0:
                neg_win = max(neg_win, int(len(current_df) * args.semi_hard_neg_frac))
            train_loader, val_loader, train_sampler = prepare_dataloaders_ddp(
                current_df,
                batch_size=args.m2_train_bs,
                split_ratio=args.m2_split,
                rank=rank, world=world,
                num_workers=args.num_workers,
                seed=args.seed,
                pin_memory=bool(args.pin_memory),
                persistent_workers=bool(args.persistent_workers),
                prefetch_factor=args.prefetch_factor,
                semi_hard_pos_window=pos_win,
                semi_hard_neg_window=neg_win
            )
            train_pairwise_ddp(
                ddp_m2, train_loader, val_loader, opt, device,
                epochs=args.m2_epochs,
                early_stop=True, patience=args.m2_patience,
                use_amp=device.type == "cuda", rank=rank, train_sampler=train_sampler
            )
        else:
            pos_win = args.semi_hard_pos_window
            neg_win = args.semi_hard_neg_window
            if args.semi_hard_pos_frac > 0:
                pos_win = max(pos_win, int(len(current_df) * args.semi_hard_pos_frac))
            if args.semi_hard_neg_frac > 0:
                neg_win = max(neg_win, int(len(current_df) * args.semi_hard_neg_frac))
            train_loader, val_loader, train_sampler = prepare_dataloaders_ddp(
                current_df,
                batch_size=args.m2_train_bs,
                split_ratio=args.m2_split,
                rank=rank, world=world,
                num_workers=args.num_workers,
                seed=args.seed,
                pin_memory=bool(args.pin_memory),
                persistent_workers=bool(args.persistent_workers),
                prefetch_factor=args.prefetch_factor,
                semi_hard_pos_window=pos_win,
                semi_hard_neg_window=neg_win
            )
            train_rankdnn_ddp(
                ddp_m2, train_loader, val_loader, opt, device,
                margin=args.margin, lambda_rank=args.lambda_rank,
                epochs=args.m2_epochs,
                early_stop=True, patience=args.m2_patience,
                use_amp=device.type == "cuda", rank=rank, train_sampler=train_sampler
            )

        if device.type == "cuda":
            torch.cuda.empty_cache()
        if rank == 0:
            state = {k: v.detach().cpu() for k, v in unwrap_model(ddp_m2).state_dict().items()}
        else:
            state = None
        state = bcast_object(state, src=0)
        return unwrap_model(ddp_m2), state

    def predict_on_embeddings_list(model, embeddings_list):
        model.eval()
        X = np.vstack(embeddings_list).astype(np.float32, copy=False)
        preds = np.empty((X.shape[0],), dtype=np.float32)
        for s in range(0, X.shape[0], args.m2_pred_bs):
            e = min(s + args.m2_pred_bs, X.shape[0])
            xb = torch.from_numpy(X[s:e]).to(device, non_blocking=True)
            y = model(xb).squeeze(-1).float().detach().cpu().numpy().astype(np.float32)
            preds[s:e] = y
        return preds

    round_summaries = []
    cumulative_rows = []
    for rd in range(1, args.rounds + 1):
        if rank == 0:
            print(f"\n[BO] Round {rd}/{args.rounds} | labeled={len(df_labeled):,} | remaining={len(remaining_ids):,}")

        # ---- train selection model ----
        if args.m2_model == "rf":
            if world != 1:
                raise RuntimeError("RandomForest model only supports single-process (world=1).")
            if rank == 0:
                try:
                    from sklearn.ensemble import RandomForestRegressor
                except Exception as e:
                    raise RuntimeError("scikit-learn is required for m2_model=rf") from e
                X = np.vstack(df_labeled["combined_embedding"].values).astype(np.float32, copy=False)
                y = df_labeled["Score"].to_numpy(np.float32, copy=False)
                rf = RandomForestRegressor(
                    n_estimators=args.rf_n_estimators,
                    max_depth=args.rf_max_depth,
                    min_samples_split=args.rf_min_samples_split,
                    min_samples_leaf=args.rf_min_samples_leaf,
                    n_jobs=args.rf_n_jobs,
                    random_state=args.seed,
                )
                rf.fit(X, y)
                last_rf_model = rf
        else:
            m2_model, last_m2_state = train_torch_model(df_labeled)

        # ---- predict on remaining ----
        if args.m2_model == "rf":
            if rank == 0:
                mu_raw = np.empty((len(remaining_ids),), dtype=np.float32)
                for s in range(0, len(remaining_ids), args.m2_pred_bs):
                    e = min(s + args.m2_pred_bs, len(remaining_ids))
                    ids = remaining_ids[s:e]
                    xb = feats.get_batch(ids).astype(np.float32, copy=False)
                    pred = last_rf_model.predict(xb).astype(np.float32, copy=False)
                    mu_raw[s:e] = pred
            else:
                mu_raw = None
        else:
            local_ids, local_mu = predict_m2_raw_sharded(
                m2_model, feats, remaining_ids, rank, world,
                batch_size=args.m2_pred_bs, device=device
            )
            all_ids, all_mu = gather_pairs_ids_vals(local_ids, local_mu, world)
            if rank == 0:
                mu_map = {int(i): float(v) for i, v in zip(all_ids.tolist(), all_mu.tolist())}
                mu_raw = np.array([mu_map[int(i)] for i in remaining_ids], dtype=np.float32)
            else:
                mu_raw = None

        # ---- selection ----
        if rank == 0:
            if args.round_prediction_dir:
                pred_top_n = max(eval_ks)
                pred_path = os.path.join(args.round_prediction_dir, f"round_{rd:02d}_prediction_top1pct.csv")
                save_prediction_ranking_csv(remaining_ids, mu_raw, pred_path, round_no=rd, top_n=pred_top_n)
                print(f"[BO] Saved round {rd} prediction ranking to: {pred_path}")

            if args.acq == "greedy":
                k = min(select_per_round, len(remaining_ids))
                if k <= 0:
                    picked_ids = np.zeros((0,), dtype=np.int64)
                else:
                    order = np.argsort(mu_raw)  # lower is better
                    if args.m2_model == "pairwise":
                        top_idx = select_topk_with_temperature(
                            order_best_first=order,
                            k=k,
                            pool_mul=args.pairwise_select_pool_mul,
                            temp=args.pairwise_select_temp,
                            seed=args.seed + rd * 10007 + 17,
                        )
                    else:
                        top_idx = order[:k]
                    picked_ids = remaining_ids[top_idx]
            else:
                cand_idx = build_mixed_candidates(
                    mu_raw, remaining_ids,
                    M=args.top_m_candidates,
                    feats=feats,
                    device=device,
                    model_svgp=model_svgp,
                    lik_svgp=lik_svgp,
                    x_mu=x_mu,
                    x_std=x_std,
                    frac_mu=args.frac_mu,
                    frac_sig=args.frac_sig,
                    frac_rnd=args.frac_rnd,
                    sigma_batch_size=args.sigma_batch_size,
                    use_likelihood_sigma=args.use_likelihood_sigma,
                    fast_pred_var=args.fast_pred_var,
                    seed=args.seed + rd
                )
                cand_ids = remaining_ids[cand_idx]

                use_like = args.use_likelihood_sigma
                if args.acq in ("poi", "eoi"):
                    use_like = False
                mu_svgp_std_cand, sig_svgp_std_cand = svgp_predict_mu_sigma_ids(
                    model_svgp, lik_svgp, feats, cand_ids,
                    x_mu=x_mu, x_std=x_std,
                    device=device,
                    batch_size=args.sigma_batch_size,
                    use_likelihood=use_like,
                    fast_pred_var=args.fast_pred_var
                )

                if args.acq in ("ucb", "lcb"):
                    mu_std_cand = mu_svgp_std_cand
                elif args.acq in ("poi", "eoi"):
                    if args.m2_model == "rf":
                        obs_pred = last_rf_model.predict(
                            np.vstack(df_labeled["combined_embedding"].values).astype(np.float32, copy=False)
                        ).astype(np.float32, copy=False)
                    else:
                        obs_pred = predict_on_embeddings_list(m2_model, df_labeled["combined_embedding"].values)
                    obs_y = df_labeled["Score"].to_numpy(np.float32, copy=False)
                    global_y_min = float(np.min(pool_scores))
                    global_y_max = float(np.max(pool_scores))
                    mu_cand_score = calibrate_mu_for_ei_pi(
                        mu_raw[cand_idx],
                        obs_pred=obs_pred,
                        obs_y=obs_y,
                        do_calibrate=args.calibrate_for_ei_pi,
                        do_minmax=args.minmax_for_ei_pi,
                        minmax_range=args.minmax_range,
                        global_y_min=global_y_min,
                        global_y_max=global_y_max
                    )
                    mu_std_cand = (mu_cand_score - y_mu) / (y_std + 1e-12)
                else:
                    mu_std_cand = (mu_raw[cand_idx] - y_mu) / (y_std + 1e-12)

                scores_cand = acq_scores(
                    mu_std_cand, sig_svgp_std_cand,
                    kind=args.acq, f_best_std=f_best_std,
                    kappa=args.kappa, xi=args.xi, minimize=True,
                    sigma_floor=args.sigma_floor, z_clip=args.z_clip
                )
                k = min(select_per_round, len(cand_ids))
                if k <= 0:
                    picked_ids = np.zeros((0,), dtype=np.int64)
                else:
                    order = np.argsort(-scores_cand)  # higher is better
                    if args.m2_model == "pairwise":
                        top_idx = select_topk_with_temperature(
                            order_best_first=order,
                            k=k,
                            pool_mul=args.pairwise_select_pool_mul,
                            temp=args.pairwise_select_temp,
                            seed=args.seed + rd * 10007 + 31,
                        )
                    else:
                        top_idx = order[:k]
                    picked_ids = cand_ids[top_idx]
        else:
            picked_ids = None

        picked_ids = bcast_object(picked_ids, src=0)
        picked_ids = np.asarray(picked_ids, dtype=np.int64)
        if picked_ids.size == 0:
            if rank == 0:
                print("[BO] No picked ids. Stop.")
            break

        picked_feats = feats.get_batch(picked_ids)
        picked_scores = pool_scores[picked_ids] if rank == 0 else None
        picked_scores = bcast_object(picked_scores, src=0)
        if rank == 0:
            picked_smiles = pool_smiles[picked_ids]
        else:
            picked_smiles = np.full(picked_ids.shape[0], "", dtype=object)
        df_selected = pd.DataFrame({
            "selection_round": np.full(picked_ids.shape[0], rd, dtype=np.int64),
            "sample_id": picked_ids,
            "SMILES": picked_smiles,
            "Score": np.asarray(picked_scores, dtype=np.float32),
            "combined_embedding": [row.copy() for row in picked_feats],
        })
        if pool_labels is not None:
            picked_labels = pool_labels[picked_ids] if rank == 0 else None
            picked_labels = bcast_object(picked_labels, src=0)
            df_selected["label"] = picked_labels
        df_labeled = pd.concat([df_labeled, df_selected], ignore_index=True)
        selected_set.update(picked_ids.tolist())

        if rank == 0:
            if args.round_selected_dir:
                round_path = os.path.join(args.round_selected_dir, f"round_{rd:02d}_selected.csv")
                save_round_selection_csv(df_selected, round_path)
                print(f"[BO] Saved round {rd} selections to: {round_path}")

            if args.round_cumulative_dir:
                cumulative_path = os.path.join(args.round_cumulative_dir, f"round_{rd:02d}_cumulative.csv")
                save_round_selection_csv(df_labeled, cumulative_path)
                print(f"[BO] Saved round {rd} cumulative selections to: {cumulative_path}")

            cumulative_selected = len(df_labeled)
            cumulative_active_found = int(df_labeled["label"].astype(float).eq(1).sum()) if "label" in df_labeled.columns else 0
            cumulative_rows.append({
                "round": int(rd),
                "cumulative_selected": int(cumulative_selected),
                "cumulative_active_found": int(cumulative_active_found),
                "cumulative_precision": (cumulative_active_found / cumulative_selected) if cumulative_selected else 0.0,
                "cumulative_recall": (cumulative_active_found / n_active_total) if n_active_total else 0.0,
                "cumulative_ef": compute_enrichment_factor(
                    cumulative_active_found,
                    cumulative_selected,
                    n_active_total,
                    n_total,
                ),
            })

        picked_set = set(map(int, picked_ids.tolist()))
        mask = np.array([int(i) not in picked_set for i in remaining_ids], dtype=bool)
        remaining_ids = remaining_ids[mask]

        if args.acq in ("poi", "eoi"):
            y_now = df_labeled["Score"].to_numpy(np.float32, copy=False)
            f_best_std = (float(np.min(y_now)) - y_mu) / (y_std + 1e-12)

        if args.acq != "greedy" and args.svgp_finetune and (rd % max(1, args.svgp_finetune_every) == 0):
            X_obs = np.vstack(df_labeled["combined_embedding"].values).astype(np.float32, copy=False)
            y_obs = df_labeled["Score"].to_numpy(np.float32, copy=False)
            if rank == 0:
                print("[BO] SVGP finetune (re-fit)...")
            model_svgp, lik_svgp, x_mu, x_std, y_mu, y_std = fit_svgp_ddp(
                X_obs, y_obs,
                device=device, rank=rank, world=world,
                M=args.svgp_M, batch_size=args.svgp_batch, iters=args.svgp_iters, lr=args.svgp_lr,
                kernel="matern", nu=1.5, seed=args.seed, use_amp=bool(args.svgp_amp), print_every=0,
                num_workers=args.svgp_num_workers,
                pin_memory=bool(args.pin_memory),
                persistent_workers=bool(args.persistent_workers),
                prefetch_factor=args.prefetch_factor
            )

        if rank == 0:
            print(f"[BO] picked={len(picked_ids):,} | labeled={len(df_labeled):,} | remaining={len(remaining_ids):,}")

        # ---- evaluation: retrieval + diversity (all ranks participate for gather) ----
        if args.eval_retrieval:
            all_ids_full = np.arange(n_total, dtype=np.int64)
            if args.m2_model == "rf":
                if rank == 0:
                    mu_all = np.empty((len(all_ids_full),), dtype=np.float32)
                    for s in range(0, len(all_ids_full), args.m2_pred_bs):
                        e = min(s + args.m2_pred_bs, len(all_ids_full))
                        ids = all_ids_full[s:e]
                        xb = feats.get_batch(ids).astype(np.float32, copy=False)
                        pred = last_rf_model.predict(xb).astype(np.float32, copy=False)
                        mu_all[s:e] = pred
                else:
                    mu_all = None
            else:
                local_ids_eval, local_mu_eval = predict_m2_raw_sharded(
                    m2_model, feats, all_ids_full, rank, world,
                    batch_size=args.m2_pred_bs, device=device
                )
                gids_eval, gmu_eval = gather_pairs_ids_vals(local_ids_eval, local_mu_eval, world)
                if rank == 0:
                    mu_map_eval = {int(i): float(v) for i, v in zip(gids_eval.tolist(), gmu_eval.tolist())}
                    mu_all = np.array([mu_map_eval[int(i)] for i in all_ids_full], dtype=np.float32)
                else:
                    mu_all = None

            if rank == 0:
                scores_all = pool_scores
                smiles_all = pool_smiles

                hits = eval_topk_overlap_hits(scores_all, mu_all, eval_ks, minimize=True)
                divs = eval_topk_scaffold_diversity(smiles_all, mu_all, eval_ks, minimize=True)
                div_map = {k: (nv, nu, r) for k, nv, nu, r in divs}
                parts = []
                for k, overlap, _ in hits:
                    nv, nu, r = div_map.get(k, (0, 0, 0.0))
                    parts.append(f"top{k}-[ {overlap} ] scaf_ratio={r:.3f}")
                round_summaries.append((rd, "  ".join(parts)))
                metrics_payload["round_metrics"].append({
                    "round": int(rd),
                    "hits": [{"k": int(k), "overlap": int(overlap), "ratio": float(ratio)} for k, overlap, ratio in hits],
                    "diversity": [{"k": int(k), "n_valid": int(nv), "n_unique": int(nu), "ratio": float(r)} for k, nv, nu, r in divs],
                })

        dist_barrier()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if rank == 0 and round_summaries:
        print("\n[BO] Round summary:")
        for rd, line in round_summaries:
            print(f"round {rd} :  {line}")

    if args.skip_final_ranking:
        if rank == 0:
            if args.metrics_out:
                os.makedirs(os.path.dirname(args.metrics_out) or ".", exist_ok=True)
                with open(args.metrics_out, "w", encoding="utf-8") as f:
                    json.dump(metrics_payload, f, indent=2)
                print(f"[BO] Saved metrics to: {args.metrics_out}")
            if args.cumulative_metrics_out:
                os.makedirs(os.path.dirname(args.cumulative_metrics_out) or ".", exist_ok=True)
                pd.DataFrame(cumulative_rows).to_csv(args.cumulative_metrics_out, index=False)
                print(f"[BO] Saved cumulative metrics to: {args.cumulative_metrics_out}")
            print("[BO] Skipped final full-pool ranking.")
        dist_barrier()
        ddp_cleanup()
        return

    # ---- final model on full labeled ----
    if args.m2_model == "rf":
        if world != 1:
            raise RuntimeError("RandomForest model only supports single-process (world=1).")
        if rank == 0:
            try:
                from sklearn.ensemble import RandomForestRegressor
            except Exception as e:
                raise RuntimeError("scikit-learn is required for m2_model=rf") from e
            X = np.vstack(df_labeled["combined_embedding"].values).astype(np.float32, copy=False)
            y = df_labeled["Score"].to_numpy(np.float32, copy=False)
            rf = RandomForestRegressor(
                n_estimators=args.rf_n_estimators,
                max_depth=args.rf_max_depth,
                min_samples_split=args.rf_min_samples_split,
                min_samples_leaf=args.rf_min_samples_leaf,
                n_jobs=args.rf_n_jobs,
                random_state=args.seed,
            )
            rf.fit(X, y)
            last_rf_model = rf
    else:
        m2_model, last_m2_state = train_torch_model(df_labeled)

    # ---- final inference ----
    all_ids_full = np.arange(n_total, dtype=np.int64)
    if args.m2_model == "rf":
        if rank == 0:
            mu_all = np.empty((len(all_ids_full),), dtype=np.float32)
            for s in range(0, len(all_ids_full), args.m2_pred_bs):
                e = min(s + args.m2_pred_bs, len(all_ids_full))
                ids = all_ids_full[s:e]
                xb = feats.get_batch(ids).astype(np.float32, copy=False)
                pred = last_rf_model.predict(xb).astype(np.float32, copy=False)
                mu_all[s:e] = pred
            top_n = max(1, int(len(mu_all) * args.final_top_frac))
            order = np.argsort(mu_all)
            top_ids = all_ids_full[order[:top_n]]
            df_out = pd.DataFrame({
                "predicted_rank": np.arange(1, top_n + 1, dtype=np.int64),
                "sample_id": top_ids,
                "SMILES": pool_smiles[top_ids],
                "Score": pool_scores[top_ids],
                "pred_score": mu_all[order[:top_n]],
            })
            if pool_labels is not None:
                df_out["label"] = pool_labels[top_ids]
            os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
            df_out.to_csv(args.out_csv, index=False)
            print(f"\n[BO] Saved top {args.final_top_frac*100:.2f}% ({top_n:,}) to: {args.out_csv}")
            if args.final_prediction_csv:
                os.makedirs(os.path.dirname(args.final_prediction_csv) or ".", exist_ok=True)
                df_out.to_csv(args.final_prediction_csv, index=False)
                print(f"[BO] Saved final prediction ranking to: {args.final_prediction_csv}")
    else:
        m2_final = RankModel(input_dim=input_dim, num_layers=args.m2_layers, dropout=args.m2_dropout).to(device)
        m2_final.load_state_dict(last_m2_state, strict=True)
        m2_final.eval()

        local_ids, local_mu = predict_m2_raw_sharded(
            m2_final, feats, all_ids_full, rank, world,
            batch_size=args.m2_pred_bs, device=device
        )
        gids, gmu = gather_pairs_ids_vals(local_ids, local_mu, world)

        if rank == 0:
            mu_map = {int(i): float(v) for i, v in zip(gids.tolist(), gmu.tolist())}
            mu_all = np.array([mu_map[int(i)] for i in all_ids_full], dtype=np.float32)
            top_n = max(1, int(len(mu_all) * args.final_top_frac))
            order = np.argsort(mu_all)
            top_ids = all_ids_full[order[:top_n]]
            df_out = pd.DataFrame({
                "predicted_rank": np.arange(1, top_n + 1, dtype=np.int64),
                "sample_id": top_ids,
                "SMILES": pool_smiles[top_ids],
                "Score": pool_scores[top_ids],
                "pred_score": mu_all[order[:top_n]],
            })
            if pool_labels is not None:
                df_out["label"] = pool_labels[top_ids]
            os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
            df_out.to_csv(args.out_csv, index=False)
            print(f"\n[BO] Saved top {args.final_top_frac*100:.2f}% ({top_n:,}) to: {args.out_csv}")
            if args.final_prediction_csv:
                os.makedirs(os.path.dirname(args.final_prediction_csv) or ".", exist_ok=True)
                df_out.to_csv(args.final_prediction_csv, index=False)
                print(f"[BO] Saved final prediction ranking to: {args.final_prediction_csv}")

    if rank == 0 and args.eval_retrieval:
        scores_all = pool_scores
        smiles_all = pool_smiles
        final_hits = eval_topk_overlap_hits(scores_all, mu_all, eval_ks, minimize=True)
        final_divs = eval_topk_scaffold_diversity(smiles_all, mu_all, eval_ks, minimize=True)
        metrics_payload["final_metrics"] = {
            "hits": [{"k": int(k), "overlap": int(overlap), "ratio": float(ratio)} for k, overlap, ratio in final_hits],
            "diversity": [{"k": int(k), "n_valid": int(nv), "n_unique": int(nu), "ratio": float(r)} for k, nv, nu, r in final_divs],
        }
        if args.metrics_out:
            os.makedirs(os.path.dirname(args.metrics_out) or ".", exist_ok=True)
            with open(args.metrics_out, "w", encoding="utf-8") as f:
                json.dump(metrics_payload, f, indent=2)
            print(f"[BO] Saved metrics to: {args.metrics_out}")

    if rank == 0 and args.cumulative_metrics_out:
        os.makedirs(os.path.dirname(args.cumulative_metrics_out) or ".", exist_ok=True)
        pd.DataFrame(cumulative_rows).to_csv(args.cumulative_metrics_out, index=False)
        print(f"[BO] Saved cumulative metrics to: {args.cumulative_metrics_out}")

    dist_barrier()
    ddp_cleanup()

# =========================
# 10) args
# =========================
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--smiles_csv", required=True)
    p.add_argument("--emb_dir", required=True)
    p.add_argument("--init_csv", required=True)
    p.add_argument("--emb_pattern", default="part_*.npy")

    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--select_frac", type=float, default=0.002)

    p.add_argument("--acq", type=str, default="greedy", choices=["greedy", "lcb", "ucb", "poi", "eoi"])
    p.add_argument("--kappa", type=float, default=2.0)
    p.add_argument("--xi", type=float, default=0.0)
    p.add_argument("--sigma_floor", type=float, default=0.05)
    p.add_argument("--z_clip", type=float, default=8.0)

    # candidate mixing
    p.add_argument("--top_m_candidates", type=int, default=200000)
    p.add_argument("--frac_mu", type=float, default=0.6)
    p.add_argument("--frac_sig", type=float, default=0.3)
    p.add_argument("--frac_rnd", type=float, default=0.1)
    p.add_argument("--sigma_batch_size", type=int, default=65536)
    p.add_argument("--use_likelihood_sigma", type=int, default=1)
    p.add_argument("--fast_pred_var", type=int, default=0)

    # EI/PI calibration
    p.add_argument("--calibrate_for_ei_pi", type=int, default=1)
    p.add_argument("--minmax_for_ei_pi", type=int, default=1)
    p.add_argument("--minmax_range", type=str, default="observed", choices=["observed", "global"])

    # SVGP
    p.add_argument("--svgp_M", type=int, default=1024)
    p.add_argument("--svgp_iters", type=int, default=1000)
    p.add_argument("--svgp_lr", type=float, default=0.01)
    p.add_argument("--svgp_batch", type=int, default=4096)
    p.add_argument("--svgp_pred_batch", type=int, default=8192)
    p.add_argument("--svgp_print_every", type=int, default=100)
    p.add_argument("--svgp_amp", type=int, default=0)
    p.add_argument("--svgp_finetune", type=int, default=0)
    p.add_argument("--svgp_finetune_every", type=int, default=1)

    # M2
    p.add_argument("--m2_model", type=str, default="triplet", choices=["triplet", "pairwise", "mlp", "rf"])
    p.add_argument("--m2_layers", type=int, default=2)
    p.add_argument("--m2_dropout", type=float, default=0.3)
    p.add_argument("--m2_lr", type=float, default=1e-3)
    p.add_argument("--m2_wd", type=float, default=1e-4)
    p.add_argument("--m2_train_bs", type=int, default=4096)
    p.add_argument("--m2_pred_bs", type=int, default=4096)
    p.add_argument("--m2_epochs", type=int, default=200)
    p.add_argument("--m2_patience", type=int, default=50)
    p.add_argument("--m2_split", type=float, default=0.9)
    p.add_argument("--semi_hard_pos_window", type=int, default=0)
    p.add_argument("--semi_hard_neg_window", type=int, default=0)
    p.add_argument("--semi_hard_pos_frac", type=float, default=0.0)
    p.add_argument("--semi_hard_neg_frac", type=float, default=0.0)

    # loss
    p.add_argument("--margin", type=float, default=0.3)
    p.add_argument("--lambda_rank", type=float, default=0.01)

    # RandomForest (m2_model=rf)
    p.add_argument("--rf_n_estimators", type=int, default=200)
    p.add_argument("--rf_max_depth", type=int, default=None)
    p.add_argument("--rf_min_samples_split", type=int, default=2)
    p.add_argument("--rf_min_samples_leaf", type=int, default=1)
    p.add_argument("--rf_n_jobs", type=int, default=-1)

    p.add_argument("--final_top_frac", type=float, default=0.01)
    p.add_argument("--out_csv", type=str, default="./outputs/138M_top1_smiles_bo.csv")
    p.add_argument("--metrics_out", type=str, default=None)
    p.add_argument("--round_selected_dir", type=str, default=None)
    p.add_argument("--round_cumulative_dir", type=str, default=None)
    p.add_argument("--round_prediction_dir", type=str, default=None)
    p.add_argument("--final_prediction_csv", type=str, default=None)
    p.add_argument("--save_initial_selection", type=int, default=1)
    p.add_argument("--cumulative_metrics_out", type=str, default=None)
    p.add_argument("--skip_final_ranking", action="store_true")

    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--svgp_num_workers", type=int, default=2)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--prefetch_factor", type=int, default=3)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--ddp_timeout_min", type=int, default=30)
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--eval_retrieval", type=int, default=1)
    p.add_argument(
        "--pairwise_select_pool_mul",
        type=float,
        default=1.8,
        help="For m2_model=pairwise: sample k picks from top (k*pool_mul) candidates.",
    )
    p.add_argument(
        "--pairwise_select_temp",
        type=float,
        default=0.35,
        help="For m2_model=pairwise: temperature for top-pool sampling (0 => deterministic top-k).",
    )

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_bo(args)
