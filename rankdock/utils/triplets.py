import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, random_split


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TripletDataset(Dataset):
    """
    Fast TripletDataset (drop-in replacement)
    - 기존 의미 유지:
        positive: docking_score < query_score
        negative: docking_score > query_score
    - 병목이던 np.where를 제거하고, 정렬+rank 기반 O(1) 샘플링으로 변경
    """

    def __init__(self, embeddings, docking_scores):
        """
        embeddings: list/np.ndarray (N, D)
        docking_scores: np.ndarray/list (N,)
        """
        # embeddings가 list여도 np array로 정리 (indexing 빠르게)
        if isinstance(embeddings, list):
            embeddings = np.asarray(embeddings, dtype=np.float32)
        self.embeddings = embeddings

        self.scores = np.asarray(docking_scores, dtype=np.float32)
        self.N = len(self.scores)

        # ---- 핵심: score 정렬 (한 번만) ----
        self.sorted_idx = np.argsort(self.scores)          # ascending (lower is better)
        # 각 sample의 rank 위치 (inverse index)
        self.rank = np.empty(self.N, dtype=np.int64)
        self.rank[self.sorted_idx] = np.arange(self.N)

    def __len__(self):
        return self.N

    def _get_positive(self, query_idx):
        # positive: score < query_score  <=> rank < r
        r = self.rank[query_idx]
        if r == 0:
            return None
        # [0, r-1]에서 랜덤으로 하나
        return int(self.sorted_idx[random.randint(0, r - 1)])

    def _get_negative(self, query_idx):
        # negative: score > query_score <=> rank > r
        r = self.rank[query_idx]
        if r == self.N - 1:
            return None
        # [r+1, N-1]에서 랜덤으로 하나
        return int(self.sorted_idx[random.randint(r + 1, self.N - 1)])

    def __getitem__(self, idx):
        q_idx = idx
        p_idx = self._get_positive(q_idx)
        n_idx = self._get_negative(q_idx)

        # 기존 코드와 동일하게 "유효한 triplet 없으면 None"
        if p_idx is None or n_idx is None or q_idx == p_idx or q_idx == n_idx:
            return None

        q_embedding = torch.tensor(self.embeddings[q_idx], dtype=torch.float32)
        p_embedding = torch.tensor(self.embeddings[p_idx], dtype=torch.float32)
        n_embedding = torch.tensor(self.embeddings[n_idx], dtype=torch.float32)

        return q_embedding, p_embedding, n_embedding


def collate_fn(batch):
    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0:
        return None  # 빈 배치 방지

    query_batch = torch.stack([b[0] for b in batch])
    positive_batch = torch.stack([b[1] for b in batch])
    negative_batch = torch.stack([b[2] for b in batch])

    return query_batch, positive_batch, negative_batch


def prepare_dataloaders(df, batch_size=256, split_ratio=0.8, collate_fn=None, seed=2025):
    """
    기존 시그니처 최대한 유지.
    df['combined_embedding'], df['Score'] 사용.
    """
    dataset = TripletDataset(df["combined_embedding"].tolist(), df["Score"].values)

    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size

    g = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=g)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"Train Dataset Size: {train_size}, Validation Dataset Size: {val_size}")
    return train_dataloader, val_dataloader


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)