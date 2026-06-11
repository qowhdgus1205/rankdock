import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.cuda.amp import autocast, GradScaler
import sys
from tqdm import tqdm
import numpy as np

class RankModel(nn.Module):
    def __init__(self, input_dim, num_layers=3, dropout=0.4):
        super(RankModel, self).__init__()

        self.layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()

        current_dim = input_dim

        for i in range(num_layers):
            if i == 0 :
                next_dim = input_dim*2

            else :
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
        # self.regressor = nn.Sequential(
        #     nn.Linear(current_dim, current_dim // 2),
        #     nn.ReLU(),
        #     nn.Linear(current_dim // 2, 1)
        # )

    def encode(self, x):
        residual = x
        for i in range(len(self.skip_layers)):
            idx = i * 3
            linear = self.layers[idx]
            relu = self.layers[idx + 1]
            dropout = self.layers[idx + 2]

            x = linear(x)
            x = relu(x)
            x = dropout(x)

            x = x + self.skip_layers[i](residual)
            residual = x
        return x

    def forward(self, x):
        x = self.encode(x)
        #x = F.normalize(x, dim=-1)
        return self.regressor(x)


def rankdnn_loss(q, p, n, margin=0.1, lambda_rank=0.1):
    # Explicitly enforce the desired ordering: p < q < n.
    loss_pq = torch.relu(margin + (p - q))
    loss_qn = torch.relu(margin + (q - n))
    order_loss = (loss_pq + loss_qn).mean()

    # Enforce an explicit global separation: (n - p) should be >= 2 * margin.
    desired_gap = 2.0 * margin
    gap_loss = F.softplus(desired_gap - (n - p)).mean()
    return order_loss + lambda_rank * gap_loss


# def train_rankdnn(model, train_dataloader, val_dataloader, optimizer, device,
#                   margin=0.2, lambda_rank=0.001, epochs=300, save_logs=False, early_stop=True, patience=10):

#     import sys

#     best_val_loss = float("inf")
#     best_model_state = None
#     early_counter = 0

#     for epoch in range(epochs):
#         model.train()
#         total_train_loss = 0

#         for query, positive, negative in train_dataloader:
#             query, positive, negative = query.to(device), positive.to(device), negative.to(device)
#             optimizer.zero_grad()

#             q_score = model(query)
#             p_score = model(positive)
#             n_score = model(negative)

#             loss = rankdnn_loss(q_score, p_score, n_score, margin=margin, lambda_rank=lambda_rank)
#             loss.backward()
#             optimizer.step()
#             total_train_loss += loss.item()

#         model.eval()
#         total_val_loss = 0
#         with torch.no_grad():
#             for query, positive, negative in val_dataloader:
#                 query, positive, negative = query.to(device), positive.to(device), negative.to(device)

#                 q_score = model(query)
#                 p_score = model(positive)
#                 n_score = model(negative)

#                 val_loss = rankdnn_loss(q_score, p_score, n_score, margin=margin, lambda_rank=lambda_rank)
#                 total_val_loss += val_loss.item()

#         avg_train_loss = total_train_loss / len(train_dataloader)
#         avg_val_loss = total_val_loss / len(val_dataloader)

#         # ✅ 한 줄로 출력
#         sys.stdout.write(f"\rEpoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
#         sys.stdout.flush()

#         if early_stop:
#             if avg_val_loss < best_val_loss:
#                 best_val_loss = avg_val_loss
#                 best_model_state = model.state_dict()
#                 early_counter = 0
#             else:
#                 early_counter += 1
#                 if early_counter >= patience:
#                     print(f"\nEarly stopping at epoch {epoch+1}")
#                     break

#     if early_stop and best_model_state:
#         model.load_state_dict(best_model_state)
#         print(f"\nRestored best model with val loss: {best_val_loss:.4f}")

def train_rankdnn(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    device,
    margin=0.2,
    lambda_rank=0.001,
    epochs=300,
    save_logs=False,
    early_stop=True,
    patience=10,
    use_amp=True,
    ddp_rank=0,          # DDP rank (single GPU면 0)
    ddp_sampler=None,    # DistributedSampler (없으면 None)
):
    best_val_loss = float("inf")
    best_model_state = None
    early_counter = 0

    scaler = GradScaler(enabled=use_amp)

    for epoch in range(epochs):

        # ✅ DDP: epoch마다 sampler 동기화
        if ddp_sampler is not None and hasattr(ddp_sampler, "set_epoch"):
            ddp_sampler.set_epoch(epoch)

        # =====================
        # Train
        # =====================
        model.train()
        total_train_loss = 0.0

        for batch in train_dataloader:
            if batch is None:
                continue
            query, positive, negative = batch
            query = query.to(device, non_blocking=True)
            positive = positive.to(device, non_blocking=True)
            negative = negative.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp and device.type == "cuda"):
                q_score = model(query)
                p_score = model(positive)
                n_score = model(negative)

                loss = rankdnn_loss(
                    q_score, p_score, n_score,
                    margin=margin,
                    lambda_rank=lambda_rank
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += float(loss.detach().cpu())

        # =====================
        # Validation
        # =====================
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in val_dataloader:
                if batch is None:
                    continue
                query, positive, negative = batch
                query = query.to(device, non_blocking=True)
                positive = positive.to(device, non_blocking=True)
                negative = negative.to(device, non_blocking=True)

                with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp and device.type == "cuda"):
                    q_score = model(query)
                    p_score = model(positive)
                    n_score = model(negative)

                    val_loss = rankdnn_loss(
                        q_score, p_score, n_score,
                        margin=margin,
                        lambda_rank=lambda_rank
                    )

                total_val_loss += float(val_loss.detach().cpu())

        avg_train_loss = total_train_loss / max(1, len(train_dataloader))
        avg_val_loss = total_val_loss / max(1, len(val_dataloader))

        # ✅ rank0만 출력
        if ddp_rank == 0:
            sys.stdout.write(
                f"\rEpoch {epoch+1}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f}"
            )
            sys.stdout.flush()

        # =====================
        # Early stopping (rank0 기준)
        # =====================
        if early_stop and ddp_rank == 0:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                best_model_state = {k: v.detach().cpu() for k, v in state.items()}
                early_counter = 0
            else:
                early_counter += 1
                if early_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

    # =====================
    # Best model restore
    # =====================
    if early_stop and best_model_state is not None:
        target = model.module if hasattr(model, "module") else model
        target.load_state_dict(best_model_state, strict=True)
        if ddp_rank == 0:
            print(f"\nRestored best model with val loss: {best_val_loss:.4f}")

def predict_with_m2_model(df, model, feature_col="combined_embedding", batch_size=256, device="cuda"):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    # ✅ 데이터 복사
    df_select = df.copy()

    # ✅ 벡터 데이터를 PyTorch Tensor로 변환
    latent_tensor = torch.tensor(np.vstack(df_select[feature_col].values)).float()

    # ✅ DataLoader 생성
    latent_loader = DataLoader(latent_tensor, batch_size=batch_size, shuffle=False)

    # ✅ 모델을 평가 모드로 설정
    model.eval()
    class_probs = []

    with torch.no_grad():
        for batch_data in tqdm(latent_loader, desc="Predicting with RankModel"):
            batch_data = batch_data.to(device)  # GPU/CPU 전송
            probs = model(batch_data)
            class_probs.append(probs.cpu())  # 결과를 CPU로 이동 후 저장

    # ✅ 리스트를 텐서로 병합
    class_probs = torch.cat(class_probs)
    #class_probs = min_max_scale(class_probs)

    # ✅ 데이터프레임에 결과 추가
    df_select["1_probs"] = class_probs.numpy()

    return df_select

############################################################################
