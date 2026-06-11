import argparse
import os
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def _is_valid_smiles(smi):
    from rdkit import Chem
    try:
        return Chem.MolFromSmiles(smi) is not None
    except Exception:
        return False


def _build_valid_mask(smiles_list, workers=1, desc="Validating SMILES"):
    from tqdm import tqdm
    if workers and workers > 1:
        from multiprocessing import Pool
        chunksize = max(1000, len(smiles_list) // (workers * 20))
        with Pool(processes=workers) as pool:
            return list(tqdm(
                pool.imap(_is_valid_smiles, smiles_list, chunksize=chunksize),
                total=len(smiles_list),
                desc=desc,
            ))
    return [_is_valid_smiles(smi) for smi in tqdm(smiles_list, desc=desc)]


# =========================
# Graph Embeddings (MolCLR-GCN)
# =========================
def graph_embeddings(csv_path, model_path, output_path,
                     smiles_column="SMILES", batch_size=512, device=None,
                     save_every=10_000_000, num_workers=4,
                     validation_workers=1):

    import numpy as np
    from tqdm import tqdm
    import torch
    from torch_geometric.loader import DataLoader
    try:
        from rankdock.models.gcn_molclr import GCN
        from rankdock.data.dataset import MoleculeDataset, read_smiles
    except ModuleNotFoundError:
        from models.gcn_molclr import GCN
        from data.dataset import MoleculeDataset, read_smiles

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    smiles_list = read_smiles(csv_path, smiles_column=smiles_column)
    valid_mask = _build_valid_mask(
        smiles_list,
        workers=validation_workers,
        desc="Validating graph SMILES",
    )
    valid_smiles = [smi for smi, is_valid in zip(smiles_list, valid_mask) if is_valid]

    dataset = MoleculeDataset(valid_smiles, assume_valid=True)
    # DataLoader optimization: num_workers, pin_memory, persistent_workers, prefetch_factor
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=3 if num_workers > 0 else None
    )

    model = GCN(num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0, pool='mean').to(device)

    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]

    model.load_state_dict(state_dict)
    model.eval()
    # `torch.compile` can fail in containerized GPU setups that lack Triton build support.
    # Keep eager execution as the default and allow opt-in via env var.
    if os.environ.get("ENABLE_TORCH_COMPILE", "0") == "1":
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as exc:
            print(f"[WARN] torch.compile disabled due to error: {exc}")
    print("[OK] Pretrained GCN loaded")

    os.makedirs(output_path, exist_ok=True)

    # The MolCLR encoder returns the pooled feature vector `h`, whose dimension
    # matches `feat_dim` on the model, not the internal node embedding size.
    emb_dim = int(getattr(model, "feat_dim", 512))
    buffer_size = min(save_every, len(smiles_list))
    buffer = np.empty((buffer_size, emb_dim), dtype=np.float32)
    total_count, file_index, buffer_idx = 0, 0, 0

    if valid_smiles:
        valid_iter = iter(dataloader)
        emb_batch = None
        emb_batch_idx = 0

        def next_valid_embedding():
            nonlocal emb_batch, emb_batch_idx
            while emb_batch is None or emb_batch_idx >= emb_batch.shape[0]:
                try:
                    batch = next(valid_iter)
                except StopIteration:
                    raise RuntimeError("Ran out of valid embeddings early")
                if batch is None:
                    continue
                data_i, _ = batch
                data_i = data_i.to(device, non_blocking=True)
                with torch.inference_mode(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    h, _ = model(data_i)
                emb_batch = h.detach().cpu().numpy().astype(np.float32, copy=False)
                emb_batch_idx = 0
            emb = emb_batch[emb_batch_idx]
            emb_batch_idx += 1
            return emb
    else:
        valid_iter = None
        emb_batch = None
        emb_batch_idx = 0

    zero_vec = np.zeros((emb_dim,), dtype=np.float32)

    for is_valid in tqdm(valid_mask, desc="Extracting graph embeddings (aligned)"):
        emb = next_valid_embedding() if is_valid else zero_vec
        if buffer_idx < buffer_size:
            buffer[buffer_idx] = emb
            buffer_idx += 1
            total_count += 1

        if buffer_idx >= buffer_size:
            save_path = os.path.join(output_path, f"part_{file_index:03d}.npy")
            np.save(save_path, buffer[:buffer_idx])
            print(f"[SAVE] {save_path} ({buffer_idx})")
            buffer_idx = 0
            file_index += 1

    if buffer_idx > 0:
        save_path = os.path.join(output_path, f"part_{file_index:03d}.npy")
        np.save(save_path, buffer[:buffer_idx])
        print(f"[SAVE] {save_path} ({buffer_idx})")

    valid_mask_path = os.path.join(output_path, "valid_mask.npy")
    np.save(valid_mask_path, np.array(valid_mask, dtype=np.uint8))
    print(f"[SAVE] {valid_mask_path} ({int(sum(valid_mask))} valid)")

    print("[DONE] MolCLR Graph Embedding Extraction")


# =========================
# SMILES Embeddings (ChemBERTa – Offline)
# =========================
def smiles_embeddings(data_path,
                      smiles_column="SMILES",
                      output_path="chemberta_embeddings",
                      model_dir="./models",
                      batch_size=128,
                      save_every=10_000_000,
                      validation_workers=1):
    import os
    import json
    import pandas as pd
    import numpy as np
    import torch
    from tqdm import tqdm
    from rdkit import Chem
    from transformers import AutoTokenizer, AutoModel, AutoConfig

    # -------------------------
    # 0) Resolve absolute paths
    # -------------------------
    cwd = os.getcwd()
    model_dir = os.path.abspath(model_dir)

    tokenizer_dir = os.path.join(model_dir, "tokenizer")
    config_path   = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        alt_config_path = os.path.join(model_dir, "chemberta_config.json")
        if os.path.isfile(alt_config_path):
            config_path = alt_config_path
    weights_path  = os.path.join(model_dir, "chemberta_state_dict.pth")

    print(f"[INFO] CWD: {cwd}")
    print(f"[INFO] model_dir(abs): {model_dir}")
    print(f"[INFO] tokenizer_dir: {tokenizer_dir}")
    print(f"[INFO] config_path  : {config_path}")
    print(f"[INFO] weights_path : {weights_path}")

    # -------------------------
    # 1) Sanity checks (fail fast)
    # -------------------------
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"model_dir not found: {model_dir}")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.json not found: {config_path}")

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"chemberta_state_dict.pth not found: {weights_path}")

    tokenizer_available = os.path.isdir(tokenizer_dir) and (
        os.path.isfile(os.path.join(tokenizer_dir, "tokenizer.json")) or
        os.path.isfile(os.path.join(tokenizer_dir, "tokenizer_config.json"))
    )

    with open(config_path, encoding="utf-8") as f:
        config_payload = json.load(f)
    pretrained_name = config_payload.get("_name_or_path", "seyonec/ChemBERTa-zinc-base-v1")

    # -------------------------
    # 2) Load tokenizer/model OFFLINE ONLY
    # -------------------------
    if tokenizer_available:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            local_files_only=True,
            use_fast=True
        )
        print("[OK] loaded tokenizer from local model_dir")
    else:
        print(
            f"[WARN] tokenizer files missing in {tokenizer_dir}. "
            f"Falling back to Hugging Face tokenizer: {pretrained_name}"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_name,
            use_fast=True
        )

    config = AutoConfig.from_pretrained(config_path)
    # 모델 생성 + pth 주입
    model = AutoModel.from_config(config)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print("[OK] ChemBERTa loaded (.pth + config + tokenizer)")

    # -------------------------
    # 3) Read SMILES list
    # -------------------------
    df = pd.read_csv(data_path) if data_path.endswith(".csv") else pd.read_table(data_path)
    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found in input. Columns: {list(df.columns)[:20]} ...")

    smiles_list = df[smiles_column].astype(str).tolist()

    os.makedirs(output_path, exist_ok=True)

    valid_mask = _build_valid_mask(
        smiles_list,
        workers=validation_workers,
        desc="Validating ChemBERTa SMILES",
    )

    buffer_size = min(save_every, len(smiles_list))
    emb_dim = config.hidden_size
    buffer = np.empty((buffer_size, emb_dim), dtype=np.float32)
    total_count = 0
    file_index = 0
    buffer_idx = 0

    # -------------------------
    # 4) Embed + save in parts
    # -------------------------
    use_amp = device.type == "cuda"
    if os.environ.get("ENABLE_TORCH_COMPILE", "0") == "1":
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as exc:
            print(f"[WARN] torch.compile disabled due to error: {exc}")

    for i in tqdm(range(0, len(smiles_list), batch_size), desc="Extracting ChemBERTa embeddings (aligned)"):
        batch = smiles_list[i:i + batch_size]
        batch_mask = valid_mask[i:i + batch_size]

        valid_smiles = []
        valid_positions = []
        for pos, (smi, is_valid) in enumerate(zip(batch, batch_mask)):
            if is_valid:
                valid_smiles.append(smi)
                valid_positions.append(pos)

        batch_emb = np.zeros((len(batch), emb_dim), dtype=np.float32)

        if valid_smiles:
            tokens = tokenizer(valid_smiles, return_tensors="pt", padding=True, truncation=True)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(**tokens)
                hidden = outputs.last_hidden_state
                valid_emb = hidden.mean(dim=1).detach().cpu().numpy().astype(np.float32, copy=False)
            for pos, emb in zip(valid_positions, valid_emb):
                batch_emb[pos] = emb

        for row in batch_emb:
            if buffer_idx < buffer_size:
                buffer[buffer_idx] = row
                buffer_idx += 1
                total_count += 1
            if buffer_idx >= buffer_size:
                save_path = os.path.join(output_path, f"part_{file_index:03d}.npy")
                np.save(save_path, buffer[:buffer_idx])
                print(f"[SAVE] {save_path} ({buffer_idx})")
                buffer_idx = 0
                file_index += 1

    if buffer_idx > 0:
        save_path = os.path.join(output_path, f"part_{file_index:03d}.npy")
        np.save(save_path, buffer[:buffer_idx])
        print(f"[SAVE] {save_path} ({buffer_idx})")

    valid_mask_path = os.path.join(output_path, "valid_mask.npy")
    np.save(valid_mask_path, np.array(valid_mask, dtype=np.uint8))
    print(f"[SAVE] {valid_mask_path} ({int(sum(valid_mask))} valid)")

    print("[DONE] ChemBERTa Embedding Save Completed")


# =========================
# Embedding Operations
# =========================
def concat_embedding(graph_emb_dir, smiles_emb_dir, output_dir):
    import glob
    import numpy as np
    from tqdm import tqdm

    os.makedirs(output_dir, exist_ok=True)

    graph_parts = sorted(glob.glob(os.path.join(graph_emb_dir, "part_*.npy")))
    smiles_parts = sorted(glob.glob(os.path.join(smiles_emb_dir, "part_*.npy")))

    if len(graph_parts) != len(smiles_parts):
        raise ValueError("Number of graph and smiles parts must match")

    for idx, (g_path, s_path) in enumerate(
        tqdm(zip(graph_parts, smiles_parts), total=len(graph_parts), desc="Processing batches")
    ):
        g = np.load(g_path, mmap_mode="r")
        s = np.load(s_path, mmap_mode="r")

        if g.shape[0] != s.shape[0]:
            raise ValueError(f"Batch size mismatch at {g_path} and {s_path}")

        combined = np.concatenate([g, s], axis=1)
        save_path = os.path.join(output_dir, f"part_{idx:03d}.npy")
        np.save(save_path, combined.astype(np.float32))

    print(f"Saved concatenated embeddings to directory: {output_dir}")


# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Molecular Embedding Generator")

    parser.add_argument("--mode", choices=["graph", "smiles", "concat"], required=True)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--graph_dir", default=None, help="Directory containing graph part_*.npy files for concat mode")
    parser.add_argument("--smiles_dir", default=None, help="Directory containing SMILES part_*.npy files for concat mode")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--validation_workers", type=int, default=1)

    parser.add_argument("--model_path", default=None,
                        help="GCN .pth path (graph mode)")
    parser.add_argument("--model_dir", default="./models",
                        help="Directory containing chemberta_state_dict.pth, config.json, tokenizer/")
    parser.add_argument("--smiles_column", default="SMILES")

    args = parser.parse_args()

    if args.mode == "graph":
        if args.input is None:
            raise ValueError("--input is required for graph mode")
        if args.model_path is None:
            raise ValueError("--model_path is required for graph mode")

        graph_embeddings(
            csv_path=args.input,
            model_path=args.model_path,
            output_path=args.output,
            smiles_column=args.smiles_column,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            validation_workers=args.validation_workers
        )

    elif args.mode == "smiles":
        if args.input is None:
            raise ValueError("--input is required for smiles mode")
        smiles_embeddings(
            data_path=args.input,
            smiles_column=args.smiles_column,
            output_path=args.output,
            model_dir=args.model_dir,
            batch_size=args.batch_size,
            validation_workers=args.validation_workers
        )

    elif args.mode == "concat":
        if args.graph_dir is None or args.smiles_dir is None:
            raise ValueError("--graph_dir and --smiles_dir are required for concat mode")
        concat_embedding(
            graph_emb_dir=args.graph_dir,
            smiles_emb_dir=args.smiles_dir,
            output_dir=args.output,
        )
