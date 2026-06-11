# RankDock

RankDock is a rank-learning active-learning pipeline for large-scale molecular docking screens. The code in this repository supports the workflow used in `sn-article_main.pdf`: embedding generation, LSH-based initialization, Bayesian optimization with RankModel baselines, and retrieval/diversity analysis.

## Repository Layout

```text
active_learning.py        # Main BO loop and RankModel training objectives
sampling.py               # LSH/random initial sampling
retrieval.py              # Round-wise retrieval summaries
validation.py             # CSV/embedding alignment checks
data/
  embeddings.py           # Graph/SMILES embedding generation and concatenation
  dataset.py              # Molecular graph dataset helpers
  README.md               # Where to place external score CSVs
models/                   # RankModel and MolCLR GCN definitions
scripts/                  # Thin command-line wrappers
docs/                     # Notes for reproducing manuscript analyses
examples/                 # Small example configs or command templates
results/                  # Cumulative selected-compound CSVs and summaries
```

The manuscript score CSVs are not committed to this repository. We use docking scores derived from the ultra-large docking data described by Lyu et al., *Nature* 566, 224-229 (2019), "Ultra-large library docking for discovering new chemotypes". Place local CSV copies under `data/` as described in [data/README.md](data/README.md). Generated embeddings, pretrained checkpoints, and docking outputs should stay under `models/`, `output/`, or `outputs/` locally.

Precomputed cumulative selection CSVs and summary metrics are included under [results/](results/), so reviewers can inspect round-wise outputs without downloading the full score tables.

## Installation

```bash
python -m pip install -e .
```

Conda users can create an environment with:

```bash
conda env create -f environment.yaml
conda activate rankdock
python -m pip install -e .
```

Graph embedding generation additionally needs the PyTorch Geometric stack matching your local PyTorch/CUDA version.

## Core Pipeline

Generate MolCLR graph embeddings:

```bash
python scripts/generate_embeddings.py \
  --mode graph \
  --input data/merged_smiles.csv \
  --output output/graph_embeddings \
  --model_path models/MolCLR.pth
```

Generate ChemBERTa SMILES embeddings:

```bash
python scripts/generate_embeddings.py \
  --mode smiles \
  --input data/merged_smiles.csv \
  --output output/smiles_embeddings \
  --model_dir models
```

Concatenate embeddings:

```bash
python scripts/concat_embeddings.py \
  --graph_dir output/graph_embeddings \
  --smiles_dir output/smiles_embeddings \
  --output_dir output/combined_embeddings
```

Select the initial LSH sample:

```bash
python scripts/initial_sampling.py \
  --csv_path data/merged_smiles.csv \
  --embedding_dir output/combined_embeddings \
  --sample_ratio 0.001 \
  --output_csv outputs/initial_selected_samples.csv \
  --seed 2025
```

Run RankDock active learning:

```bash
python scripts/run_rankdock.py \
  --smiles_csv data/merged_smiles.csv \
  --emb_dir output/combined_embeddings \
  --init_csv outputs/initial_selected_samples.csv \
  --m2_model pairwise \
  --acq greedy \
  --out_csv outputs/top1_pairwise.csv
```

The `--m2_model` options share the same `RankModel` architecture except `rf`:

- `mlp`: RankModel trained with MSE regression on docking scores
- `triplet`: RankModel trained with triplet ranking loss
- `pairwise`: RankModel trained with pairwise logistic ranking loss
- `rf`: Random forest regression baseline

## Validation

Before running large experiments, check that CSV rows and embedding parts are aligned:

```bash
python scripts/validate_embeddings.py \
  --csv data/merged_smiles.csv \
  --smiles_dir output/smiles_embeddings \
  --graph_dir output/graph_embeddings \
  --combined_dir output/combined_embeddings
```

Syntax check for the core public modules:

```bash
python -m py_compile \
  active_learning.py \
  data/embeddings.py \
  sampling.py \
  retrieval.py \
  score_final.py \
  validation.py
```
