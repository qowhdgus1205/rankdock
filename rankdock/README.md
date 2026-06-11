# RankDock

RankDock is a rank-learning active-learning pipeline for large-scale molecular docking screens. It supports embedding generation, LSH-based initialization, Bayesian optimization with RankModel baselines, and retrieval/diversity analysis.

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

The score CSVs are not committed to this repository. Place local CSV copies under `data/` as described in [data/README.md](data/README.md). Generated embeddings, pretrained checkpoints, and docking outputs should stay under `models/`, `output/`, or `outputs/` locally.

## Data Provenance

- `138M_scores.csv`: docking scores derived from the ultra-large library docking screen reported by Lyu et al., "Ultra-large library docking for discovering new chemotypes," *Nature* 566, 224-229 (2019), doi: [10.1038/s41586-019-0917-9](https://doi.org/10.1038/s41586-019-0917-9).
- `EnamineHTS_scores.csv`: Enamine HTS / Enamine2M docking benchmark table used through the MolPAL-style active-learning benchmark workflow; see Graff, Shakhnovich, and Coley, "Accelerating high-throughput virtual screening through molecular pool-based active learning," arXiv: [2012.07127](https://arxiv.org/abs/2012.07127).

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
