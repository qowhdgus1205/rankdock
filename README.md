# Overview

RankDock is a rank-learning active-learning pipeline for large-scale molecular docking screens. It supports molecular embedding generation, LSH-based initial sampling, Bayesian-optimization-style candidate selection, RankModel baselines, random forest baselines, and retrieval/diversity analysis.

RankDock is organized around a practical screening loop:

1. Generate graph and SMILES embeddings for a docking-score table.
2. Concatenate embeddings into row-aligned `part_*.npy` files.
3. Select an initial diverse sample with LSH.
4. Run active learning with one of the supported model baselines.
5. Summarize round-wise retrieval and diversity.

The public neural baselines use the same `RankModel` backbone so that comparisons isolate the training objective:

- `mlp`: RankModel trained with MSE regression on docking scores
- `triplet`: RankModel trained with triplet ranking loss
- `pairwise`: RankModel trained with pairwise logistic ranking loss
- `rf`: random forest regression baseline

## Table of Contents

- [Repository Layout](#repository-layout)
- [Data Provenance](#data-provenance)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running](#running)
- [Validation](#validation)
- [Optional Settings](#optional-settings)
- [Outputs and Large Files](#outputs-and-large-files)
- [License](#license)

## Repository Layout

```text
rankdock/
  active_learning.py      # Main active-learning loop and RankModel objectives
  sampling.py             # LSH/random initial sampling
  retrieval.py            # Round-wise retrieval summaries
  score_final.py          # Final compound scoring from cumulative rounds
  validation.py           # CSV/embedding alignment checks
  data/
    embeddings.py         # Graph/SMILES embedding generation and concatenation
    dataset.py            # Molecular graph dataset helpers
    README.md             # Expected local score-table filenames
  models/                 # RankModel and MolCLR GCN definitions
  results/                # Lightweight result summaries and Enamine2M cumulatives
scripts/                  # Thin command-line wrappers
docs/                     # Reproducibility notes
examples/                 # Example shell commands
```

## Data Provenance

The score CSVs are not committed to this repository. Keep local copies under `rankdock/data/` or `data/`.

- `138M_scores.csv`: docking scores derived from the ultra-large library docking screen reported by Lyu et al., "Ultra-large library docking for discovering new chemotypes," *Nature* 566, 224-229 (2019), doi: [10.1038/s41586-019-0917-9](https://doi.org/10.1038/s41586-019-0917-9).
- `EnamineHTS_scores.csv`: Enamine HTS / Enamine2M docking benchmark table used through the MolPAL-style active-learning benchmark workflow; see Graff, Shakhnovich, and Coley, "Accelerating high-throughput virtual screening through molecular pool-based active learning," arXiv: [2012.07127](https://arxiv.org/abs/2012.07127).

Expected score-table columns:

```text
SMILES,Score
```

## Requirements

Reference environment:

- Linux workstation or server
- Python 3.10 or newer
- Conda or virtualenv
- PyTorch
- RDKit
- GPyTorch
- scikit-learn
- pandas, numpy, tqdm
- transformers for ChemBERTa embeddings
- PyTorch Geometric stack for MolCLR graph embeddings

The included `rankdock/environment.yaml` captures the conda environment used for the packaged workflow:

```bash
conda env create -f rankdock/environment.yaml
conda activate rankdock
```

GPU/CUDA is optional for smoke tests and analysis scripts, but strongly recommended for large embedding generation and neural model training. CPU execution is supported for lightweight validation.

## Installation

Install the package in editable mode from the repository root:

```bash
python -m pip install -e .
```

If graph embedding generation is needed, install the PyTorch Geometric packages that match your local PyTorch/CUDA build. The exact command depends on your CUDA and PyTorch versions.

## Running

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

Concatenate graph and SMILES embeddings:

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
  --model pairwise \
  --acq greedy \
  --out_csv outputs/top1_pairwise.csv
```

Recompute retrieval summaries from included cumulative selections:

```bash
python scripts/round_retrieval.py \
  --scores_csv data/EnamineHTS_scores.csv \
  --root rankdock/results/cumulative/enamine2m \
  --out_dir outputs/enamine2m_retrieval
```

## Validation

Before running large experiments, check that CSV rows and embedding parts are aligned:

```bash
python scripts/validate_embeddings.py \
  --csv data/merged_smiles.csv \
  --smiles_dir output/smiles_embeddings \
  --graph_dir output/graph_embeddings \
  --combined_dir output/combined_embeddings
```

For a concat check without ChemBERTa re-embedding:

```bash
python scripts/validate_embeddings.py \
  --csv data/EnamineHTS_scores.csv \
  --smiles_dir output/smiles/2M \
  --graph_dir output/graph/2M \
  --combined_dir output/combined_embeddings/2M \
  --device cpu
```

Syntax check for the public modules:

```bash
python -m py_compile \
  rankdock/active_learning.py \
  rankdock/data/embeddings.py \
  rankdock/sampling.py \
  rankdock/retrieval.py \
  rankdock/score_final.py \
  rankdock/validation.py
```

## Optional Settings

Useful active-learning options:

- `--model {mlp,triplet,pairwise,rf}`: choose the model baseline.
- `--acq {greedy,ucb,poi,eoi}`: choose acquisition mode.
- `--rounds`: number of active-learning rounds.
- `--select_frac`: fraction selected per round.
- `--seed`: reproducibility seed.
- `--round_selected_dir`: save per-round selections.
- `--round_cumulative_dir`: save cumulative selections.
- `--metrics_out`: save JSON metrics.
- `--cumulative_metrics_out`: save cumulative retrieval/diversity metrics.

Performance-related options:

- `--num_workers`
- `--model_train_bs`, `--model_pred_bs`
- `--rf_n_jobs`
- `--no_compile` to disable `torch.compile`

Set `ENABLE_TORCH_COMPILE=1` only if your PyTorch environment has working compile support.

## Outputs and Large Files

Large files are intentionally kept out of the repository:

- raw docking score CSVs
- embedding `part_*.npy` files
- pretrained checkpoints
- generated `output/` and `outputs/`
- receptor-specific docking folders

This repository includes lightweight summaries and Enamine2M cumulative selections under `rankdock/results/`. The larger 138M cumulative selection files should be distributed separately, for example through Git LFS, GitHub Releases, or an external artifact store.

## License

The RankDock source code is released under the MIT License. See [LICENSE](LICENSE).

Docking score tables, molecular libraries, pretrained checkpoints, and third-party model weights are not covered by this repository license. Use and redistribution of those assets should follow the terms from their original providers, including Enamine, MolPAL-related benchmark resources, MolCLR, ChemBERTa, and the cited docking datasets.
