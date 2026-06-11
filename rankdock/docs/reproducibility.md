# Reproducibility Notes

This repository is organized so that model comparisons isolate the training objective as much as possible.

## Model Baselines

All neural M2 baselines use the same `RankModel` backbone:

- `mlp`: MSE regression on docking scores
- `triplet`: triplet ranking objective
- `pairwise`: pairwise logistic ranking objective

All public neural baselines use the same `RankModel` backbone, so reported differences isolate the training objective rather than changing architecture and loss at the same time.

`rf` remains a non-neural random forest baseline.

## Large Files

The following are intentionally not tracked:

- docking score CSVs
- embedding `part_*.npy` files
- pretrained checkpoints
- generated `outputs/`
- receptor-specific docking folders

For a clean public release, provide either download links or small toy examples instead of committing these artifacts.
