# Data Files

This directory contains code for molecular datasets and is also the default location for local docking score tables.

The large docking score CSVs are not committed to the repository. Download or reconstruct them from the original data sources, then place them here with these filenames:

```text
data/138M_scores.csv
data/EnamineHTS_scores.csv
```

Expected columns:

```text
SMILES,Score
```

Data provenance:

- `138M_scores.csv`: Lyu et al., "Ultra-large library docking for discovering new chemotypes," *Nature* 566, 224-229 (2019), doi: `10.1038/s41586-019-0917-9`.
- `EnamineHTS_scores.csv`: Enamine HTS / Enamine2M docking benchmark table used through the MolPAL-style active-learning benchmark workflow; see Graff, Shakhnovich, and Coley, "Accelerating high-throughput virtual screening through molecular pool-based active learning," arXiv: `2012.07127`.
