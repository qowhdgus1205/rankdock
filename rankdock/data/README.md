# Data Files

This directory contains code for molecular datasets and is also the default location for local docking score tables.

The large docking score CSVs used in the manuscript are not committed to the repository. Download or reconstruct them from the original data source, then place them here with these filenames:

```text
data/138M_scores.csv
data/EnamineHTS_scores.csv
```

Expected columns:

```text
SMILES,Score
```

The 138M docking data should be cited to Lyu et al., *Nature* 566, 224-229 (2019), "Ultra-large library docking for discovering new chemotypes".
