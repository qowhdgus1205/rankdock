# Result Tables

This directory contains lightweight result CSVs that allow reviewers to inspect round-wise RankDock outputs without rerunning the full active-learning pipeline.

## Cumulative Selection CSVs

Round-wise cumulative selected compounds for Enamine2M are stored under:

```text
results/cumulative/enamine2m/<run>/round_cumulative/round_XX_cumulative.csv
```

The Enamine2M files are plain CSV. The 138M cumulative CSVs are not included in this GitHub repository because the compressed set is large for ordinary source control; keep them as release artifacts, external downloads, or Git LFS objects when distributing the full manuscript artifact bundle.

Each file has the same columns:

```text
selection_round,sample_id,SMILES,Score
```

You can load either plain or compressed files directly with pandas:

```python
import pandas as pd

df = pd.read_csv(
    "results/cumulative/enamine2m/pairwise_greedy/round_cumulative/round_10_cumulative.csv"
)
```

The file inventory is in:

```text
results/cumulative_manifest.csv
```

## Included Runs

Enamine2M:

- `rf_greedy`
- `mlp_greedy` (RankModel backbone trained with MSE regression loss)
- `triplet_greedy`
- `pairwise_greedy`

138M cumulative selections are excluded from this repository. The 138M summary metrics below are included.

## Summary Metrics

Precomputed round-wise retrieval summaries are stored under:

```text
results/summaries/enamine2m/
results/summaries/138m/
```

These summary files are the quickest way to reproduce the main round-wise retrieval tables from the manuscript.
