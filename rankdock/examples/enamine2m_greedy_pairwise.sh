#!/usr/bin/env bash
set -euo pipefail

python scripts/run_rankdock.py \
  --smiles_csv data/EnamineHTS_scores.csv \
  --emb_dir output/combined_embeddings/2M \
  --init_csv outputs/enamine2m_lsh_init0p2/initial/lsh_initial_0p2pct.csv \
  --rounds 10 \
  --select_frac 0.001 \
  --final_top_frac 0.01 \
  --model pairwise \
  --acq greedy \
  --out_csv outputs/enamine2m_lsh_init0p2/bo/pairwise_greedy/top1pct.csv \
  --round_selected_dir outputs/enamine2m_lsh_init0p2/bo/pairwise_greedy/round_selected \
  --round_cumulative_dir outputs/enamine2m_lsh_init0p2/bo/pairwise_greedy/round_cumulative \
  --cumulative_metrics_out outputs/enamine2m_lsh_init0p2/bo/pairwise_greedy/cumulative_metrics.csv
