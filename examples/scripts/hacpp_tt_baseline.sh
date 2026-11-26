#!/usr/bin/env bash
# Example: run HAC++ on Tanks&Temples `train` (matches open_stats/static_rd.json benchmark).
#
# Prerequisites:
#   bash examples/_setup_hacpp.sh
#   conda activate HAC_env
#   tmc3 on PATH
#   data at examples/third_party/HAC-plus/data/tandt/train
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d "third_party/HAC-plus/train.py" ]; then
  echo "Run: bash _setup_hacpp.sh first"
  exit 1
fi

# Default rate point from upstream run_shell_tnt.py
python run_hacpp_baseline.py --dataset tandt --scene train --lmbda 0.004 "$@"

# RD sweep (uncomment):
# for lmbda in 0.004 0.002 0.001; do
#   python run_hacpp_baseline.py --dataset tandt --scene train --lmbda "$lmbda"
# done
