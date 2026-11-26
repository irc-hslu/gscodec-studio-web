#!/usr/bin/env bash
# Example: run ContextGS on Tanks&Temples `train` (matches open_stats/static_rd.json).
#
# Prerequisites:
#   bash examples/_setup_contextgs.sh
#   conda activate contextgs
#   data at examples/third_party/ContextGS/data/tandt/train
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f "third_party/ContextGS/train.py" ]; then
  echo "Run: bash _setup_contextgs.sh first"
  exit 1
fi

python run_contextgs_baseline.py --dataset tandt --scene train --lmbda 0.004 "$@"

# RD sweep (uncomment):
# for lmbda in 0.004 0.002 0.001 0.0005; do
#   python run_contextgs_baseline.py --dataset tandt --scene train --lmbda "$lmbda"
# done
