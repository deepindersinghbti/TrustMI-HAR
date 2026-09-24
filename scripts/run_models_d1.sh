#!/usr/bin/env bash
# Train the UTD-MHAD models given as arguments, 8 folds x 3 seeds each; skips runs whose result exists.
set -u
cd "$(dirname "$0")/.."
for model in "$@"; do
  for s in 0 1 2; do
    for f in 0 1 2 3 4 5 6 7; do
      id=$(printf "BASE_d1_%s_loso_f%02d_s%d" "$model" "$f" "$s")
      [ -f "results/raw/$id.parquet" ] && { echo "skip $id"; continue; }
      python scripts/train.py --dataset d1 --model "$model" --fold "$f" --seed "$s" \
        || echo "$id" >> runs/failures.txt
    done
  done
done
echo "ALL DONE $(date '+%F %T')"
