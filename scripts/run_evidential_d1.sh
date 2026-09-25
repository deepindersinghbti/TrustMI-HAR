#!/usr/bin/env bash
# Literature baselines (E) on UTD-MHAD: TMC + ECML x 8 LOSO folds x 3 seeds.
set -u
cd "$(dirname "$0")/.."
mkdir -p runs
for seed in 0 1 2; do
  for fold in $(seq 0 7); do
    for m in tmc ecml; do
      id=$(printf "BASE_d1_%s_loso_f%02d_s%d" "$m" "$fold" "$seed")
      [ -f "results/raw/${id}.parquet" ] && continue
      python scripts/train.py --dataset d1 --model "$m" --fold "$fold" --seed "$seed" \
        || echo "$(date -Is) FAILED ${id}" >> runs/failures.txt
    done
  done
done
echo "ALL DONE $(date -Is)"
