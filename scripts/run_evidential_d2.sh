#!/usr/bin/env bash
# Literature baselines (E) on MMAct: TMC-s + ECML-s (watch + phone) x 20 LOSO folds x 3 seeds.
set -u
cd "$(dirname "$0")/.."
mkdir -p runs
for seed in 0 1 2; do
  for fold in $(seq 0 19); do
    for m in tmc_s ecml_s; do
      id=$(printf "BASE_d2_%s_loso_f%02d_s%d" "$m" "$fold" "$seed")
      [ -f "results/raw/${id}.parquet" ] && continue
      python scripts/train.py --dataset d2 --model "$m" --fold "$fold" --seed "$seed" \
        || echo "$(date -Is) FAILED ${id}" >> runs/failures.txt
    done
  done
done
echo "ALL DONE $(date -Is)"
