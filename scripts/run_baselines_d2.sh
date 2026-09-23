#!/usr/bin/env bash
# MMAct sensor baselines: B1-watch, B1-phone, B4-sensor x 20 LOSO folds x 3 seeds, + N1 control (seed 0).
set -u
cd "$(dirname "$0")/.."
mkdir -p runs
for seed in 0 1 2; do
  for fold in $(seq 0 19); do
    for m in b1_watch b1_phone b4s; do
      id=$(printf "BASE_d2_%s_loso_f%02d_s%d" "$m" "$fold" "$seed")
      [ -f "results/raw/${id}.parquet" ] && continue
      python scripts/train.py --dataset d2 --model "$m" --fold "$fold" --seed "$seed" \
        || echo "$(date -Is) FAILED ${id}" >> runs/failures.txt
    done
  done
done
for fold in $(seq 0 19); do
  id=$(printf "N1_d2_b4s_loso_f%02d_s0" "$fold")
  [ -f "results/raw/${id}.parquet" ] && continue
  python scripts/train.py --dataset d2 --model b4s --fold "$fold" --seed 0 --shuffle_labels \
    || echo "$(date -Is) FAILED ${id}" >> runs/failures.txt
done
echo "ALL DONE $(date -Is)"
