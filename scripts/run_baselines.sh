#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
mkdir -p runs
for seed in 0 1 2; do
  for fold in 0 1 2 3 4 5 6 7; do
    for m in b1_video b1_skel b1_imu b4; do
      id="BASE_d1_${m}_loso_f0${fold}_s${seed}"
      [ -f "results/raw/${id}.parquet" ] && continue
      python scripts/train.py --model "$m" --fold "$fold" --seed "$seed" \
        || echo "$(date -Is) FAILED ${id}" >> runs/failures.txt
    done
  done
done
for fold in 0 1 2 3 4 5 6 7; do
  id="N1_d1_b4_loso_f0${fold}_s0"
  [ -f "results/raw/${id}.parquet" ] && continue
  python scripts/train.py --model b4 --fold "$fold" --seed 0 --shuffle_labels \
    || echo "$(date -Is) FAILED ${id}" >> runs/failures.txt
done
