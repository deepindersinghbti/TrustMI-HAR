#!/usr/bin/env bash
# Certificate option A: UTD-MHAD backup model (B4 without the IMU = skeleton + video), 8 folds x 3 seeds.
set -u
cd "$(dirname "$0")/.."
for s in 0 1 2; do
  for f in 0 1 2 3 4 5 6 7; do
    id=$(printf "BASE_d1_b4_noimu_loso_f%02d_s%d" "$f" "$s")
    [ -f "results/raw/$id.parquet" ] && { echo "skip $id"; continue; }
    python scripts/train.py --dataset d1 --model b4_noimu --fold "$f" --seed "$s" \
      || echo "$id" >> runs/failures.txt
  done
done
echo "ALL DONE $(date '+%F %T')"
