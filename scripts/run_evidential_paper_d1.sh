#!/usr/bin/env bash
# E fairness check on UTD-MHAD: TMC / ECML with the papers' optimiser settings
# (weight decay 1e-5; lr 3e-4 TMC, 3e-3 ECML), our epoch budget and early stopping. 8 folds x 3 seeds.
set -u
cd "$(dirname "$0")/.."
mkdir -p runs
for seed in 0 1 2; do
  for fold in $(seq 0 7); do
    for m in tmc_p ecml_p; do
      id=$(printf "BASE_d1_%s_loso_f%02d_s%d" "$m" "$fold" "$seed")
      [ -f "results/raw/${id}.parquet" ] && continue
      if [ "$m" = "ecml_p" ]; then extra="--lr 3e-3"; else extra=""; fi
      python scripts/train.py --dataset d1 --model "$m" --fold "$fold" --seed "$seed" --wd 1e-5 $extra \
        || echo "$(date -Is) FAILED ${id}" >> runs/failures.txt
    done
  done
done
echo "ALL DONE $(date -Is)"
