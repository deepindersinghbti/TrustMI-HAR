#!/usr/bin/env bash
# Journal (D), queue D: SENSOR-DROPOUT ablation (src/trustmi/sensor_dropout.py).
# Retrains the main fused models with one sensor hidden in part of the training clips, then puts them through
# the same lie tests as the standard models. Rates fixed in advance: sd50 = 0.50 (main), sd25 = 0.25 (sensitivity).
#   d1v2  b4_sd50, b4_sd25       8 folds x 3 seeds each  (seconds per run)
#   d2v   b4s3_sd50, b4s3_sd25  20 folds x 3 seeds each  (a few minutes per run; about 3-5 h per rate)
# Order = priority: both UTD-MHAD models and the main MMAct rate first, the sensitivity rate last.
# Skips finished runs / existing output files, so it can simply be started again after any stop.
# Log: logs/journal_queue_d.log (via tee) ; failed runs: runs/failures.txt
set -u
cd "$(dirname "$0")/.."
mkdir -p logs runs
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "uncommitted changes: commit first, so every run records a clean git version"; exit 1
fi
source trustmi-env/bin/activate

run () {       # dataset n_folds model : 3 seeds x all folds (seed-major, as queue A)
  local d=$1 nf=$2 m=$3
  for s in 0 1 2; do
    for f in $(seq 0 $((nf - 1))); do
      id=$(printf "BASE_%s_%s_loso_f%02d_s%d" "$d" "$m" "$f" "$s")
      [ -f "results/raw/$id.parquet" ] && continue
      python scripts/train.py --dataset "$d" --model "$m" --fold "$f" --seed "$s" \
        || echo "$(date -Is) FAILED $id" >> runs/failures.txt
    done
  done
}

lies () {      # dataset model : every operator on every sensor through this model (as queue C, step 1)
  local d=$1 m=$2 out="results/operators/A3_${1}_${2}_all_s0-1-2.parquet"
  if [ -f "$out" ]; then echo "skip (exists): $out"; return; fi
  echo; echo "=== $(date -Is) lie tests $d $m -> $out"
  python scripts/eval_operators.py --dataset "$d" --model "$m" \
    || echo "$(date -Is) FAILED queue D lie tests: $d $m" >> runs/failures.txt
}

echo "START D $(date -Is) at $(git rev-parse --short HEAD)"
# 1. UTD-MHAD, both rates (fast; also the first live check of the new training option)
for m in b4_sd50 b4_sd25; do run d1v2 8 "$m"; lies d1v2 "$m"; done
echo "STEP 1 DONE (d1v2, both rates) $(date -Is)"
# 2. MMAct, main rate
run d2v 20 b4s3_sd50; lies d2v b4s3_sd50
echo "STEP 2 DONE (d2v sd50) $(date -Is)"
# 3. MMAct, sensitivity rate
run d2v 20 b4s3_sd25; lies d2v b4s3_sd25
echo "ALL DONE D $(date -Is)"
