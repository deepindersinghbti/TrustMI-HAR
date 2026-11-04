#!/usr/bin/env bash
# Plan steps 3 + 4 overnight: train every MI matcher, then per-clip MI for every condition.
# Order: MAIN version (own) on both datasets incl. per-clip MI first, then the frozen comparison.
# Skips finished runs (results/raw/MI_*.parquet), so it can simply be started again after any stop.
# MMAct: 1000 fresh random training clips per epoch (fixed 2026-11-04 from the timing run, before any result).
# Log: logs/mi_overnight.log ; failed runs: runs/failures.txt
set -u
cd "$(dirname "$0")/.."
mkdir -p logs runs
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "uncommitted changes: commit first, so every run records a clean git version"; exit 1
fi
source trustmi-env/bin/activate

train () {   # dataset version folds extra-args...
  local d=$1 v=$2 nf=$3; shift 3
  for s in 0 1 2; do
    for f in $(seq 0 $((nf - 1))); do
      id=$(printf "MI_%s_%s_loso_f%02d_s%d" "$d" "$v" "$f" "$s")
      [ -f "results/raw/$id.parquet" ] && continue
      python scripts/train_mi.py --dataset "$d" --version "$v" --fold "$f" --seed "$s" "$@" \
        || echo "$(date -Is) FAILED $id" >> runs/failures.txt
    done
  done
}

score () {   # dataset version extra-args...
  local d=$1 v=$2; shift 2
  python scripts/cert_mi.py --dataset "$d" --version "$v" "$@" \
    || echo "$(date -Is) FAILED cert_mi $d $v" >> runs/failures.txt
}

echo "START $(date -Is) at $(git rev-parse --short HEAD)"
train d1 own 8
train d2 own 20 --anchors-per-epoch 1000
score d1 own --attack imu,skel,video
score d2 own
echo "MAIN VERSION DONE $(date -Is)"
train d1 frozen 8
train d2 frozen 20 --anchors-per-epoch 1000
score d1 frozen --attack imu,skel,video
score d2 frozen
echo "ALL DONE $(date -Is)"
