#!/usr/bin/env bash
# Journal (D), overnight queue B: MI matchers for d1v2 and d2v (docs/plan_mi_2026-11-04.md, step 3).
# Order: MAIN version (own) first, then the frozen comparison. MI-frozen needs the fold's witnesses
# (BASE_<d>_b1_<sensor>_..., trained by queue A); a fold/seed whose witnesses are missing is skipped
# with a message, so start this script again later to fill the gaps.
# MMAct: 1000 fresh random training clips per epoch (fixed 2026-11-04, unchanged).
# Skips finished runs (results/raw/MI_*.parquet). Log: logs/journal_queue_b.log (via tee).
set -u
cd "$(dirname "$0")/.."
mkdir -p logs runs
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "uncommitted changes: commit first, so every run records a clean git version"; exit 1
fi
source trustmi-env/bin/activate
declare -A SENSORS=( [d1v2]="video skel imu" [d2v]="watch phone video" )

train () {   # dataset version n_folds extra-args...
  local d=$1 v=$2 nf=$3; shift 3
  for s in 0 1 2; do
    for f in $(seq 0 $((nf - 1))); do
      id=$(printf "MI_%s_%s_loso_f%02d_s%d" "$d" "$v" "$f" "$s")
      [ -f "results/raw/$id.parquet" ] && continue
      if [ "$v" = frozen ]; then
        missing=""
        for m in ${SENSORS[$d]}; do
          w=$(printf "results/raw/BASE_%s_b1_%s_loso_f%02d_s%d.parquet" "$d" "$m" "$f" "$s")
          [ -f "$w" ] || missing="$missing $m"
        done
        if [ -n "$missing" ]; then echo "skip $id: witnesses not trained yet:$missing"; continue; fi
      fi
      python scripts/train_mi.py --dataset "$d" --version "$v" --fold "$f" --seed "$s" "$@" \
        || echo "$(date -Is) FAILED $id" >> runs/failures.txt
    done
  done
}

echo "START B $(date -Is) at $(git rev-parse --short HEAD)"
train d1v2 own 8
train d2v own 20 --anchors-per-epoch 1000
echo "MAIN VERSION DONE $(date -Is)"
train d1v2 frozen 8
train d2v frozen 20 --anchors-per-epoch 1000
echo "ALL DONE B $(date -Is)"
