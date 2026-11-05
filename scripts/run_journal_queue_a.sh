#!/usr/bin/env bash
# Journal (D), overnight queue A: every train.py run for
#   d1v2  UTD-MHAD with whole-frame video features (v2)
#   d2v   MMAct with video as a 3rd sensor (main3 clips)
# Order = priority: what Friday's certificate needs comes first; TMC/ECML (3 sensors) last (can move to Tuesday).
# Skips finished runs (results/raw/<id>.parquet), so it can simply be started again after any stop.
# Log: logs/journal_queue_a.log (via tee) ; failed runs: runs/failures.txt
set -u
cd "$(dirname "$0")/.."
mkdir -p logs runs
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "uncommitted changes: commit first, so every run records a clean git version"; exit 1
fi
source trustmi-env/bin/activate

run () {       # dataset n_folds model : 3 seeds x all folds (seed-major, as the conference scripts)
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

control () {   # dataset n_folds model : shuffled-label negative control N1, seed 0
  local d=$1 nf=$2 m=$3
  for f in $(seq 0 $((nf - 1))); do
    id=$(printf "N1_%s_%s_loso_f%02d_s0" "$d" "$m" "$f")
    [ -f "results/raw/$id.parquet" ] && continue
    python scripts/train.py --dataset "$d" --model "$m" --fold "$f" --seed 0 --shuffle_labels \
      || echo "$(date -Is) FAILED $id" >> runs/failures.txt
  done
}

echo "START A $(date -Is) at $(git rev-parse --short HEAD)"
# 1. UTD-MHAD v2: the full conference d1 set (each run takes seconds)
for m in b1_video b1_skel b1_imu b4 b4_noimu b4_noskel b4_novideo tmc ecml; do run d1v2 8 "$m"; done
control d1v2 8 b4
echo "STEP 1 DONE (d1v2) $(date -Is)"
# 2. MMAct + video: single-sensor witnesses (queue B's MI-frozen needs them)
for m in b1_video b1_watch b1_phone; do run d2v 20 "$m"; done
echo "STEP 2 DONE (d2v witnesses) $(date -Is)"
# 3. MMAct + video: main model, drop-one-sensor backups, and watch+phone on the same clips
for m in b4s3 b4s_nowatch b4s_nophone b4s; do run d2v 20 "$m"; done
echo "STEP 3 DONE (d2v main + backups) $(date -Is)"
# 4. negative control for the main model
control d2v 20 b4s3
echo "STEP 4 DONE (d2v control) $(date -Is)"
# 5. literature baselines with 3 sensors
for m in tmc_s3 ecml_s3; do run d2v 20 "$m"; done
echo "ALL DONE A $(date -Is)"
