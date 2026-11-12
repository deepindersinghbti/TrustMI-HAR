#!/usr/bin/env bash
# Journal, queue J: the IMU-off test on WearGait-PD (docs/plan_followup_2026-11-12.md, Part A).
# NO training. Puts the trained d3 models (queue H) through the IMU-off bench: k IMUs of one body group switched
# off (training mean) for the whole clip, k = 1 .. all, on the clean test clips of every fold.
#
#   FOLDS=0 SEEDS=0 bash scripts/run_journal_queue_j.sh     smoke run (fold 0, seed 0; code check)
#   bash scripts/run_journal_queue_j.sh                     full run (all 10 folds, seeds 0,1,2)
#
# Steps run ONE AT A TIME (every loader rewrites the fold's normalisation file). A step whose output exists is
# skipped, so the script can be started again after any stop.
# Log: logs/journal_queue_j.log (via tee) ; failed steps: runs/failures.txt
set -u
cd "$(dirname "$0")/.."
mkdir -p logs runs
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "uncommitted changes: commit first, so the results belong to one git version"; exit 1
fi
source trustmi-env/bin/activate
FOLDS=${FOLDS:-all}
SEEDS=${SEEDS:-0,1,2}
if [ "$FOLDS" = all ]; then TAG=all; else TAG="f${FOLDS//,/-}"; fi
SFX="${TAG}_s${SEEDS//,/-}"
CE=results/certificate
D=d3; EXTRA=b4w_noarms,b4w_notrunk,b4w_nolegs

step () {      # output-file script args... : run unless the output exists
  local out=$1; shift
  if [ -f "$out" ]; then echo "skip (exists): $out"; return; fi
  echo; echo "=== $(date -Is) RUN $* -> $out"
  python "$@" --folds "$FOLDS" --seeds "$SEEDS" \
    || echo "$(date -Is) FAILED queue J: $* ($SFX)" >> runs/failures.txt
}

echo "START J $(date -Is) at $(git rev-parse --short HEAD) | folds $FOLDS seeds $SEEDS"
step $CE/experts_${D}_${SFX}_imuoff.parquet scripts/cert_experts.py --dataset $D --conditions imuoff --extra-models $EXTRA
step $CE/mi_${D}_own_${SFX}_imuoff.parquet scripts/cert_mi.py --dataset $D --version own --conditions imuoff
step $CE/evidential_${D}_${SFX}_imuoff.parquet scripts/cert_evidential.py --dataset $D --conditions imuoff
step $CE/dropout_${D}_sd50_${SFX}_imuoff.parquet scripts/cert_dropout.py --dataset $D --rate sd50 --conditions imuoff
echo "ALL DONE J $(date -Is)"
