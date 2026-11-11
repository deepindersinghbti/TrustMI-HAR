#!/usr/bin/env bash
# Journal, queue H: every training run for WearGait-PD (dataset d3; docs/plan_weargait_2026-11-11.md).
# 10 person-wise folds x 3 seeds; same models, settings and order as the MMAct runs (queues A, B, D).
#
#   SMOKE=1 bash scripts/run_journal_queue_h.sh     code check (minutes): every model overfits 8 clips, the MI
#                                                   matcher's loss falls, then ONE real run of b4w and of
#                                                   b1_legs (fold 0, seed 0) to time a full run
#   bash scripts/run_journal_queue_h.sh             full run (1-2 days)
#
# Skips finished runs (results/raw/<id>.parquet), so it can simply be started again after any stop.
# Log: logs/journal_queue_h.log (via tee) ; failed runs: runs/failures.txt
set -u
cd "$(dirname "$0")/.."
mkdir -p logs runs
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "uncommitted changes: commit first, so every run records a clean git version"; exit 1
fi
source trustmi-env/bin/activate
D=d3; NF=10

run () {       # model : 3 seeds x all folds (seed-major)
  local m=$1
  for s in 0 1 2; do
    for f in $(seq 0 $((NF - 1))); do
      id=$(printf "BASE_%s_%s_loso_f%02d_s%d" "$D" "$m" "$f" "$s")
      [ -f "results/raw/$id.parquet" ] && continue
      python scripts/train.py --dataset "$D" --model "$m" --fold "$f" --seed "$s" \
        || echo "$(date -Is) FAILED $id" >> runs/failures.txt
    done
  done
}

control () {   # model : shuffled-label negative control, seed 0
  local m=$1
  for f in $(seq 0 $((NF - 1))); do
    id=$(printf "N1_%s_%s_loso_f%02d_s0" "$D" "$m" "$f")
    [ -f "results/raw/$id.parquet" ] && continue
    python scripts/train.py --dataset "$D" --model "$m" --fold "$f" --seed 0 --shuffle_labels \
      || echo "$(date -Is) FAILED $id" >> runs/failures.txt
  done
}

mi () {        # version : MI matcher, 1,000 anchors per epoch as for MMAct (queue B)
  local v=$1
  for s in 0 1 2; do
    for f in $(seq 0 $((NF - 1))); do
      id=$(printf "MI_%s_%s_loso_f%02d_s%d" "$D" "$v" "$f" "$s")
      [ -f "results/raw/$id.parquet" ] && continue
      python scripts/train_mi.py --dataset "$D" --version "$v" --fold "$f" --seed "$s" --anchors-per-epoch 1000 \
        || echo "$(date -Is) FAILED $id" >> runs/failures.txt
    done
  done
}

if [ "${SMOKE:-0}" = 1 ]; then
  echo "SMOKE H $(date -Is) at $(git rev-parse --short HEAD)"
  for m in b1_arms b1_trunk b1_legs b4w b4w_noarms b4w_notrunk b4w_nolegs tmc_w ecml_w b4w_sd50; do
    python scripts/train.py --dataset $D --model $m --fold 0 --smoke || echo "SMOKE FAILED $m"
  done
  python scripts/train_mi.py --dataset $D --version own --fold 0 --smoke || echo "SMOKE FAILED MI own"
  for m in b4w b1_legs; do
    t0=$(date +%s)
    python scripts/train.py --dataset $D --model $m --fold 0 --seed 0 || echo "SMOKE FAILED $m full run"
    echo "timing: $m fold 0 seed 0 took $(( $(date +%s) - t0 )) s"
  done
  echo "SMOKE H DONE $(date -Is)"
  exit 0
fi

echo "START H $(date -Is) at $(git rev-parse --short HEAD)"
for m in b1_arms b1_trunk b1_legs; do run "$m"; done
echo "STEP 1 DONE (witnesses) $(date -Is)"
for m in b4w b4w_noarms b4w_notrunk b4w_nolegs; do run "$m"; done
echo "STEP 2 DONE (main + backups) $(date -Is)"
control b4w
echo "STEP 3 DONE (negative control) $(date -Is)"
run b4w_sd50
echo "STEP 4 DONE (sensor-dropout model) $(date -Is)"
for m in tmc_w ecml_w; do run "$m"; done
echo "STEP 5 DONE (TMC / ECML) $(date -Is)"
mi own
mi frozen
echo "ALL DONE H $(date -Is)"
