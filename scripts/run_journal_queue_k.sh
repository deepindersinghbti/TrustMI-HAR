#!/usr/bin/env bash
# Journal, queue K: Additions 3 and 2 of docs/plan_additions_2026-11-14.md. NO training.
#   Addition 3: MMAct no-watch clips with the watch as raw 0          (--conditions nowatch2, d2v)
#   Addition 2: WearGait-PD IMUs off during 25 / 50 / 75 % of the clip (--conditions imuoff2, d3)
#
#   FOLDS_NW=19 FOLDS_IO=0 SEEDS=0 bash scripts/run_journal_queue_k.sh     smoke run (code check)
#   bash scripts/run_journal_queue_k.sh                                    full run (all folds, seeds 0,1,2)
#
# Steps run ONE AT A TIME (every loader rewrites the fold's normalisation file). A step whose output exists is
# skipped, so the script can be started again after any stop. Log: logs/journal_queue_k.log ; failures: runs/failures.txt
set -u
cd "$(dirname "$0")/.."
mkdir -p logs runs
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "uncommitted changes: commit first, so the results belong to one git version"; exit 1
fi
source trustmi-env/bin/activate
FOLDS_NW=${FOLDS_NW:-all}
FOLDS_IO=${FOLDS_IO:-all}
SEEDS=${SEEDS:-0,1,2}
CE=results/certificate
sfx () { if [ "$1" = all ]; then echo "all_s${SEEDS//,/-}"; else echo "f${1//,/-}_s${SEEDS//,/-}"; fi; }

step () {      # folds output-file script args... : run unless the output exists
  local folds=$1 out=$2; shift 2
  if [ -f "$out" ]; then echo "skip (exists): $out"; return; fi
  echo; echo "=== $(date -Is) RUN $* -> $out"
  python "$@" --folds "$folds" --seeds "$SEEDS" \
    || echo "$(date -Is) FAILED queue K: $* (folds $folds seeds $SEEDS)" >> runs/failures.txt
}

{
echo "START K $(date -Is) at $(git rev-parse --short HEAD) | nowatch2 folds $FOLDS_NW | imuoff2 folds $FOLDS_IO | seeds $SEEDS"
S=$(sfx "$FOLDS_NW"); D=d2v; X=b4s_nowatch,b4s_nophone,b4s
step "$FOLDS_NW" $CE/experts_${D}_${S}_nowatch2.parquet scripts/cert_experts.py --dataset $D --conditions nowatch2 --extra-models $X
step "$FOLDS_NW" $CE/mi_${D}_own_${S}_nowatch2.parquet scripts/cert_mi.py --dataset $D --version own --conditions nowatch2
step "$FOLDS_NW" $CE/evidential_${D}_${S}_nowatch2.parquet scripts/cert_evidential.py --dataset $D --conditions nowatch2
step "$FOLDS_NW" $CE/dropout_${D}_sd50_${S}_nowatch2.parquet scripts/cert_dropout.py --dataset $D --rate sd50 --conditions nowatch2
S=$(sfx "$FOLDS_IO"); D=d3; X=b4w_noarms,b4w_notrunk,b4w_nolegs
step "$FOLDS_IO" $CE/experts_${D}_${S}_imuoff2.parquet scripts/cert_experts.py --dataset $D --conditions imuoff2 --extra-models $X
step "$FOLDS_IO" $CE/mi_${D}_own_${S}_imuoff2.parquet scripts/cert_mi.py --dataset $D --version own --conditions imuoff2
step "$FOLDS_IO" $CE/evidential_${D}_${S}_imuoff2.parquet scripts/cert_evidential.py --dataset $D --conditions imuoff2
step "$FOLDS_IO" $CE/dropout_${D}_sd50_${S}_imuoff2.parquet scripts/cert_dropout.py --dataset $D --rate sd50 --conditions imuoff2
echo "ALL DONE K $(date -Is)"
} 2>&1 | tee -a logs/journal_queue_k.log
