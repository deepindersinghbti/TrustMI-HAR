#!/usr/bin/env bash
# Journal, queue L: Addition 1 of docs/plan_additions_2026-11-14.md (stuck sensor + saturation). NO training.
#   1. video fault features made on the frames: UTD-MHAD (about a minute), MMAct (about 2 h, resumable by subject)
#   2. every model run on the new bench (--conditions faults2) for UTD-MHAD, MMAct and WearGait-PD
#
#   SMOKE=1 bash scripts/run_journal_queue_l.sh    smoke run: fold 0 seed 0; MMAct video timed on ONE subject
#                                                  (its shard is kept and reused); MMAct without video faults
#   bash scripts/run_journal_queue_l.sh            full run (all folds, seeds 0,1,2)
#
# Steps run ONE AT A TIME. A step whose output exists is skipped, so the script can be started again after any stop.
# Log: logs/journal_queue_l.log ; failures: runs/failures.txt
set -u
cd "$(dirname "$0")/.."
mkdir -p logs runs
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "uncommitted changes: commit first, so the results belong to one git version"; exit 1
fi
source trustmi-env/bin/activate
SMOKE=${SMOKE:-0}
if [ "$SMOKE" = 1 ]; then FOLDS=0; SEEDS=0; else FOLDS=all; SEEDS=0,1,2; fi
CE=results/certificate
if [ "$FOLDS" = all ]; then S="all_s${SEEDS//,/-}"; else S="f${FOLDS//,/-}_s${SEEDS//,/-}"; fi

step () {      # output-file script args... : run unless the output exists
  local out=$1; shift
  if [ -f "$out" ]; then echo "skip (exists): $out"; return; fi
  echo; echo "=== $(date -Is) RUN $* -> $out"
  python "$@" --folds "$FOLDS" --seeds "$SEEDS" \
    || echo "$(date -Is) FAILED queue L: $* (folds $FOLDS seeds $SEEDS)" >> runs/failures.txt
}

{
echo "START L $(date -Is) at $(git rev-parse --short HEAD) | smoke $SMOKE | folds $FOLDS | seeds $SEEDS"
if [ -f data/features/video_r2plus1d18_v2_faults2.npz ]; then echo "skip (exists): UTD-MHAD video faults2"
else echo; echo "=== $(date -Is) RUN video faults2 UTD-MHAD"
     python scripts/extract_video_faults2.py --dataset utd || echo "$(date -Is) FAILED queue L: video utd" >> runs/failures.txt; fi
if [ "$SMOKE" = 1 ]; then
  echo; echo "=== $(date -Is) RUN video faults2 MMAct, timing run on subject 1"
  python scripts/extract_video_faults2.py --dataset mmact --subjects 1 || echo "$(date -Is) FAILED queue L: video mmact s1" >> runs/failures.txt
elif [ -f data/features/mmact_video_r2plus1d18_v2_faults2.npz ]; then echo "skip (exists): MMAct video faults2"
else echo; echo "=== $(date -Is) RUN video faults2 MMAct, all subjects"
     python scripts/extract_video_faults2.py --dataset mmact || echo "$(date -Is) FAILED queue L: video mmact" >> runs/failures.txt; fi

for D in d1v2 d2v d3; do
  case $D in
    d1v2) X=b4_noimu,b4_noskel,b4_novideo ;;
    d2v)  X=b4s_nowatch,b4s_nophone,b4s ;;
    d3)   X=b4w_noarms,b4w_notrunk,b4w_nolegs ;;
  esac
  NV=""; SUF=_faults2
  if [ "$SMOKE" = 1 ] && [ $D = d2v ]; then NV=--no-video-faults; SUF=_faults2-novideo; fi
  step $CE/experts_${D}_${S}${SUF}.parquet scripts/cert_experts.py --dataset $D --conditions faults2 --extra-models $X $NV
  step $CE/mi_${D}_own_${S}${SUF}.parquet scripts/cert_mi.py --dataset $D --version own --conditions faults2 $NV
  step $CE/evidential_${D}_${S}${SUF}.parquet scripts/cert_evidential.py --dataset $D --conditions faults2 $NV
  step $CE/dropout_${D}_sd50_${S}${SUF}.parquet scripts/cert_dropout.py --dataset $D --rate sd50 --conditions faults2 $NV
done
echo "ALL DONE L $(date -Is)"
} 2>&1 | tee -a logs/journal_queue_l.log
