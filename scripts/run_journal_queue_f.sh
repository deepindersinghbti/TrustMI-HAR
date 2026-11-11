#!/usr/bin/env bash
# Journal (D), queue F: the FAULT test bench (docs/plan_faults_2026-11-11.md, steps 3 and 5). NO training.
# Every journal model is asked about every clip under every planned fault (drift, gap, shift at 3 levels),
# on the validation and test person of each fold, exactly as queue C did for the lies.
#   1. witnesses + the fused model's answers + the candidate backups   (cert_experts.py)
#   2. per-clip mutual information, own matcher                         (cert_mi.py)
#   3. TMC / ECML                                                       (cert_evidential.py)
#   4. the 50% sensor-dropout model (row D)                             (cert_dropout.py)
#
#   bash scripts/run_journal_queue_f.sh                              full run (needs the video fault features)
#   FOLDS=0 SEEDS=0 NOVIDEO=1 bash scripts/run_journal_queue_f.sh    smoke run, sensor faults only (minutes)
#
# Steps run ONE AT A TIME (each loader rewrites the fold's normalisation file). A step whose output file
# exists is skipped, so the script can simply be started again after any stop.
# Log: logs/journal_queue_f.log (via tee) ; failed steps: runs/failures.txt
set -u
cd "$(dirname "$0")/.."
mkdir -p logs runs
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "uncommitted changes: commit first, so the results belong to one git version"; exit 1
fi
source trustmi-env/bin/activate
FOLDS=${FOLDS:-all}
SEEDS=${SEEDS:-0,1,2}
NOVIDEO=${NOVIDEO:-0}
if [ "$FOLDS" = all ]; then TAG=all; else TAG="f${FOLDS//,/-}"; fi
if [ "$NOVIDEO" = 1 ]; then VF=--no-video-faults; FS=_faults-novideo; else VF=; FS=_faults; fi
SFX="${TAG}_s${SEEDS//,/-}"
CE=results/certificate

step () {      # output-file script args... : run unless the output exists
  local out=$1; shift
  if [ -f "$out" ]; then echo "skip (exists): $out"; return; fi
  echo; echo "=== $(date -Is) RUN $* -> $out"
  python "$@" --conditions faults $VF --folds "$FOLDS" --seeds "$SEEDS" \
    || echo "$(date -Is) FAILED queue F: $* ($SFX$FS)" >> runs/failures.txt
}

echo "START F $(date -Is) at $(git rev-parse --short HEAD) | folds $FOLDS seeds $SEEDS novideo $NOVIDEO"
step $CE/experts_d1v2_$SFX$FS.parquet scripts/cert_experts.py --dataset d1v2 --extra-models b4_noimu,b4_noskel,b4_novideo
step $CE/experts_d2v_$SFX$FS.parquet  scripts/cert_experts.py --dataset d2v  --extra-models b4s_nowatch,b4s_nophone,b4s
echo "STEP 1 DONE (witnesses) $(date -Is)"
for d in d1v2 d2v; do step $CE/mi_${d}_own_$SFX$FS.parquet scripts/cert_mi.py --dataset $d --version own; done
echo "STEP 2 DONE (MI) $(date -Is)"
for d in d1v2 d2v; do step $CE/evidential_${d}_$SFX$FS.parquet scripts/cert_evidential.py --dataset $d; done
echo "STEP 3 DONE (TMC / ECML) $(date -Is)"
for d in d1v2 d2v; do step $CE/dropout_${d}_sd50_$SFX$FS.parquet scripts/cert_dropout.py --dataset $d --rate sd50; done
echo "ALL DONE F (test bench) $(date -Is)"
