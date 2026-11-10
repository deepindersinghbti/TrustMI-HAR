#!/usr/bin/env bash
# Journal (D), queue C: put the trained journal models through the tests. NO training, GPU only for speed.
#   d1v2  UTD-MHAD, whole-frame video (B4, witnesses, TMC/ECML, MI matchers from queues A and B)
#   d2v   MMAct with video as a 3rd sensor (B4s3, ...)
# Lies are planted in EVERY sensor of both datasets (default of the journal datasets, cert_signals.CFG).
#
#   bash scripts/run_journal_queue_c.sh                    full run: all folds, seeds 0,1,2  (about 1-1.5 h)
#   FOLDS=0 SEEDS=0 bash scripts/run_journal_queue_c.sh    smoke run: fold 0, seed 0         (a few minutes)
#
# Steps run ONE AT A TIME on purpose: every loader rewrites the fold's normalisation file, which the
# operators read, so two scoring scripts must not run side by side.
# A step whose output file exists is skipped, so the script can simply be started again after any stop.
# Log: logs/journal_queue_c.log (via tee) ; failed steps: runs/failures.txt
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
SFX="${TAG}_s${SEEDS//,/-}"                 # the scripts' own naming: <folds>_s<seeds>
OP=results/operators; CE=results/certificate

step () {      # output-file script args... : run unless the output exists
  local out=$1; shift
  if [ -f "$out" ]; then echo "skip (exists): $out"; return; fi
  echo; echo "=== $(date -Is) RUN $* -> $out"
  python "$@" --folds "$FOLDS" --seeds "$SEEDS" \
    || echo "$(date -Is) FAILED queue C: $* ($SFX)" >> runs/failures.txt
}

echo "START C $(date -Is) at $(git rev-parse --short HEAD) | folds $FOLDS seeds $SEEDS"
# 1. lie tests: every operator on every sensor, through the fused model
step $OP/A3_d1v2_b4_$SFX.parquet    scripts/eval_operators.py --dataset d1v2 --model b4
step $OP/A3_d2v_b4s3_$SFX.parquet   scripts/eval_operators.py --dataset d2v  --model b4s3
echo "STEP 1 DONE (lie tests) $(date -Is)"
# 2. witnesses (single-sensor models) on every clip and condition, plus the answers of every candidate backup
step $CE/experts_d1v2_$SFX.parquet  scripts/cert_experts.py --dataset d1v2 --extra-models b4_noimu,b4_noskel,b4_novideo
step $CE/experts_d2v_$SFX.parquet   scripts/cert_experts.py --dataset d2v  --extra-models b4s_nowatch,b4s_nophone,b4s
echo "STEP 2 DONE (witnesses) $(date -Is)"
# 3. per-clip mutual information: main version (own) first, then the frozen comparison
for v in own frozen; do
  step $CE/mi_d1v2_${v}_$SFX.parquet scripts/cert_mi.py --dataset d1v2 --version $v
  step $CE/mi_d2v_${v}_$SFX.parquet  scripts/cert_mi.py --dataset d2v  --version $v
done
echo "STEP 3 DONE (MI) $(date -Is)"
# 4. literature baselines TMC / ECML on the same clips and lies
step $CE/evidential_d1v2_$SFX.parquet scripts/cert_evidential.py --dataset d1v2
step $CE/evidential_d2v_$SFX.parquet  scripts/cert_evidential.py --dataset d2v
echo "STEP 4 DONE (TMC / ECML) $(date -Is)"
# 5. witnesses judging TMC's and ECML's answers (certificate on top of an evidential model)
for m in tmc ecml; do
  step $CE/experts_d1v2_${SFX}_main-$m.parquet      scripts/cert_experts.py --dataset d1v2 --main $m
  step $CE/experts_d2v_${SFX}_main-${m}_s3.parquet  scripts/cert_experts.py --dataset d2v  --main ${m}_s3
done
echo "STEP 5 DONE (witnesses on TMC / ECML) $(date -Is)"
# 6. certificate v1 signs (harmless-change stability, hide-one-sensor): ablation rows; slowest step, so last
step $CE/signals_d1v2_$SFX.parquet  scripts/cert_signals.py --dataset d1v2
step $CE/signals_d2v_$SFX.parquet   scripts/cert_signals.py --dataset d2v
echo "ALL DONE C $(date -Is)"
