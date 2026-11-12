#!/usr/bin/env bash
# Journal, queue I: put the trained WearGait-PD models (dataset d3, queue H) through every test bench.
# NO training. Same steps as queues C (lies), E (dropout model), F (faults) and G (missing data) for MMAct.
# Needs: queue H finished, and the d3 backups chosen and committed:
#     python scripts/choose_backups_journal.py --datasets d3 && git add configs/backups.json && git commit ...
#
#   FOLDS=0 SEEDS=0 bash scripts/run_journal_queue_i.sh     smoke run (fold 0, seed 0; code check)
#   bash scripts/run_journal_queue_i.sh                     full run (all 10 folds, seeds 0,1,2)
#
# Steps run ONE AT A TIME (every loader rewrites the fold's normalisation file, which the operators read).
# A step whose output exists is skipped, so the script can be started again after any stop.
# Log: logs/journal_queue_i.log (via tee) ; failed steps: runs/failures.txt
set -u
cd "$(dirname "$0")/.."
mkdir -p logs runs
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "uncommitted changes: commit first, so the results belong to one git version"; exit 1
fi
if ! grep -q '"d3"' configs/backups.json; then
  echo "configs/backups.json has no d3 entry: run scripts/choose_backups_journal.py --datasets d3 and commit"; exit 1
fi
source trustmi-env/bin/activate
FOLDS=${FOLDS:-all}
SEEDS=${SEEDS:-0,1,2}
if [ "$FOLDS" = all ]; then TAG=all; else TAG="f${FOLDS//,/-}"; fi
SFX="${TAG}_s${SEEDS//,/-}"
OP=results/operators; CE=results/certificate
D=d3; EXTRA=b4w_noarms,b4w_notrunk,b4w_nolegs

step () {      # output-file script args... : run unless the output exists
  local out=$1; shift
  if [ -f "$out" ]; then echo "skip (exists): $out"; return; fi
  echo; echo "=== $(date -Is) RUN $* -> $out"
  python "$@" --folds "$FOLDS" --seeds "$SEEDS" \
    || echo "$(date -Is) FAILED queue I: $* ($SFX)" >> runs/failures.txt
}

echo "START I $(date -Is) at $(git rev-parse --short HEAD) | folds $FOLDS seeds $SEEDS"
# 1. lies: every operator on every sensor group, through the fused model (operator table)
step $OP/A3_${D}_b4w_$SFX.parquet scripts/eval_operators.py --dataset $D --model b4w
# 2. lie bench: witnesses + fused model + backups; MI (own = main, frozen = comparison); TMC / ECML
step $CE/experts_${D}_$SFX.parquet scripts/cert_experts.py --dataset $D --extra-models $EXTRA
for v in own frozen; do step $CE/mi_${D}_${v}_$SFX.parquet scripts/cert_mi.py --dataset $D --version $v; done
step $CE/evidential_${D}_$SFX.parquet scripts/cert_evidential.py --dataset $D
for m in tmc_w ecml_w; do
  step $CE/experts_${D}_${SFX}_main-$m.parquet scripts/cert_experts.py --dataset $D --main $m
done
step $CE/dropout_${D}_sd50_$SFX.parquet scripts/cert_dropout.py --dataset $D --rate sd50
echo "STEP 2 DONE (lie bench) $(date -Is)"
# 3. real sensor faults (drift, gap, timing error on every group)
for c in faults realdrop; do
  step $CE/experts_${D}_${SFX}_$c.parquet scripts/cert_experts.py --dataset $D --conditions $c --extra-models $EXTRA
  step $CE/mi_${D}_own_${SFX}_$c.parquet scripts/cert_mi.py --dataset $D --version own --conditions $c
  step $CE/evidential_${D}_${SFX}_$c.parquet scripts/cert_evidential.py --dataset $D --conditions $c
  step $CE/dropout_${D}_sd50_${SFX}_$c.parquet scripts/cert_dropout.py --dataset $D --rate sd50 --conditions $c
  echo "STEP $c DONE $(date -Is)"
done
# 4. certificate v1 signs (needed only for the dropout ablation's row B); slowest step, so last
step $CE/signals_${D}_$SFX.parquet scripts/cert_signals.py --dataset $D
echo "ALL DONE I $(date -Is)"
