#!/usr/bin/env bash
# Journal (D), queue E: score the SENSOR-DROPOUT ablation. NO training, GPU only for speed (minutes).
#   step 1  scripts/cert_dropout.py        the dropout models on every clip and condition (val + test)
#   step 2  scripts/cert_dropout_score.py  rows A-D of the ablation table, per test person
#
#   bash scripts/run_journal_queue_e.sh                    full run: all folds, seeds 0,1,2
#   FOLDS=0 SEEDS=0 bash scripts/run_journal_queue_e.sh    smoke run: fold 0, seed 0 (code check only)
#
# Steps run ONE AT A TIME (every loader rewrites the fold's normalisation file; see queue C), and not
# while another queue is running. A step whose output file exists is skipped.
# Log: logs/journal_queue_e.log (via tee) ; failed steps: runs/failures.txt
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

echo "START E $(date -Is) at $(git rev-parse --short HEAD) | folds $FOLDS seeds $SEEDS"
for d in d1v2 d2v; do
  for r in sd50 sd25; do
    out=$CE/dropout_${d}_${r}_$SFX.parquet
    if [ -f "$out" ]; then echo "skip (exists): $out"; continue; fi
    echo; echo "=== $(date -Is) RUN cert_dropout.py $d $r -> $out"
    python scripts/cert_dropout.py --dataset "$d" --rate "$r" --folds "$FOLDS" --seeds "$SEEDS" \
      || echo "$(date -Is) FAILED queue E: cert_dropout $d $r ($SFX)" >> runs/failures.txt
  done
done
echo "STEP 1 DONE (dropout models on the test bench) $(date -Is)"
for d in d1v2 d2v; do
  echo; echo "=== $(date -Is) SCORE $d"
  python scripts/cert_dropout_score.py --dataset "$d" --tag "$SFX" \
    || echo "$(date -Is) FAILED queue E: cert_dropout_score $d ($SFX)" >> runs/failures.txt
done
echo "ALL DONE E $(date -Is)"
