#!/usr/bin/env bash
# Journal, queue G: the 247 MMAct clips WITHOUT a watch (docs/plan_nowatch_2026-11-11.md, step 5). NO training.
# Only d2v, only the 6 folds whose test person has no-watch clips (1, 9, 10, 12, 15, 19). Per fold:
#   part 'test'     clean + r_fill:watch on the test person's normal clips (reference, and the stored-run check)
#   part 'nowatch'  nw_fill:watch on the no-watch clips (the watch set to its training mean)
#   1. witnesses + the fused model (also with the watch flagged absent) + the backups   (cert_experts.py)
#   2. per-clip mutual information, own matcher                                        (cert_mi.py)
#   3. TMC / ECML                                                                      (cert_evidential.py)
#   4. the 50% sensor-dropout model (row D)                                            (cert_dropout.py)
#
#   bash scripts/run_journal_queue_g.sh                         full run (a few minutes)
#   FOLDS=19 SEEDS=0 bash scripts/run_journal_queue_g.sh        smoke run (person 20, one seed)
#
# Steps run ONE AT A TIME. A step whose output file exists is skipped, so the script can simply be started
# again after any stop. Log: logs/journal_queue_g.log (via tee) ; failed steps: runs/failures.txt
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
FS=_nowatch
CE=results/certificate

step () {      # output-file script args... : run unless the output exists
  local out=$1; shift
  if [ -f "$out" ]; then echo "skip (exists): $out"; return; fi
  echo; echo "=== $(date -Is) RUN $* -> $out"
  python "$@" --conditions nowatch --folds "$FOLDS" --seeds "$SEEDS" \
    || echo "$(date -Is) FAILED queue G: $* ($SFX$FS)" >> runs/failures.txt
}

echo "START G $(date -Is) at $(git rev-parse --short HEAD) | folds $FOLDS seeds $SEEDS"
step $CE/experts_d2v_$SFX$FS.parquet scripts/cert_experts.py --dataset d2v --extra-models b4s_nowatch,b4s_nophone,b4s
echo "STEP 1 DONE (witnesses) $(date -Is)"
step $CE/mi_d2v_own_$SFX$FS.parquet scripts/cert_mi.py --dataset d2v --version own
echo "STEP 2 DONE (MI) $(date -Is)"
step $CE/evidential_d2v_$SFX$FS.parquet scripts/cert_evidential.py --dataset d2v
echo "STEP 3 DONE (TMC / ECML) $(date -Is)"
step $CE/dropout_d2v_sd50_$SFX$FS.parquet scripts/cert_dropout.py --dataset d2v --rate sd50
echo "ALL DONE G (no-watch bench) $(date -Is)"
