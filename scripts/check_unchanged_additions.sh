#!/usr/bin/env bash
# docs/plan_additions_2026-11-14.md, Checks 1 and 3: the four model-running scripts gained two condition sets
# (nowatch2, imuoff2). Every EXISTING path must still give exactly the stored results.
# Runs in a throwaway clone ($CLONE), so no frozen file in the project is written (the fold-0 scratch files that
# such checks overwrite are part of results/FROZEN_MANIFEST.sha256). About 15 minutes; run inside tmux.
#   1. unit tests: tests/test_additions.py (incl. the MMAct data test), test_imuoff.py, test_nowatch.py
#   2. scripts/check_lies_unchanged_journal.sh   (conference datasets; lie bench of d1v2 and d2v)
#   3. WearGait-PD (d3), fold 0 seed 0: lies, faults, real dropouts, IMU-off   (4 scripts each)
#   4. MMAct (d2v), fold 19 seed 0: the no-watch bench                         (4 scripts)
# each compared row by row with the stored all-folds file (scripts/check_identical.py). Log: $CLONE/check.log
set -u
P="$(cd "$(dirname "$0")/.." && pwd)"
CLONE=${CLONE:-$HOME/trustmi_check_k}
if [ -e "$CLONE" ]; then echo "$CLONE exists: remove it first (rm -rf $CLONE)"; exit 1; fi
if [ -n "$(git -C "$P" status --porcelain --untracked-files=no)" ]; then echo "uncommitted changes: commit first"; exit 1; fi
git clone -q "$P" "$CLONE" || exit 1
exec > >(tee "$CLONE/check.log") 2>&1
cd "$CLONE"
echo "START $(date -Is) | clone of $P at $(git rev-parse --short HEAD)"
ln -s "$P/trustmi-env" trustmi-env; ln -s "$P/data" data; ln -s "$P/runs" runs
ln -s "$P/results/raw" results/raw
mkdir -p logs
cp --reflink=auto "$P"/results/certificate/*.parquet results/certificate/ || exit 1
source trustmi-env/bin/activate
CE=results/certificate
fail=0
cmpf () { echo "--- $2"; python scripts/check_identical.py "$CE/$1" "$CE/$2" --fold "$3" "${@:4}" || fail=1; }
run () { local f=$1; shift; python "$@" --folds "$f" --seeds 0 > /dev/null || { echo "ERROR running $*"; fail=1; }; }

echo; echo "=== 1. unit tests"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_additions.py tests/test_imuoff.py tests/test_nowatch.py -q -rs \
  > logs/check_k_pytest.out 2>&1; rc=$?
tail -4 logs/check_k_pytest.out
[ $rc = 0 ] || { echo "UNIT TESTS FAILED (see $CLONE/logs/check_k_pytest.out)"; fail=1; }

echo; echo "=== 2. lie bench of the other datasets"
bash scripts/check_lies_unchanged_journal.sh | tee /tmp/check_lies_unchanged_k.out
grep -q "^ALL IDENTICAL: lie results unchanged by the fault option" /tmp/check_lies_unchanged_k.out || fail=1

echo; echo "=== 3. WearGait-PD (d3), fold 0 seed 0"
X=b4w_noarms,b4w_notrunk,b4w_nolegs
for c in lies faults realdrop imuoff; do
  if [ $c = lies ]; then S=""; else S="_$c"; fi
  echo; echo "### $c"
  run 0 scripts/cert_experts.py --dataset d3 --conditions $c --extra-models $X
  cmpf experts_d3_all_s0-1-2$S.parquet experts_d3_f0_s0$S.parquet 0
  run 0 scripts/cert_mi.py --dataset d3 --version own --conditions $c
  cmpf mi_d3_own_all_s0-1-2$S.parquet mi_d3_own_f0_s0$S.parquet 0
  run 0 scripts/cert_evidential.py --dataset d3 --conditions $c
  cmpf evidential_d3_all_s0-1-2$S.parquet evidential_d3_f0_s0$S.parquet 0 --by method
  run 0 scripts/cert_dropout.py --dataset d3 --rate sd50 --conditions $c
  cmpf dropout_d3_sd50_all_s0-1-2$S.parquet dropout_d3_sd50_f0_s0$S.parquet 0
done

echo; echo "=== 4. MMAct (d2v), fold 19 seed 0: no-watch bench"
X=b4s_nowatch,b4s_nophone,b4s
run 19 scripts/cert_experts.py --dataset d2v --conditions nowatch --extra-models $X
cmpf experts_d2v_all_s0-1-2_nowatch.parquet experts_d2v_f19_s0_nowatch.parquet 19
run 19 scripts/cert_mi.py --dataset d2v --version own --conditions nowatch
cmpf mi_d2v_own_all_s0-1-2_nowatch.parquet mi_d2v_own_f19_s0_nowatch.parquet 19
run 19 scripts/cert_evidential.py --dataset d2v --conditions nowatch
cmpf evidential_d2v_all_s0-1-2_nowatch.parquet evidential_d2v_f19_s0_nowatch.parquet 19 --by method
run 19 scripts/cert_dropout.py --dataset d2v --rate sd50 --conditions nowatch
cmpf dropout_d2v_sd50_all_s0-1-2_nowatch.parquet dropout_d2v_sd50_f19_s0_nowatch.parquet 19
echo
[ $fail = 0 ] && echo "END $(date -Is) | ALL IDENTICAL: existing results unchanged by the two new condition sets" \
              || echo "END $(date -Is) | AT LEAST ONE DIFFERENCE OR ERROR: stop and paste this"
