#!/usr/bin/env bash
# Follow-up plan (docs/plan_followup_2026-11-12.md), Part A check: the four test-bench scripts gained a fifth
# condition set, --conditions imuoff. Every existing path must still give EXACTLY the stored results:
#   1. scripts/check_lies_unchanged_journal.sh  (lie bench of d1, d2, d1v2, d2v)
#   2. WearGait-PD (d3), fold 0 seed 0: lies, faults and real dropouts through cert_experts, cert_mi (own),
#      cert_evidential and cert_dropout (sd50), each compared row by row with the stored all-folds file.
# Overwrites only fold-0 scratch files (*_f0_s0*), never an all-folds file. About 10 minutes.
set -u
cd "$(dirname "$0")/.."
source trustmi-env/bin/activate
CE=results/certificate
fail=0
cmp () { echo "--- $2"; python scripts/check_identical.py "$CE/$1" "$CE/$2" --fold 0 "${@:3}" || fail=1; }
run () { python "$@" --folds 0 --seeds 0 > /dev/null || { echo "ERROR running $*"; fail=1; }; }

echo "=== 1. lie bench of the other datasets"
bash scripts/check_lies_unchanged_journal.sh | tee /tmp/check_lies_unchanged.out
grep -q "^ALL IDENTICAL: lie results unchanged by the fault option" /tmp/check_lies_unchanged.out || fail=1

echo; echo "=== 2. WearGait-PD (d3), fold 0 seed 0: lies, faults, real dropouts"
X=b4w_noarms,b4w_notrunk,b4w_nolegs
for c in lies faults realdrop; do
  if [ $c = lies ]; then S=""; else S="_$c"; fi
  echo; echo "### $c"
  run scripts/cert_experts.py --dataset d3 --conditions $c --extra-models $X
  cmp experts_d3_all_s0-1-2$S.parquet experts_d3_f0_s0$S.parquet
  run scripts/cert_mi.py --dataset d3 --version own --conditions $c
  cmp mi_d3_own_all_s0-1-2$S.parquet mi_d3_own_f0_s0$S.parquet
  run scripts/cert_evidential.py --dataset d3 --conditions $c
  cmp evidential_d3_all_s0-1-2$S.parquet evidential_d3_f0_s0$S.parquet --by method
  run scripts/cert_dropout.py --dataset d3 --rate sd50 --conditions $c
  cmp dropout_d3_sd50_all_s0-1-2$S.parquet dropout_d3_sd50_f0_s0$S.parquet
done
echo
[ $fail = 0 ] && echo "ALL IDENTICAL: existing results unchanged by the imuoff option" \
              || echo "AT LEAST ONE DIFFERENCE OR ERROR: stop and paste this"
