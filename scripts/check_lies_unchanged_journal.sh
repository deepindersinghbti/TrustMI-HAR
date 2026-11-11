#!/usr/bin/env bash
# Fault plan, step 3 check: the four test-bench scripts gained a --conditions option (default "lies").
# With the default they must give EXACTLY the stored results. This runs:
#   1. scripts/check_conference_unchanged.sh   (conference datasets d1, d2; experts, MI, signals)
#   2. fold 0, seed 0 of the journal datasets d1v2 and d2v through cert_experts, cert_mi (own),
#      cert_evidential and cert_dropout (sd50), each compared row by row with the stored all-folds file.
# Overwrites only the fold-0 scratch files (*_f0_s0*), never an all-folds file. About 5 minutes.
set -u
cd "$(dirname "$0")/.."
source trustmi-env/bin/activate
CE=results/certificate
fail=0
cmp () { echo "--- $2"; python scripts/check_identical.py "$CE/$1" "$CE/$2" --fold 0 "${@:3}" || fail=1; }
run () { python "$@" --folds 0 --seeds 0 > /dev/null || { echo "ERROR running $*"; fail=1; }; }

echo "=== 1. conference datasets"
bash scripts/check_conference_unchanged.sh || fail=1

echo; echo "=== 2. journal datasets, fold 0 seed 0"
run scripts/cert_experts.py --dataset d1v2 --extra-models b4_noimu,b4_noskel,b4_novideo
cmp experts_d1v2_all_s0-1-2.parquet experts_d1v2_f0_s0.parquet
run scripts/cert_experts.py --dataset d2v --extra-models b4s_nowatch,b4s_nophone,b4s
cmp experts_d2v_all_s0-1-2.parquet experts_d2v_f0_s0.parquet
for d in d1v2 d2v; do
  run scripts/cert_mi.py --dataset $d --version own
  cmp mi_${d}_own_all_s0-1-2.parquet mi_${d}_own_f0_s0.parquet
  run scripts/cert_evidential.py --dataset $d
  cmp evidential_${d}_all_s0-1-2.parquet evidential_${d}_f0_s0.parquet --by method
  run scripts/cert_dropout.py --dataset $d --rate sd50
  cmp dropout_${d}_sd50_all_s0-1-2.parquet dropout_${d}_sd50_f0_s0.parquet
done
echo
[ $fail = 0 ] && echo "ALL IDENTICAL: lie results unchanged by the fault option" \
              || echo "AT LEAST ONE DIFFERENCE OR ERROR: stop and paste this"
