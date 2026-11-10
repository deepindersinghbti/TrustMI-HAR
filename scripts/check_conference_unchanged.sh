#!/usr/bin/env bash
# After editing a scoring script: re-run fold 0 of the CONFERENCE datasets (d1, d2) and compare every row with
# the stored all-folds files. Each comparison must print IDENTICAL, so the conference numbers stay reproducible.
# Takes about 3 minutes. Overwrites only the fold-0 scratch files (*_f0_*), never the all-folds files.
set -u
cd "$(dirname "$0")/.."
source trustmi-env/bin/activate
CE=results/certificate
fail=0
cmp () { echo "--- $2"; python scripts/check_identical.py "$CE/$1" "$CE/$2" --fold 0 || fail=1; }

python scripts/cert_experts.py --dataset d1 --folds 0 --attack imu,skel,video --extra-models b4_noskel,b4_novideo > /dev/null || fail=1
cmp experts_d1_all_s0-1-2_att-imu-skel-video.parquet experts_d1_f0_s0-1-2_att-imu-skel-video.parquet
python scripts/cert_experts.py --dataset d2 --folds 0 > /dev/null || fail=1
cmp experts_d2_all_s0-1-2.parquet experts_d2_f0_s0-1-2.parquet
python scripts/cert_mi.py --dataset d1 --version own --folds 0 --attack imu,skel,video > /dev/null || fail=1
cmp mi_d1_own_all_s0-1-2_att-imu-skel-video.parquet mi_d1_own_f0_s0-1-2_att-imu-skel-video.parquet
python scripts/cert_mi.py --dataset d2 --version own --folds 0 > /dev/null || fail=1
cmp mi_d2_own_all_s0-1-2.parquet mi_d2_own_f0_s0-1-2.parquet
python scripts/cert_signals.py --dataset d2 --folds 0 > /dev/null || fail=1
cmp signals_d2_all_s0-1-2.parquet signals_d2_f0_s0-1-2.parquet
[ $fail = 0 ] && echo "ALL FIVE IDENTICAL: conference results unchanged" || echo "AT LEAST ONE DIFFERENCE OR ERROR: stop and paste this"
