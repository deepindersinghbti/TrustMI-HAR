#!/usr/bin/env bash
# Part C.3 of docs/plan_followup_2026-11-12.md: reproducibility check from a fresh clone.
#   bash scripts/check_reproducible.sh            (about 30-60 min; run inside tmux)
# 1. Clones this repository (committed state) to $CLONE and checks that its scorer code, src, configs and splits are
#    exactly those of the tag journal-results-frozen (only the three Part C scripts and docs/paper may be newer).
# 2. Copies the frozen stored model outputs (results/certificate/*.parquet) into the clone, checks them against
#    results/FROZEN_MANIFEST.sha256, and DELETES in the clone every file the scorers write, so each must be re-made.
# 3. Re-runs every journal scorer in dependency order, with the same environment (trustmi-env), then the table script.
# 4. Compares every re-made table, row file and journal table with the frozen one (PASS/FAIL), and lists any line of
#    the re-run scorer logs that differs from the frozen log (notes: the 10-11 Nov logs predate WearGait-PD rows).
# Reads the project; writes only inside $CLONE. Log: $CLONE/repro.log
set -u
P="$(cd "$(dirname "$0")/.." && pwd)"
CLONE=${CLONE:-$HOME/trustmi_repro_check}
TAG=journal-results-frozen
if [ -e "$CLONE" ]; then echo "$CLONE exists: remove it first (rm -rf $CLONE)"; exit 1; fi
if [ -n "$(git -C "$P" status --porcelain --untracked-files=no)" ]; then echo "uncommitted changes: commit first"; exit 1; fi
git clone -q "$P" "$CLONE" || exit 1
exec > >(tee "$CLONE/repro.log") 2>&1
cd "$CLONE"
echo "START $(date -Is) | clone of $P at $(git rev-parse --short HEAD) | tag $TAG = commit $(git rev-parse --short "$TAG^{commit}")"

echo; echo "== 1. code at the tag"
CHANGED=$(git diff --name-only "$TAG" HEAD -- . ':!docs' ':!paper' ':!scripts/make_journal_tables.py' \
          ':!scripts/compare_outputs.py' ':!scripts/check_reproducible.sh' ':!results/FROZEN_MANIFEST.sha256' \
          ':!scripts/freeze_manifest.py')
if [ -n "$CHANGED" ]; then echo "STOP: files changed since the tag:"; echo "$CHANGED"; exit 1; fi
echo "scorer code, src, configs, splits and result tables identical to the tag"

echo; echo "== 2. inputs"
ln -s "$P/trustmi-env" trustmi-env
ln -s "$P/data" data
ln -s "$P/runs" runs
ln -s "$P/results/raw" results/raw     # MI training records (mi_bound_summary, T12)
mkdir -p logs
cp "$P"/logs/*.log logs/        # the frozen scorer logs (the table script checks its numbers against them)
cp --reflink=auto "$P"/results/certificate/*.parquet results/certificate/ || exit 1
OUT_CSV="mi_d1v2_per_subject.csv mi_d2v_per_subject.csv mi_d3_per_subject.csv
 mi_d1v2_on_tmc_per_subject.csv mi_d1v2_on_ecml_per_subject.csv mi_d2v_on_tmc_s3_per_subject.csv
 mi_d2v_on_ecml_s3_per_subject.csv mi_d3_on_tmc_w_per_subject.csv mi_d3_on_ecml_w_per_subject.csv
 mi_d1v2_per_sensor.csv mi_d2v_per_sensor.csv mi_d3_per_sensor.csv
 mi_d1v2_naming_by_sensor.csv mi_d2v_naming_by_sensor.csv mi_d3_naming_by_sensor.csv mi_bound_summary.csv
 faults_d1v2_per_subject.csv faults_d2v_per_subject.csv faults_d3_per_subject.csv
 dropout_d1v2_per_subject.csv dropout_d2v_per_subject.csv dropout_d3_per_subject.csv
 v1_d1v2_auroc_per_subject.csv v1_d1v2_auroc_summary.csv v1_d1v2_per_subject.csv
 v1_d2v_auroc_per_subject.csv v1_d2v_auroc_summary.csv v1_d2v_per_subject.csv
 nowatch_d2v_per_subject.csv realdrop_d3_per_fold.csv
 imuoff_d3_per_fold.csv imuoff_d3_answers_per_fold.csv imuoff_d3_predictions.csv
 posthoc_per_unit.csv posthoc_imuoff_per_fold.csv posthoc_tau.csv"
OUT_PQ="nowatch_d2v_rows.parquet realdrop_d3_rows.parquet imuoff_d3_rev3_rows.parquet"
# the frozen originals stay in $P; the clone keeps only the INPUTS, so every output has to be made again
source trustmi-env/bin/activate
python - "$P" <<'PY' || { echo "STOP: copied model outputs differ from the manifest"; exit 1; }
import hashlib, sys
from pathlib import Path
ref = {}
for l in (Path(sys.argv[1]) / "results/FROZEN_MANIFEST.sha256").read_text().splitlines():
    if not l.startswith("#") and l.strip():
        h, n, rel = l.split("  ", 2); ref[rel] = h
pq = sorted(Path("results/certificate").glob("*.parquet"))
bad = [p for p in pq if hashlib.sha256(p.read_bytes()).hexdigest() != ref.get(p.as_posix())]
print(f"copied model outputs: {len(pq)} parquet files, {len(pq) - len(bad)} match the manifest")
sys.exit(1 if bad else 0)
PY
for f in $OUT_CSV $OUT_PQ; do rm -f "results/certificate/$f"; done
echo "removed $(echo $OUT_CSV $OUT_PQ | wc -w) scorer outputs in the clone (they must be re-made)"

echo; echo "== 3. re-run every scorer (same order as the dependencies)"
run () { local log=$1; shift; echo "--- $(date +%T) $*"; python "$@" > "logs/$log" 2>&1 || { echo "FAILED: $* (see logs/$log)"; exit 1; }; }
for d in d1v2 d2v d3; do run "rerun_mi_$d.log" scripts/cert_mi_score.py --dataset $d; done
for d in d1v2 d2v d3; do run "rerun_fault_$d.log" scripts/cert_fault_score.py --dataset $d; done
for d in d1v2 d2v d3; do run "rerun_dropout_$d.log" scripts/cert_dropout_score.py --dataset $d; done
for d in d1v2 d2v; do run "rerun_v1_$d.log" scripts/cert_v1_score.py --dataset $d; done
run rerun_nowatch_d2v.log scripts/cert_nowatch_score.py
run rerun_realdrop_d3.log scripts/cert_realdrop_score.py
run rerun_imuoff_d3.log scripts/cert_imuoff_score.py
run rerun_posthoc.log scripts/cert_posthoc_score.py
run rerun_tables.log scripts/make_journal_tables.py --out "$CLONE/paper/journal"

echo; echo "== 4. compare with the frozen outputs"
FILES=""; for f in $OUT_CSV $OUT_PQ; do FILES="$FILES results/certificate/$f"; done
for f in "$P"/paper/journal/tables/*.csv; do FILES="$FILES paper/journal/tables/$(basename "$f")"; done
FILES="$FILES paper/journal/checks.txt"
python scripts/compare_outputs.py --frozen "$P" --rerun "$CLONE" --files $FILES --logs \
  logs/cert_mi_score_d1v2.log:logs/rerun_mi_d1v2.log logs/cert_mi_score_d2v.log:logs/rerun_mi_d2v.log \
  logs/score_mi_d3.log:logs/rerun_mi_d3.log \
  logs/cert_fault_score_d1v2.log:logs/rerun_fault_d1v2.log logs/cert_fault_score_d2v.log:logs/rerun_fault_d2v.log \
  logs/score_fault_d3.log:logs/rerun_fault_d3.log logs/score_dropout_d3.log:logs/rerun_dropout_d3.log \
  logs/cert_v1_score_d1v2.log:logs/rerun_v1_d1v2.log logs/cert_v1_score_d2v.log:logs/rerun_v1_d2v.log \
  logs/cert_nowatch_score_d2v.log:logs/rerun_nowatch_d2v.log logs/score_realdrop_d3.log:logs/rerun_realdrop_d3.log \
  logs/score_imuoff_d3.log:logs/rerun_imuoff_d3.log logs/score_posthoc.log:logs/rerun_posthoc.log
RC=$?
echo; echo "END $(date -Is) | $([ $RC -eq 0 ] && echo 'REPRODUCED: every output identical (or within 1e-12)' || echo 'DIFFERENCES FOUND: see above')"
exit $RC
