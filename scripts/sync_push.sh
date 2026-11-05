#!/usr/bin/env bash
# Every 15 min (cron): copy run outputs + logs (no weights) to the outputs repo and push;
# then commit results/ and docs/ in the code repo and push. Never commits code.
exec 9>/tmp/trustmi_sync.lock
flock -n 9 || exit 0
P=/home/deepinder/Research_Projects/TrustMI-HAR
O=/home/deepinder/TrustMI-HAR-outputs
push() { for try in 1 2 3; do git push -q origin HEAD && return 0; sleep 60; done; return 1; }

cd "$O" || exit 1
rsync -a --exclude='*.pt' --exclude='*.tmp' "$P/runs" "$P/logs" "$O/"
git add -A
git diff --cached --quiet || git commit -q -m "sync: run outputs $(date '+%Y-%m-%d %H:%M')"
push && echo "$(date '+%F %T') outputs pushed" || echo "$(date '+%F %T') outputs PUSH FAILED"

cd "$P" || exit 1
git add -- results docs
git diff --cached --quiet -- results docs || git commit -q -m "sync: results/docs $(date '+%Y-%m-%d %H:%M')" -- results docs
push && echo "$(date '+%F %T') code pushed" || echo "$(date '+%F %T') code PUSH FAILED"
