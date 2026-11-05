#!/usr/bin/env bash
# Commit new run outputs, logs and notes (never code), then push. Run by cron every 15 min.
cd /home/deepinder/Research_Projects/TrustMI-HAR || exit 1
exec 9>/tmp/trustmi_sync.lock
flock -n 9 || exit 0                      # previous sync still running: skip this round
PATHS="results runs logs docs"
git add -- $PATHS
if ! git diff --cached --quiet -- $PATHS; then
  git commit -q -m "sync: run outputs $(date '+%Y-%m-%d %H:%M')" -- $PATHS
fi
for try in 1 2 3; do
  git push -q origin master && { echo "$(date '+%F %T') pushed"; exit 0; }
  sleep 60
done
echo "$(date '+%F %T') PUSH FAILED, will retry next round"
