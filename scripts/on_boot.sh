#!/usr/bin/env bash
# Run once at boot (cron @reboot): commit leftover outputs, then restart the journal queues in tmux.
# Finished queues only print their skip / DONE lines and exit, so restarting all of them is harmless.
sleep 120                                    # let the GPU driver and network come up
cd /home/deepinder/Research_Projects/TrustMI-HAR || exit 1
bash scripts/sync_push.sh                    # commits logs/results so the clean-tree check passes
for q in a b d; do
  tmux new -d -s queue${q^^} "cd /home/deepinder/Research_Projects/TrustMI-HAR && source trustmi-env/bin/activate && bash scripts/run_journal_queue_${q}.sh 2>&1 | tee -a logs/journal_queue_${q}.log"
done
