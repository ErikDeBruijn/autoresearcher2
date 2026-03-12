#!/bin/bash
# Watch evidence_run.log for the bayesian summary line.
# Kill the process BEFORE autoresearch starts.
# The log shows "--- Approach: AUTORESEARCH" when it transitions.

LOG="/Users/erik/github.com/erikdebruijn/autoresearcher2/artifacts/evidence_run.log"
PID=45609

echo "$(date): Watching for bayesian→autoresearch transition..."
echo "$(date): Will kill PID $PID when autoresearch is about to start."

# tail -f and grep for the autoresearch start marker
tail -n 0 -f "$LOG" | while read -r line; do
    if echo "$line" | grep -q "AUTORESEARCH BASELINE"; then
        echo "$(date): DETECTED autoresearch start. Killing PID $PID..."
        kill $PID 2>/dev/null
        echo "$(date): Process killed. Random + bayesian data preserved."
        echo "$(date): Run autoresearch + full with fixed code next."
        exit 0
    fi
    # Also log bayesian progress
    if echo "$line" | grep -q "bayesian:"; then
        echo "$(date): BAYESIAN SUMMARY: $line"
    fi
done
