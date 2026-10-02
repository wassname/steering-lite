#!/usr/bin/env bash
# Budget cap (<$50): stop every bsbench container started after CUTOFF, so queued random-user
# seeds never run; running ones finish. Loops until the spawning `modal run` (pid $1) exits. PI/OpenAI
set -euo pipefail
pid=$1; cutoff=$2
while kill -0 "$pid" 2>/dev/null; do
  uv run --extra benchmark modal container list --json 2>/dev/null | python3 -c "
import json, sys
for c in json.load(sys.stdin):
    if c['app_name'] == 'steering-lite-bsbench-v3' and c['start_time'] > sys.argv[1]:
        print(c['container_id'], c['start_time'])
" "$cutoff" | while read -r id start; do
    echo "$(date +%T) stop $id started $start"
    uv run --extra benchmark modal container stop "$id" || echo "stop failed $id"
  done
  sleep 20
done
echo "$(date +%T) spawner $pid exited; watcher done"
