#!/bin/bash
# Start H2O-Lightning-4B, run the best config with it in place of Jev, then stop the server however this exits.
# The server is a multi-GB model, so run this under the shared heavy slot:
#   python3 ~/.valet/slot.py --name team-11 --heavy -- ./run_local.sh 0 1 2 3 4
set -u -m
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
mkdir -p runs
( cd ~/projects/h2o-lightning-mlx && exec ./serve.sh ) > runs/h2o-server.log 2>&1 &
SERVER=$!
trap 'kill -TERM -- -$SERVER 2>/dev/null; sleep 3; kill -KILL -- -$SERVER 2>/dev/null' EXIT
for i in $(seq 1 120); do
  curl -s -m 2 http://127.0.0.1:8741/health | grep -q '"ok": true' && break
  sleep 5
done
uv run python -m jev_skills.run --config final-local --decider local --seeds "$@"
