#!/bin/bash
# Team 6's program with local H2O perception, under errors, three variants. The H2O server is a multi-GB model:
#   python3 ~/.valet/slot.py --name team-9 --heavy -- ./run_aspire_h2o.sh /path/to/aspire
set -u -m
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
ASPIRE="$1"; shift
mkdir -p results
# the plumbing first, at no cost: one scene per variant with the true boxes; stop before the server if it crashes
SMOKE="$(mktemp -t aspire-smoke)"
uv run python run_aspire.py --aspire "$ASPIRE" --backend oracle --seeds 0 --out "$SMOKE" --workers 3 || exit 1
if grep -q crashed "$SMOKE"; then echo "smoke test crashed; not starting H2O" >&2; exit 1; fi
( cd ~/projects/h2o-lightning-mlx && exec ./serve.sh ) > results/h2o-server.log 2>&1 &
SERVER=$!
trap 'kill -TERM -- -$SERVER 2>/dev/null; sleep 3; kill -KILL -- -$SERVER 2>/dev/null' EXIT
for i in $(seq 1 120); do
  curl -s -m 2 http://127.0.0.1:8741/health | grep -q '"ok": true' && break
  sleep 5
done
uv run python run_aspire.py --aspire "$ASPIRE" --backend h2o --record "$@"
