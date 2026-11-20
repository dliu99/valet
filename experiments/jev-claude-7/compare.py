"""One line per run: uv run python compare.py runs/v1-plan runs/v2-lessons ..."""
import json, sys
print(f"{'run':16} {'seeds':>5} {'placed':>6} {'lifted':>6} {'self ok':>7} {'med s':>6} {'claude':>7} {'$claude':>8} {'jev':>5} {'$jev':>8} {'replans':>7}")
for d in sys.argv[1:]:
    s = json.load(open(f"{d}/summary.json"))["summary"]
    print(f"{s['name']:16} {s['episodes']:>5} {s['placed']:>6} {s['lifted']:>6} {s['self_agrees']:>7} {s['median_seconds']:>6.0f} "
          f"{s['claude_calls_per_ep']:>7} {s['claude_usd_per_ep']:>8.3f} {s['jev_calls_per_ep']:>5} {s['jev_usd_per_ep']:>8.5f} {s['replans_per_ep']:>7}")
