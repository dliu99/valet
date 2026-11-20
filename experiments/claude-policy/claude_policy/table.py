"""Tabulate runs for the report from their per-seed result.json files (a crashed seed re-run in a second directory
can be merged in with `label=dir1,dir2`).

    uv run python -m claude_policy.table w0=runs/2026...-w0-wrist w3=runs/...-w3-centred,runs/...-w3-centred-seed4-rerun
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

JEV_USD_PER_MTOK = 0.042


def load(dirs: list[str]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for d in dirs:
        for f in sorted(Path(d).glob("seed*/result.json")):
            r = json.loads(f.read_text())
            out[r["seed"]] = r
    return out


def row(label: str, results: dict[int, dict]) -> dict:
    rs = list(results.values())
    n = len(rs)
    mean = lambda k: sum(r.get(k, 0) for r in rs) / n  # noqa: E731
    return {
        "label": label, "n": n,
        "placed": sum(r["placed"] for r in rs), "lifted": sum(r["lifted"] for r in rs),
        "median_s": statistics.median(r["wall_s"] for r in rs),
        "median_motion_s": statistics.median(r.get("motion_s", 0) for r in rs),
        "claude_calls": mean("claude_calls"), "jev_calls": mean("jev_calls"),
        "in_tokens": mean("input_tokens") + mean("cache_write_tokens") + mean("cache_read_tokens"),
        "cache_write": mean("cache_write_tokens"), "out_tokens": mean("output_tokens"),
        "claude_usd": mean("claude_usd"), "jev_usd": mean("jev_input_tokens") * JEV_USD_PER_MTOK / 1e6,
        "outcomes": " ".join(f"{s}:{'P' if r['placed'] else 'L' if r['lifted'] else '-'}" for s, r in sorted(results.items())),
    }


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    print("| run | placed | lifted | median s | Claude calls | Jev calls | input tok | output tok | $ Claude | $ Jev | per seed |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for arg in argv:
        label, _, dirs = arg.partition("=")
        r = row(label, load(dirs.split(",")))
        print(f"| {r['label']} | {r['placed']}/{r['n']} | {r['lifted']}/{r['n']} | {r['median_s']:.0f} | {r['claude_calls']:.1f} | "
              f"{r['jev_calls']:.1f} | {r['in_tokens'] / 1000:.0f}k | {r['out_tokens'] / 1000:.1f}k | {r['claude_usd']:.2f} | "
              f"{r['jev_usd']:.5f} | {r['outcomes']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
