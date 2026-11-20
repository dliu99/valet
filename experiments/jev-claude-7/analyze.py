"""Grading-side analysis of a run, after the fact: how far the first measurements were from the true layout, and
how each episode ended. Reads the episode logs and `task.episode(seed)`; nothing here feeds back into the policy.

  uv run python analyze.py runs/full-opus [runs/...]"""

import json
import statistics
import sys
from pathlib import Path

import numpy as np

from valet_sim.task import episode


def first_estimates(events: Path) -> dict:
    out = {}
    for line in events.read_text().splitlines():
        e = json.loads(line)
        if e["kind"] == "estimate" and e["what"] not in out:
            out[e["what"]] = e
        if e["kind"] in ("plan", "error"):
            break
    return out


def outcome(r: dict) -> str:
    if r.get("placed"):
        return "placed"
    if r.get("tilt_deg", 0) > 15:
        return "tipped"
    if not r.get("released", True):
        return "still held"
    if not r.get("lifted"):
        return "not lifted"
    return f"off by {r['off_target_m'] * 100:.1f} cm"


for run in map(Path, sys.argv[1:]):
    box_err, sq_err, h_err, rows = [], [], [], []
    for d in sorted(run.glob("seed-*")):
        if not (d / "result.json").exists():
            continue
        r = json.loads((d / "result.json").read_text())
        ep = episode(r["seed"])
        est = first_estimates(d / "events.jsonl")
        if "box" in est:
            box_err.append(np.hypot(est["box"]["x_cm"] / 100 - ep.box.x, est["box"]["y_cm"] / 100 - ep.box.y) * 100)
            h_err.append(est["box"]["height_cm"] - ep.box.h * 100)
        if "square" in est:
            sq_err.append(np.hypot(est["square"]["x_cm"] / 100 - ep.target_xy[0], est["square"]["y_cm"] / 100 - ep.target_xy[1]) * 100)
        rows.append((r["seed"], outcome(r), r.get("replans"), r["claude"]["calls"], r["jev"]["calls"], r["seconds"]))
    print(f"== {run.name}: {len(rows)} episodes")
    if box_err:
        print(f"  first box estimate: median {statistics.median(box_err):.1f} cm, max {max(box_err):.1f}; "
              f"height error median {statistics.median(h_err):+.1f} cm (|max| {max(map(abs, h_err)):.1f})")
        print(f"  first square estimate: median {statistics.median(sq_err):.1f} cm, max {max(sq_err):.1f}")
    counts: dict[str, int] = {}
    for _, o, *_ in rows:
        key = "off target" if o.startswith("off by") else o
        counts[key] = counts.get(key, 0) + 1
    print("  outcomes:", counts)
    for row in rows:
        print("   seed %2d  %-16s replans %s  claude %s  jev %s  %.0fs" % row)
