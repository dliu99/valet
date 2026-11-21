"""Part 4: `uv run python run_policy.py --seeds 0-19 --out results/policy.jsonl`. `--backfill` copies the results to
the shared scoreboard (results/episodes.jsonl) instead of running."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from apriltag_exp import policy
from run_localize import seeds, todo

CONDITIONS = (("none", "fk"), ("all", "fk"), ("all", "tag"), ("all", "cal"))
CONFIGS = {("none", "fk"): "own-policy/no-errors", ("all", "fk"): "own-policy/errors/no-tag",
           ("all", "tag"): "own-policy/errors/tag-runtime", ("all", "cal"): "own-policy/errors/tag-calibrated"}


def backfill(results: Path) -> int:
    """No model runs in this policy: every episode is 0 calls and $0."""
    from valet_sim.scoreboard import Episode, record

    rows = [json.loads(line) for line in results.read_text().splitlines() if line.strip()]
    for r in rows:
        record(results.parent / "episodes.jsonl", Episode(
            experiment="apriltag", config=CONFIGS[(r["kind"], r["method"])], seed=r["seed"], placed=bool(r["placed"]),
            lifted=bool(r["lifted"]), seconds=float(r.get("sim_s", 0.0)), slow_calls=0, fast_calls=0, usd=0.0))
    return len(rows)


def one(job):
    seed, kind, method = job
    t0 = time.time()
    try:
        r = policy.run(seed, kind, method)
    except Exception as e:
        r = {"seed": seed, "kind": kind, "method": method, "placed": False, "lifted": False, "note": f"{type(e).__name__}: {e}"}
    r["total_wall_s"] = round(time.time() - t0, 1)
    return r


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=seeds, default=seeds("0-9"))
    p.add_argument("--conditions", nargs="*", default=[f"{k}/{m}" for k, m in CONDITIONS])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("results/policy.jsonl"))
    p.add_argument("--backfill", action="store_true")
    a = p.parse_args()
    if a.backfill:
        print(backfill(a.out), "episodes recorded")
        return
    a.out.parent.mkdir(parents=True, exist_ok=True)
    jobs = todo(a.out, [(s, *c.split("/")) for c in a.conditions for s in a.seeds], lambda r: (r["seed"], r["kind"], r["method"]))
    with a.out.open("a") as f, ProcessPoolExecutor(a.workers) as pool:
        for r in pool.map(one, jobs):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(r["kind"], r["method"], r["seed"], "placed" if r["placed"] else "lifted" if r["lifted"] else "failed", r.get("note"), r["total_wall_s"], "s", flush=True)


if __name__ == "__main__":
    main()
