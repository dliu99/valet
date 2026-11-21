"""The grid check for grid-drawing drivers: `uv run python run_grid.py --seeds 0-19 --out results/grid.jsonl`."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from apriltag_exp import grid
from run_localize import seeds, todo


def one(job):
    seed, kind = job
    try:
        return grid.run(seed, kind)
    except Exception as e:
        return {"seed": seed, "kind": kind, "failed": f"{type(e).__name__}: {e}"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=seeds, default=seeds("0-19"))
    p.add_argument("--kinds", nargs="*", default=["offsets", "all"])
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", type=Path, default=Path("results/grid.jsonl"))
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    jobs = todo(a.out, [(s, k) for k in a.kinds for s in a.seeds], lambda r: (r["seed"], r["kind"]))
    with a.out.open("a") as f, ProcessPoolExecutor(a.workers) as pool:
        for r in pool.map(one, jobs):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(r["kind"], r["seed"], r.get("failed", "ok"), flush=True)


if __name__ == "__main__":
    main()
