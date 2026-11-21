"""The closed-loop probe: `uv run python run_servo.py --seeds 0-19 --out results/servo.jsonl`."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from apriltag_exp import servo
from run_localize import seeds, todo


def one(job):
    seed, kind, ref = job
    try:
        return servo.run(seed, kind, ref)
    except Exception as e:
        return {"seed": seed, "kind": kind, "reference": ref, "failed": f"{type(e).__name__}: {e}"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=seeds, default=seeds("0-19"))
    p.add_argument("--kinds", nargs="*", default=["none", "offsets", "sag", "mount", "all"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("results/servo.jsonl"))
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    jobs = todo(a.out, [(s, k, r) for k in a.kinds for r in ("nominal", "taught") for s in a.seeds],
                lambda r: (r["seed"], r["kind"], r["reference"]))
    with a.out.open("a") as f, ProcessPoolExecutor(a.workers) as pool:
        for r in pool.map(one, jobs):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(r["kind"], r["reference"], r["seed"], r.get("failed", round(r.get("reach_mm", -1), 1)), flush=True)


if __name__ == "__main__":
    main()
