"""Part 2 and 3 over many scenes: `uv run python run_localize.py --seeds 0-19 --out results/localize.jsonl`."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from apriltag_exp import errors, localize


def seeds(text: str) -> list[int]:
    a, _, b = text.partition("-")
    return list(range(int(a), int(b or a) + 1))


def todo(path: Path, jobs: list[tuple], key) -> list[tuple]:
    """The jobs whose results are not in `path` yet, so an interrupted run picks up where it stopped."""
    done = {key(json.loads(line)) for line in path.read_text().splitlines() if line.strip()} if path.exists() else set()
    return [j for j in jobs if j not in done]


def one(job):
    seed, kind = job
    t0 = time.time()
    try:
        r = localize.run(seed, kind)
    except Exception as e:  # keep the sweep going; the failure is a result too
        r = {"seed": seed, "kind": kind, "failed": f"{type(e).__name__}: {e}"}
    r["wall_s"] = round(time.time() - t0, 1)
    return r


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=seeds, default=seeds("0-19"))
    p.add_argument("--kinds", nargs="*", default=list(errors.KINDS))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("results/localize.jsonl"))
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    jobs = todo(a.out, [(s, k) for k in a.kinds for s in a.seeds], lambda r: (r["seed"], r["kind"]))
    with a.out.open("a") as f, ProcessPoolExecutor(a.workers) as pool:
        for r in pool.map(one, jobs):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(r["kind"], r["seed"], r.get("failed", "ok"), r["wall_s"], "s", flush=True)


if __name__ == "__main__":
    main()
