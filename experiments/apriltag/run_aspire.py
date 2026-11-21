"""Team 6's final program under this experiment's errors, with and without the tag.

  git archive d8e9c0f experiments/aspire | tar -x -C /tmp/aspire       # their code, read-only, at PR #8's commit
  uv run python run_aspire.py --aspire /tmp/aspire/experiments/aspire --backend h2o --seeds 0-19 --record

`--backend h2o` needs the local H2O server (see run_aspire_h2o.sh); `oracle` answers `localize` with the truth and is
for checking the plumbing only. `--record` appends benchmark episodes to results/episodes.jsonl (the scoreboard)."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from apriltag_exp import aspire_bridge
from run_localize import seeds, todo

RESULTS = Path(__file__).parent / "results"


def one(job):
    seed, variant, backend, src, kind = job
    code = (Path(src) / "programs" / "benchmark_final.py").read_text()
    try:
        return aspire_bridge.run_episode(code, seed, variant, backend, Path(src), kind)
    except Exception as e:
        return {"seed": seed, "variant": variant, "backend": backend, "kind": kind, "error": f"crashed: {type(e).__name__}: {e}",
                "score": {"placed": False, "lifted": False}}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aspire", type=Path, required=True)
    p.add_argument("--backend", default="h2o", choices=["h2o", "oracle", "jev", "claude"])
    p.add_argument("--variants", nargs="*", default=list(aspire_bridge.VARIANTS))
    p.add_argument("--kind", default="all")
    p.add_argument("--seeds", type=seeds, default=seeds("0-19"))
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", type=Path, default=RESULTS / "aspire.jsonl")
    p.add_argument("--record", action="store_true", help="append to results/episodes.jsonl for the scoreboard")
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    jobs = [(s, v, a.backend, str(a.aspire.resolve()), a.kind) for v in a.variants for s in a.seeds]
    jobs = todo(a.out, jobs, lambda r: (r["seed"], r["variant"], r["backend"], str(a.aspire.resolve()), r["kind"]))
    with a.out.open("a") as f, ProcessPoolExecutor(a.workers) as pool:
        for r in pool.map(one, jobs):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if a.record and r["backend"] != "oracle":
                from valet_sim.scoreboard import Episode, record

                n = r.get("localize_calls", 0)
                record(RESULTS / "episodes.jsonl", Episode(
                    experiment="apriltag", config=f"aspire-final/errors/{r['variant']}", seed=r["seed"],
                    placed=bool(r["score"]["placed"]), lifted=bool(r["score"]["lifted"]), seconds=float(r.get("sim_s", 0.0)),
                    slow_calls=n if r["backend"] == "claude" else 0, fast_calls=0 if r["backend"] == "claude" else n,
                    usd=0.0 if r["backend"] == "h2o" else float("nan"), calls={f"localize:{r['backend']}": n}))
            s = r["score"]
            print(r["variant"], r["seed"], "placed" if s["placed"] else "lifted" if s["lifted"] else "failed",
                  r.get("error"), r.get("wall_s"), "s", flush=True)


if __name__ == "__main__":
    main()
