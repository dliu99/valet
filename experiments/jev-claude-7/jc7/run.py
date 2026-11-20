"""Run episodes and summarise them.

  uv run python -m jc7.run --seeds 0 1 2 3 4 --name v1 [--jobs 5] [--effort medium] [--lessons runs/v1/lessons.md]

With --lessons, episodes run one after another so each can read the lessons of the ones before."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from valet_sim.scoreboard import Episode, record

from .episode import Config, Run

HERE = Path(__file__).resolve().parents[1]
SCOREBOARD = HERE / "results" / "episodes.jsonl"


def to_scoreboard(r: dict, config: str) -> Episode:
    """Claude is the slow model (plans, replans, lessons); Jev (or the local model) answers every frame."""
    return Episode(experiment="jev-claude-7", config=config, seed=r["seed"], placed=bool(r.get("placed")),
                   lifted=bool(r.get("lifted")), seconds=r["seconds"], slow_calls=r["claude"]["calls"],
                   fast_calls=r["jev"]["calls"], usd=round(r["claude"]["usd"] + r["jev"]["usd"], 5),
                   calls={"claude": r["claude"]["calls"], r.get("fast_model", "jev"): r["jev"]["calls"]})


def one(args: tuple[int, Config, str, str]) -> dict:
    seed, cfg, out, config = args
    done = Path(out) / f"seed-{seed:02d}" / "result.json"
    if done.exists():  # a finished episode is kept, so an interrupted run can resume
        return json.loads(done.read_text())
    r = Run(seed, cfg, Path(out) / f"seed-{seed:02d}").run()
    record(SCOREBOARD, to_scoreboard(r, config))
    status = "PLACED" if r.get("placed") else "lifted" if r.get("lifted") else "failed"
    print(f"seed {seed:2}: {status:6} off {r.get('off_target_m')} tilt {r.get('tilt_deg')} released {r.get('released')} "
          f"self {r.get("self_check", {}).get('placed_self')} replans {r.get('replans')} {r['seconds']}s "
          f"claude ${r['claude']['usd']} jev ${r['jev']['usd']} {r.get('error', '')}", flush=True)
    return r


def summarise(results: list[dict]) -> dict:
    n = len(results)
    return {
        "episodes": n,
        "placed": sum(bool(r.get("placed")) for r in results),
        "lifted": sum(bool(r.get("lifted")) for r in results),
        "self_placed": sum(bool(r.get("self_check", {}).get("placed_self")) for r in results),
        "self_agrees": sum(bool(r.get("self_check", {}).get("placed_self")) == bool(r.get("placed")) for r in results),
        "median_seconds": statistics.median(r["seconds"] for r in results),
        "claude_calls_per_ep": round(sum(r["claude"]["calls"] for r in results) / n, 2),
        "claude_usd_per_ep": round(sum(r["claude"]["usd"] for r in results) / n, 4),
        "jev_calls_per_ep": round(sum(r["jev"]["calls"] for r in results) / n, 2),
        "jev_usd_per_ep": round(sum(r["jev"]["usd"] for r in results) / n, 5),
        "replans_per_ep": round(sum(r.get("replans", 0) for r in results) / n, 2),
        "errors": [r["seed"] for r in results if r.get("error")],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m jc7.run")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    p.add_argument("--name", required=True)
    p.add_argument("--jobs", type=int, default=5)
    p.add_argument("--effort", default="medium")
    p.add_argument("--lessons", type=Path, default=None)
    p.add_argument("--model", default="claude-opus-5-5", help="the Claude model that plans")
    p.add_argument("--fast", default="jev", choices=["jev", "h2o"], help="the per-frame model: Jev, or H2O-Lightning-4B served locally")
    p.add_argument("--config", default=None, help="the config label on the scoreboard (default: the run name)")
    p.add_argument("--max-usd", type=float, default=None, help="stop starting episodes once Claude spend reaches this")
    a = p.parse_args(argv)
    out = HERE / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config(effort=a.effort, lessons=a.lessons, model=a.model, fast=a.fast)
    t0 = time.time()
    jobs = [(s, cfg, str(out), a.config or a.name) for s in a.seeds]
    if a.lessons is not None or a.jobs <= 1:
        results = []
        for j in jobs:
            if a.max_usd is not None and sum(r["claude"]["usd"] for r in results) >= a.max_usd:
                print(f"stopping: Claude spend reached ${a.max_usd}", flush=True)
                break
            results.append(one(j))
    else:
        with ProcessPoolExecutor(max_workers=a.jobs) as pool:
            results = list(pool.map(one, jobs))
    summary = {"name": a.name, "seeds": a.seeds, "model": a.model, "fast": a.fast, "effort": a.effort, "lessons": str(a.lessons) if a.lessons else None,
               **summarise(results), "wall_seconds": round(time.time() - t0)}
    (out / "summary.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2, default=str))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
