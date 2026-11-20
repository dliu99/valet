"""Run a condition on benchmark seeds and write runs/<name>/: one folder per seed (events, frames, result) and
summary.json. Episodes run in parallel processes; each owns its sim. Seeds that already have a result in that
folder are kept, so a run can be extended (0-4, then 0-19) or resumed after a crash.

    uv run physical-agent --condition demo --seeds 0-4 --name opus-wrist-base-demo
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from .agent import Config, run_episode
from .demo import DEMO_SEED, find
from .scene import VIEWS
from .tool import FRAMES

HERE = Path(__file__).resolve().parent.parent
CONDITIONS = ("zero", "demo", "distractor")


def load_env() -> None:
    """KEY=VALUE lines from the nearest .env above this directory; variables already set win."""
    for d in [HERE, *HERE.parents]:
        f = d / ".env"
        if f.is_file():
            for line in f.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return


def seeds_arg(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def one(seed: int, condition: str, cfg: Config, out: Path, demo_seed: int = DEMO_SEED) -> dict:
    import anthropic

    load_env()
    client = anthropic.Anthropic(max_retries=6)
    demo = None if condition == "zero" else find("task" if condition == "demo" else "distractor", demo_seed)
    ep = run_episode(client, seed, cfg, demo, out / f"seed-{seed:02d}")
    row = {"seed": seed, "condition": condition, "model": cfg.model, "effort": cfg.effort, "views": cfg.views,
           "frame": cfg.frame, "demo_seed": demo.seed if demo else None, **ep.result, "turns": ep.turns, "ended": ep.ended,
           "wall_s": round(ep.wall_s, 1), "sim_s": round(ep.sim_s, 1), "usage": asdict(ep.usage),
           "tokens": ep.usage.total_tokens(), "dollars": round(ep.usage.dollars(cfg.model), 4)}
    (out / f"seed-{seed:02d}" / "result.json").write_text(json.dumps(row, indent=1))
    return row


def kept(out: Path, seeds: list[int], condition: str, cfg: Config) -> list[dict]:
    """Earlier results in this folder for these seeds, if they were run with the same settings."""
    rows = []
    for s in seeds:
        f = out / f"seed-{s:02d}" / "result.json"
        if f.is_file():
            r = json.loads(f.read_text())
            same = (r["condition"], r["model"], r["effort"], r.get("views", "both"), r.get("frame", "base")) == (
                condition, cfg.model, cfg.effort, cfg.views, cfg.frame)
            if not same:
                raise SystemExit(f"{out} has seed {s} from different settings; use another --name")
            rows.append(r)
    return rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    med = lambda k: round(statistics.median(r[k] for r in rows), 1) if rows else None
    return {"n": n, "placed": sum(r["placed"] for r in rows), "lifted": sum(r["lifted"] for r in rows),
            "median_turns": med("turns"), "median_wall_s": med("wall_s"), "median_sim_s": med("sim_s"),
            "median_tokens": med("tokens"), "mean_dollars": round(sum(r["dollars"] for r in rows) / max(n, 1), 3),
            "total_dollars": round(sum(r["dollars"] for r in rows), 2),
            "ended": {e: sum(r["ended"] == e for r in rows) for e in sorted({r["ended"] for r in rows})}}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="physical-agent", description=__doc__)
    p.add_argument("--condition", choices=CONDITIONS, default="zero")
    p.add_argument("--seeds", type=seeds_arg, default=seeds_arg("0-4"))
    p.add_argument("--model", default=Config.model)
    p.add_argument("--effort", default=Config.effort)
    p.add_argument("--turns", type=int, default=Config.turns)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--views", choices=VIEWS, default=Config.views, help="wrist only (the real arm), or scene too")
    p.add_argument("--frame", choices=FRAMES, default=Config.frame, help="absolute base-frame or gripper-frame targets")
    p.add_argument("--demo-seed", type=int, default=DEMO_SEED)
    p.add_argument("--max-dollars", type=float, default=20.0, help="stop starting episodes past this spend")
    p.add_argument("--name")
    a = p.parse_args(argv)
    cfg = Config(model=a.model, effort=a.effort, turns=a.turns, views=a.views, frame=a.frame)
    name = a.name or f"{time.strftime('%Y%m%d-%H%M%S')}-{a.model}-{a.condition}"
    out = HERE / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    rows = kept(out, a.seeds, a.condition, cfg)
    if rows:
        print(f"keeping seeds {[r['seed'] for r in rows]} from an earlier run", flush=True)
    spent = sum(r["dollars"] for r in rows)
    pending = [s for s in a.seeds if s not in {r["seed"] for r in rows}]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = {}
        while pending or futures:
            while pending and len(futures) < a.workers and spent < a.max_dollars:
                s = pending.pop(0)
                futures[pool.submit(one, s, a.condition, cfg, out, a.demo_seed)] = s
            if not futures:
                break
            done = next(as_completed(futures))
            seed = futures.pop(done)
            try:
                row = done.result()
            except Exception as e:  # an API or sim failure ends that episode; the rest go on
                print(f"seed {seed:2}: error {type(e).__name__}: {e}", flush=True)
                continue
            rows.append(row)
            spent += row["dollars"]
            verdict = "placed" if row["placed"] else "lifted" if row["lifted"] else "failed"
            print(f"seed {seed:2}: {verdict:6} turns {row['turns']:2} ({row['ended']}) {row['wall_s']:5.0f} s "
                  f"${row['dollars']:.3f}  off {row['off_target_m']}  [spent ${spent:.2f}]", flush=True)
    if pending:
        print(f"stopped at ${spent:.2f}: seeds {pending} not run", flush=True)
    rows.sort(key=lambda r: r["seed"])
    summary = {"condition": a.condition, "model": a.model, "effort": a.effort, "turn_budget": a.turns,
               "views": a.views, "frame": a.frame,
               "demo_seed": None if a.condition == "zero" else a.demo_seed,
               "seeds": a.seeds, "not_run": pending, **summarize(rows)}
    (out / "summary.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
