"""Run Claude on benchmark seeds in simulated time and grade each episode with the sim's own score.

    uv run claude-policy --seeds 0 1 2 3 4 --tag baseline
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

from .agent import Config, run_episode

HERE = Path(__file__).resolve().parent.parent


def load_env() -> None:
    """KEY=VALUE lines from the nearest .env above this directory; existing variables win."""
    for d in [HERE, *HERE.parents]:
        f = d / ".env"
        if f.is_file():
            for line in f.read_text().splitlines():
                k, sep, v = line.strip().partition("=")
                if sep and not k.startswith("#"):
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return


def run_seed(seed: int, cfg: Config, out: Path) -> dict:
    import anthropic

    from valet_sim.sim import Sim, SimSpec, with_box
    from valet_sim.task import episode, score

    from .arm import Arm

    d = out / f"seed{seed:02d}"
    d.mkdir(parents=True, exist_ok=True)
    events = (d / "events.jsonl").open("w")
    n = [0]

    def record(e: dict) -> None:
        events.write(json.dumps(e, default=str) + "\n")
        events.flush()

    def save_frame(cam: str, jpeg: bytes) -> None:
        n[0] += 1
        (d / f"{n[0]:03d}-{cam}.jpg").write_bytes(jpeg)

    ep = episode(seed)
    sim = Sim(with_box(SimSpec(), ep.box, ep.target_xy))
    arm = Arm(sim, max_side=cfg.max_side, overlay=cfg.overlay)
    checker = None
    if cfg.jev_checks:
        from .jev import Jev

        checker = Jev()
    client = anthropic.Anthropic(max_retries=6)
    t0 = time.monotonic()
    try:
        res = run_episode(arm, client, cfg, record, save_frame, checker)
    except Exception as e:  # an API or sim failure ends the episode; it is graded as it stands
        res = {"end": f"error: {type(e).__name__}: {e}"}
    wall = time.monotonic() - t0
    sim.step(0.5)  # let the scene settle before grading
    result = {"seed": seed, **score(sim.truth()), "wall_s": round(wall, 1), **res}
    record({"type": "result", **result})
    events.close()
    (d / "result.json").write_text(json.dumps(result, indent=1))
    return result


def summarize(results: list[dict], cfg: Config) -> dict:
    n = len(results)
    med = lambda k: round(statistics.median(r.get(k, 0) for r in results), 1) if n else 0  # noqa: E731
    mean = lambda k: round(sum(r.get(k, 0) for r in results) / n, 4) if n else 0  # noqa: E731
    return {
        "config": asdict(cfg),
        "n": n,
        "placed": sum(r["placed"] for r in results),
        "lifted": sum(r["lifted"] for r in results),
        "median_wall_s": med("wall_s"),
        "median_claude_s": med("claude_s"),
        "median_motion_s": med("motion_s"),
        "claude_calls_per_ep": mean("claude_calls"),
        "jev_calls_per_ep": mean("jev_calls"),
        "input_tokens_per_ep": mean("input_tokens"),
        "cache_write_tokens_per_ep": mean("cache_write_tokens"),
        "cache_read_tokens_per_ep": mean("cache_read_tokens"),
        "output_tokens_per_ep": mean("output_tokens"),
        "jev_input_tokens_per_ep": mean("jev_input_tokens"),
        "claude_usd_per_ep": mean("claude_usd"),
        "claude_usd_total": round(sum(r.get("claude_usd", 0) for r in results), 3),
        "per_seed": {r["seed"]: ("placed" if r["placed"] else "lifted" if r["lifted"] else "failed") for r in results},
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="claude-policy", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    p.add_argument("--tag", default="run")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--model", default=Config.model)
    p.add_argument("--effort", default=Config.effort)
    p.add_argument("--max-side", type=int, default=Config.max_side)
    p.add_argument("--cams", nargs="+", default=list(Config.cams))
    p.add_argument("--jev", action="store_true", help="let move and gripper calls carry a yes/no check answered by Jev")
    p.add_argument("--overlay", action="store_true", help="draw the table grid and grasp point on the frames")
    p.add_argument("--max-turns", type=int, default=Config.max_turns)
    p.add_argument("--max-cost", type=float, default=Config.max_cost_usd, help="per-episode cap in dollars")
    p.add_argument("--out", type=Path, default=HERE / "runs")
    a = p.parse_args(argv)
    load_env()
    cfg = replace(Config(), model=a.model, effort=a.effort, max_side=a.max_side, cams=tuple(a.cams), jev_checks=a.jev, overlay=a.overlay,
                  max_turns=a.max_turns, max_cost_usd=a.max_cost)
    out = a.out / f"{time.strftime('%Y%m%d-%H%M%S')}-{a.tag}"
    out.mkdir(parents=True)
    print(f"{out}  {cfg}", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(run_seed, s, cfg, out): s for s in a.seeds}
        for f in futures:
            r = f.result()
            results.append(r)
            outcome = "placed" if r["placed"] else "lifted" if r["lifted"] else "failed"
            print(f"seed {r['seed']:2}: {outcome:6}  off {r['off_target_m']:.3f} m  {r.get('claude_calls', 0)} calls  "
                  f"${r.get('claude_usd', 0):.2f}  {r['wall_s']:.0f} s  end={r.get('end')}", flush=True)
    s = summarize(sorted(results, key=lambda r: r["seed"]), cfg)
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    (out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sorted(results, key=lambda r: r["seed"])))
    print(json.dumps({k: v for k, v in s.items() if k != "config"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
