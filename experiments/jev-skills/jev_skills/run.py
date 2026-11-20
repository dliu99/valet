"""Run benchmark episodes one after another, each in its own process (a renderer crash costs one episode, and only
one sim is alive at a time), and record each to the shared scoreboard.

    uv run python -m jev_skills.run --config base --seeds 0 1 2 3 4
    uv run python -m jev_skills.run --config base-local --decider local --seeds 0 1 2 3 4
    uv run python -m jev_skills.run --config x --set grip_below_top_m=0.03 --skills look_workspace,look_box,...

`--chooser oracle` is a diagnostic, not a policy: it picks the skill from the simulator's truth, to separate the
skills' mechanics from Jev's choices. Its episodes are never recorded to the scoreboard."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

from valet_sim.scoreboard import Episode as Row, record

from . import skills as S
from .runtime import Config, Episode, config_dict
from .models import load_env

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results" / "episodes.jsonl"
RUNS = HERE / "runs"
SLOW_CALL_S = 1.0
ORDER = ["done", "back_off", "place", "carry", "look_square", "grasp_near_top", "look_box", "look_workspace"]


class OracleChooser(Episode):
    def choose(self, frame):
        ok = self.label_choice() or {S.DONE_KEY}
        pick = next(k for k in ORDER if k in ok)
        self.ledger.note("oracle:choose")
        return pick, {"choice": pick, "confidence": 1.0, "probabilities": {}}


def child(a) -> None:
    load_env()
    d = json.loads(a.cfg_json)
    params = S.Params(**d.pop("params"))
    cfg = Config(**{**d, "skills": tuple(d["skills"]), "params": params})
    cls = OracleChooser if a.chooser == "oracle" else Episode
    out = cls(a.seed, cfg, Path(a.out)).run()
    print("SUMMARY " + json.dumps(out), flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m jev_skills.run", description=__doc__)
    p.add_argument("--config", default="base")
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    p.add_argument("--decider", default="jev", choices=["jev", "local"])
    p.add_argument("--no-claude", action="store_true")
    p.add_argument("--chooser", default="jev", choices=["jev", "oracle"])
    p.add_argument("--skills", default=",".join(S.DEFAULT_SKILLS))
    p.add_argument("--set", nargs="*", default=[], help="Params or Config fields, key=value")
    p.add_argument("--no-record", action="store_true")
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    p.add_argument("--cfg-json", help=argparse.SUPPRESS)
    p.add_argument("--out", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.child:
        child(a)
        return 0

    params = S.Params()
    cfg = Config(name=a.config, decider=a.decider, claude=not a.no_claude, skills=tuple(a.skills.split(",")), params=params)
    pnames, cnames = {f.name for f in fields(S.Params)}, {f.name for f in fields(Config)}
    for kv in a.set:
        k, v = kv.split("=", 1)
        target = params if k in pnames else cfg if k in cnames else None
        if target is None:
            raise SystemExit(f"unknown setting {k}")
        cur = getattr(target, k)
        setattr(target, k, (v.lower() in ("1", "true", "yes")) if isinstance(cur, bool) else type(cur)(v))
    run_dir = RUNS / a.config
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({**config_dict(cfg), "chooser": a.chooser}, indent=1))
    rows = []
    for seed in a.seeds:
        out = run_dir / f"seed{seed:02d}"
        cmd = [sys.executable, "-m", "jev_skills.run", "--child", "--seed", str(seed), "--chooser", a.chooser,
               "--cfg-json", json.dumps(config_dict(cfg)), "--out", str(out)]
        summary = None
        for attempt in range(2):
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
            line = next((l for l in proc.stdout.splitlines() if l.startswith("SUMMARY ")), None)
            if line:
                summary = json.loads(line[8:])
                break
            (out / f"error{attempt}.txt").write_text(proc.stdout[-4000:] + "\n" + proc.stderr[-8000:])
            print(f"seed {seed}: child failed (attempt {attempt}); see {out}/error{attempt}.txt", flush=True)
        if summary is None:
            continue
        rows.append(summary)
        print(f"seed {seed:2}: {'PLACED' if summary['placed'] else 'lifted' if summary['lifted'] else 'failed'} "
              f"off {100 * summary['off_target_m']:.1f} cm tilt {summary['tilt_deg']} finished {summary['finished']} "
              f"fast {summary['fast_calls']} slow {summary['slow_calls']} {summary['seconds']} s ${summary['usd']}", flush=True)
        if not a.no_record and a.chooser == "jev":
            calls = {k: v for k, v in summary["calls"].items()}
            slow, fast = int(summary["slow_calls"]), int(summary["fast_calls"])
            if fast and summary["fast_s"] / fast > SLOW_CALL_S:  # the scoreboard counts any model over ~1 s as slow
                slow, fast = slow + fast, 0
            record(RESULTS, Row("jev-skills", a.config, seed, bool(summary["placed"]), bool(summary["lifted"]),
                                float(summary["seconds"]), slow, fast, float(summary["usd"]), calls))
    if rows:
        n = len(rows)
        agg = {"config": a.config, "seeds": [r["seed"] for r in rows], "placed": sum(r["placed"] for r in rows),
               "lifted": sum(r["lifted"] for r in rows), "n": n,
               "median_s": statistics.median(r["seconds"] for r in rows),
               "fast_calls": statistics.mean(r["fast_calls"] for r in rows),
               "slow_calls": statistics.mean(r["slow_calls"] for r in rows),
               "usd": round(sum(r["usd"] for r in rows), 4)}
        (run_dir / "summary.json").write_text(json.dumps({"agg": agg, "episodes": rows}, indent=1))
        print(json.dumps(agg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
