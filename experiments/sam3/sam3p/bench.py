"""Run the SAM 3 policy on benchmark seeds, grade with `task.score`, and record every episode on the shared
scoreboard (experiments/sam3/results/episodes.jsonl). SAM 3 loads once per process.

    uv run python -m sam3p.bench --seeds 0-4 --config base [--record]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from valet_sim.scoreboard import Episode, record
from valet_sim.sim import Sim, SimSpec, with_box
from valet_sim.task import episode, score

from .perceive import Perception
from .policy import Config, SamPolicy

RESULTS = Path(__file__).resolve().parents[1] / "results"
CONFIGS = {
    "base": Config(name="sam3-504"),
    "pregrasp": Config(name="sam3-504-pregrasp", pre_grasp_look=True),
    "roll": Config(name="sam3-504-roll", align_roll=True),
    "nooffset": Config(name="sam3-504-nooffset", place_offset=False),
    "twoview": Config(name="sam3-504-twoview", two_view=True),
}


def parse_seeds(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m sam3p.bench", description=__doc__)
    p.add_argument("--seeds", default="0-4")
    p.add_argument("--config", default="base", help=f"one of {sorted(CONFIGS)}; join with + to combine flags")
    p.add_argument("--res", type=int, default=504)
    p.add_argument("--record", action="store_true", help="append to the shared scoreboard")
    p.add_argument("--tag", default="")
    p.add_argument("--oracle", action="store_true", help="true masks instead of SAM 3 (debugging; never recorded)")
    p.add_argument("--prompt", default="white box", help="SAM 3's words for the box")
    p.add_argument("--merge", action="store_true", help="join touching box instances (a shaded face can come back alone)")
    p.add_argument("--fallback", default=None, help="a second wording for the box when the first finds nothing")
    a = p.parse_args(argv)
    cfg = Config()
    for part in a.config.split("+"):
        c = CONFIGS[part]
        cfg = replace(cfg, **{k: getattr(c, k) for k in ("pre_grasp_look", "align_roll", "place_offset", "two_view", "grasp_tries", "place_tries")
                              if getattr(c, k) != getattr(Config(), k)})
    cfg = replace(cfg, name=f"sam3-{a.res}" + "".join(f"-{x}" for x in a.config.split("+") if x != "base")
                  + ("" if a.prompt == "white box" else "-" + a.prompt.replace(" ", "")) + ("-merge" if a.merge else "") + (f"-or{a.fallback.replace(' ', '')}" if a.fallback else "")
                  + (f"-{a.tag}" if a.tag else ""))

    if a.oracle:
        from .render import OracleSegmenter

        seg = OracleSegmenter()
        cfg = replace(cfg, name=cfg.name.replace("sam3-", "oracle-"))
        a.record = False
    else:
        from .seg import Segmenter

        seg = Segmenter(resolution=a.res)
    perception = Perception(seg, box_prompt=a.prompt, merge=a.merge, fallback=a.fallback)
    log_dir = RESULTS / "runs" / cfg.name
    log_dir.mkdir(parents=True, exist_ok=True)
    placed = lifted = 0
    rows = []
    for seed in parse_seeds(a.seeds):
        ep = episode(seed)
        sim = Sim(with_box(SimSpec(), ep.box, ep.target_xy))
        if a.oracle:
            seg.attach(sim)
        graded = []

        def witness(tag: str, rec: dict, sim=sim, graded=graded) -> None:  # grading only: never fed back
            t = sim.truth()
            g = {"tag": tag}
            if rec["box"] is not None:
                g["box_cm"] = round(100 * float(np.hypot(rec["box"][0] - t["box"]["x"], rec["box"][1] - t["box"]["y"])), 2)
            if rec["square"] is not None:
                g["square_cm"] = round(100 * float(np.hypot(rec["square"][0] - t["target"][0], rec["square"][1] - t["target"][1])), 2)
            graded.append(g)

        seg.calls.clear()
        t0 = time.time()
        policy = SamPolicy(sim, perception, cfg, witness)
        try:
            out = policy.run()
        except Exception as e:  # an episode that crashes is a failed episode, and its error is kept
            out = {"outcome": f"error: {type(e).__name__}: {e}"}
        wall = time.time() - t0
        s = score(sim.truth())
        placed += s["placed"]
        lifted += s["lifted"]
        n_seg = sum(1 for c in seg.calls if c.get("new_frame", True))  # one fast call per frame segmented
        seg_s = [c["s"] for c in seg.calls]
        row = {"seed": seed, **s, **out, "wall_s": round(wall, 1), "sam_calls": n_seg,
               "sam_s_median": round(statistics.median(seg_s), 3) if seg_s else None, "graded": graded,
               "truth": sim.truth()["box"], "events": policy.events}
        rows.append(row)
        (log_dir / f"seed-{seed:02d}.json").write_text(json.dumps(row, indent=1, default=float))
        if a.record:
            record(RESULTS / "episodes.jsonl", Episode(experiment="sam3", config=cfg.name, seed=seed, placed=bool(s["placed"]),
                                                       lifted=bool(s["lifted"]), seconds=round(wall, 1), slow_calls=0,
                                                       fast_calls=n_seg, usd=0.0, calls={"sam3": n_seg}))
        print(f"seed {seed:2}: {'placed' if s['placed'] else 'lifted' if s['lifted'] else 'failed':7} off {s['off_target_m'] * 100:5.1f} cm "
              f"tilt {s['tilt_deg']:5.1f}  {out.get('outcome')}  sam {n_seg} x {row['sam_s_median']} s  wall {wall:.0f} s  graded {graded}", flush=True)
    n = len(rows)
    print(f"{cfg.name}: placed {placed}/{n}, lifted {lifted}/{n}, median wall {statistics.median(r['wall_s'] for r in rows):.0f} s, "
          f"SAM calls/ep {statistics.mean(r['sam_calls'] for r in rows):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
