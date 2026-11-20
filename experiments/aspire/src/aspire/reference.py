"""The scripted expert (reads the true box pose) on any task and seed set, as the reference to beat.

    uv run python -m aspire.reference --task variant --seeds heldout
"""

from __future__ import annotations

import argparse
import json

from valet_sim import expert
from valet_sim.sim import Sim, SimSpec, with_box

from . import tasks


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m aspire.reference")
    p.add_argument("--task", default="benchmark", choices=sorted(tasks.EPISODES))
    p.add_argument("--seeds", default="heldout", choices=["heldout", "debug"])
    a = p.parse_args(argv)
    seeds = (tasks.HELDOUT_SEEDS if a.seeds == "heldout" else tasks.DEBUG_SEEDS)[a.task]
    placed = lifted = 0
    rows = []
    for s in seeds:
        ep = tasks.make(a.task, s)
        r = expert.run(Sim(with_box(SimSpec(), ep.box, ep.target_xy)), ep)
        placed += r["placed"]
        lifted += r["lifted"]
        rows.append({"seed": s, "placed": r["placed"], "lifted": r["lifted"]})
    print(json.dumps({"task": a.task, "seeds": a.seeds, "placed": placed, "lifted": lifted, "n": len(seeds), "rows": rows}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
