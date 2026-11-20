"""How well does Jev read where the box and the square are from one wrist frame? Frames with the open jaws at known
offsets from the box or the square, graded from the true pose. A development check for the questions, not part of
the policy."""

from __future__ import annotations

import argparse
import json

import numpy as np
from valet_sim.sim import Sim, SimSpec, with_box
from valet_sim.task import episode

from jev_policy import (
    NEAR_CLIP_M,
    Arm,
    Jev,
    Spend,
    jpeg,
    load_env,
    questions_for,
    step_vector,
)

GOALS = {"box": "Move the open jaws over the white box.", "square": "Carry the box over the blue square."}


def cases(n_per_seed: int, seeds: range) -> list[dict]:
    rng = np.random.default_rng(7)
    out = []
    for seed in seeds:
        ep = episode(seed)
        for k in range(n_per_seed):
            a = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(0.02, 0.09)
            obj = "box" if k % 2 == 0 else "square"
            xy = (ep.box.x, ep.box.y) if obj == "box" else ep.target_xy
            out.append({"seed": seed, "obj": obj, "xy": xy, "dx": r * np.cos(a), "dy": r * np.sin(a), "z": rng.uniform(0.08, 0.17), "ep": ep})
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--view", default="", help="'' for the wrist questions, '_side' for the side ones")
    a = p.parse_args()
    load_env()
    spend = Spend()
    jev = Jev(spend)
    rows = []
    for c in cases(a.n, range(5)):
        sim = Sim(with_box(SimSpec(), c["ep"].box, c["ep"].target_xy))
        sim.model.vis.map.znear = NEAR_CLIP_M / sim.model.stat.extent
        arm = Arm(sim)
        q = arm.ik.solve([c["xy"][0] + c["dx"], c["xy"][1] + c["dy"], c["z"]], 60.0, -90.0, gripper=70.0)
        sim.set_goals(q)
        arm.settle(q, 4.0)
        gp = np.array(sim.state()["grasp_point"])
        off = np.array([*c["xy"], 0]) - gp * [1, 1, 0]
        ax = arm.axes()
        comp = {m: float(off @ ax[m]) for m in ("ahead", "back", "left", "right")}
        best = max(comp, key=comp.get)
        ans = jev.ask({"goal": GOALS[c["obj"]], "arm": arm.reading()}, {"wrist": jpeg(arm.marked_frame()), "side": jpeg(arm.marked_frame("side"))}, questions_for(True))
        if a.view:
            ax = arm.side_axes()
            comp = {m: float(off @ ax[m]) for m in ("ahead", "back", "left", "right")}
            best = max(comp, key=comp.get)
        where = ans[f"where{a.view}:{c['obj']}"]
        move = step_vector(ans, c["obj"], ax, a.view)
        rows.append({"seed": c["seed"], "obj": c["obj"], "best": best, "got": where["choice"], "p_best": where["probabilities"][best],
                     "move:next": ans["move:next"]["choice"], "cm_before": round(float(np.linalg.norm(off)) * 100, 1),
                     "cm_after": round(float(np.linalg.norm(off - move)) * 100, 1)})
        print(rows[-1], flush=True)
    for obj in ("box", "square"):
        rs = [r for r in rows if r["obj"] == obj]
        print(json.dumps({"obj": obj, "n": len(rs), "accuracy": round(float(np.mean([r["best"] == r["got"] for r in rs])), 2),
                          "mean_p_best": round(float(np.mean([r["p_best"] for r in rs])), 3),
                          "closer": round(float(np.mean([r["cm_after"] < r["cm_before"] for r in rs])), 2),
                          "median_cm_before": float(np.median([r["cm_before"] for r in rs])),
                          "median_cm_after": float(np.median([r["cm_after"] for r in rs])),
                          "move_next": {m: sum(r["move:next"] == m for r in rs) for m in {r["move:next"] for r in rs}}}))
    print("jev tokens", spend.jev_tokens)


if __name__ == "__main__":
    main()
