"""A scripted pick-and-place that reads the box's true pose: proof each benchmark scene is solvable, a reference
score, and a source of demonstrations. It cheats by design; no policy under test may read `truth()`."""

from __future__ import annotations

import argparse
import time

import numpy as np

from .ik import IK, Unreachable
from .scene import MOTORS, BoxSpec
from .sim import Sim, SimSpec, with_box
from .task import BENCHMARK_SEEDS, Episode, episode, score

OPEN_PCT = 70.0
PITCHES_DEG = (85.0, 70.0, 55.0, 40.0, 30.0)
APPROACH_M = 0.07
APPROACH_STEPS = 7
FIXED_JAW_TO_SITE_M = 0.020
JAW_CLEARANCE_M = 0.008
LIFT_M = 0.05
SETTLE_DEG = 0.5


def move(sim: Sim, goals: dict[str, float], timeout_s: float = 4.0) -> None:
    sim.set_goals(goals)
    t_end = sim.data.time + timeout_s
    while sim.data.time < t_end:
        sim.step(0.05)
        now = sim.joints()
        if all(abs(now[j] - goals[j]) < SETTLE_DEG for j in MOTORS if j != "gripper"):
            break


def close_gripper(sim: Sim) -> None:
    sim.set_goals({"gripper": 0.0})
    last = sim.joints()["gripper"]
    for _ in range(60):
        sim.step(0.05)
        now = sim.joints()["gripper"]
        if abs(now - last) < 0.05:
            break
        last = now
    sim.set_goals({"gripper": max(0.0, now - 3.0)})
    sim.step(0.3)


def _closing_misalignment(ik: IK, q: dict[str, float], box_yaw_deg: float) -> float:
    ik.pose(q)
    axis = ik.data.xmat[ik.gripper].reshape(3, 3)[:, 0]
    ang = np.degrees(np.arctan2(axis[1], axis[0])) - box_yaw_deg
    return abs((ang + 45) % 90 - 45)


def plan_grasp(ik: IK, centre: np.ndarray, box: BoxSpec) -> tuple[list[dict], float]:
    """Waypoints from above the box down to the grasp, with the fixed jaw just clear of the box's near face."""
    for pitch in PITCHES_DEG:
        rolls = np.arange(-180, 181, 10) if pitch >= 60 else (-90.0,)
        options = []
        for roll in rolls:
            try:
                q = ik.solve(centre, pitch, float(roll), gripper=OPEN_PCT)
            except Unreachable:
                continue
            options.append((_closing_misalignment(ik, q, box.yaw_deg), abs(roll + 90), float(roll), q))
        if not options:
            continue
        _, _, roll, q = min(options, key=lambda o: (round(o[0] / 5), o[1]))
        ik.pose(q)
        frame = ik.data.xmat[ik.gripper].reshape(3, 3)
        closing, approach = frame[:, 0], -frame[:, 2]
        site = centre + closing * (FIXED_JAW_TO_SITE_M - box.w / 2 - JAW_CLEARANCE_M)
        try:
            path, seed = [], q
            for k in range(APPROACH_STEPS, -1, -1):
                seed = ik.solve(site - approach * APPROACH_M * k / APPROACH_STEPS, pitch, roll, seed=seed, gripper=OPEN_PCT)
                path.append(seed)
        except Unreachable:
            continue
        return path, pitch
    raise Unreachable(f"no grasp reaches the box at {np.round(centre, 3).tolist()}")


def solve_near(ik: IK, xyz, pitch: float, roll: float, seed: dict, gripper: float) -> dict:
    for p in (pitch, pitch - 10, pitch - 20, pitch - 30, pitch + 10):
        try:
            return ik.solve(xyz, p, roll, seed=seed, gripper=gripper)
        except Unreachable:
            continue
    raise Unreachable(f"nothing near {pitch:.0f} deg reaches {np.round(xyz, 3).tolist()}")


def run(sim: Sim, ep: Episode) -> dict:
    ik = IK(sim.model)
    truth = sim.truth()
    b = truth["box"]
    centre = np.array([b["x"], b["y"], min(b["h"] / 2, 0.04)])
    try:
        path, pitch = plan_grasp(ik, centre, ep.box)
    except Unreachable as e:
        return {**score(sim.truth()), "note": str(e)}
    for q in path:
        move(sim, q, timeout_s=1.5)
    grasp = path[-1]
    close_gripper(sim)
    held = sim.joints()
    tx, ty = ep.target_xy
    roll, g = grasp["wrist_roll"], held["gripper"]
    try:
        lift = solve_near(ik, centre + [0, 0, LIFT_M], pitch, roll, held, g)
        carry = solve_near(ik, [tx, ty, centre[2] + LIFT_M], pitch, roll, lift, g)
        place = solve_near(ik, [tx, ty, centre[2] + 0.004], pitch, roll, carry, g)
    except Unreachable as e:
        return {**score(sim.truth()), "note": f"carry: {e}"}
    move(sim, lift)
    move(sim, carry)
    move(sim, place)
    move(sim, {**place, "gripper": OPEN_PCT})
    move(sim, {**carry, "gripper": OPEN_PCT})
    sim.step(0.5)
    return score(sim.truth())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m valet_sim.expert", description=__doc__)
    p.add_argument("--seeds", type=int, nargs="*", default=list(BENCHMARK_SEEDS))
    a = p.parse_args(argv)
    placed = lifted = 0
    t0 = time.time()
    for seed in a.seeds:
        ep = episode(seed)
        sim = Sim(with_box(SimSpec(), ep.box, ep.target_xy))
        result = run(sim, ep)
        placed += result["placed"]
        lifted += result["lifted"]
        print(f"seed {seed:2}: {'placed' if result['placed'] else 'lifted' if result['lifted'] else 'failed'}  {result}", flush=True)
    n = len(a.seeds)
    print(f"expert: placed {placed}/{n}, lifted {lifted}/{n}, {time.time() - t0:.0f} s wall")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
