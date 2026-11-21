"""Team 6's code-as-policy program (`experiments/aspire`, PR #8) under this experiment's error knobs, without
changing a line of it: their engine's sandbox, `Robot` and `localize` backends, with the scene built here.

Their `Robot` takes its camera model from `sim.model`; with `Errors` that model carries the true (perturbed) camera
mount, so here it always gets a camera from `Arm` instead, which knows only what the policy may know. Variants:

  no-tag        the nominal arm and camera: what the program would run with on the real arm today
  tag-runtime   the tag on the table; `localize`, `pixel_to_world` and `camera_ray` use the camera pose from the tag
                in any frame that shows it, carried by FK from the last such frame otherwise. Moves are unchanged.
  tag-calibrated  the tag only beforehand: a sweep of tag views fits the mount, joint zeros and sag (`calib`), and
                the program then runs through that arm. Readings, goals, FK and the camera are all corrected; the tag
                is off the table at run time.
  (oracle perception) for smoke tests only: `localize` answers with the true box or square, projected.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from valet_sim.scene import MOTORS, TagSpec
from valet_sim.task import episode, score

from . import tags
from .camera import MJ_TO_CV, Arm, Lens, true_camera
from .localize import calibrate, make_sim

VARIANTS = ("no-tag", "tag-runtime", "tag-calibrated")


def load_aspire(src: Path):
    """Import their package from an extracted copy (`git archive <commit> experiments/aspire`)."""
    sys.path.insert(0, str(Path(src) / "src"))
    from aspire import engine, perceive, robot  # noqa: F401

    return engine, perceive, robot


class ArmCameras:
    """Their `Cameras` interface, backed by an `Arm`. `anchors` (tag-runtime) maps joint readings to tag poses."""

    def __init__(self, base, arm: Arm, anchors: list | None = None):
        self.base, self.arm, self.anchors = base, arm, anchors

    def pose(self, joints: dict[str, float]) -> np.ndarray:
        model = self.arm.camera(joints)
        if not self.anchors:
            return model
        q = np.array([joints[j] for j in MOTORS[:5]])
        for aj, T in reversed(self.anchors):  # the frame that saw the tag, if this is it
            if np.abs(q - aj).max() < 0.05:
                return T
        aj, T = self.anchors[-1]  # else the last tag view, carried by FK
        return T @ np.linalg.inv(self.arm.camera(dict(zip(MOTORS, [*aj, 0.0])))) @ model

    def view(self, name: str, joints: dict[str, float]):
        v = self.base.view(name, joints)
        if name != "wrist":
            return v
        T = self.pose(joints)
        return type(v)(v.name, T[:3, 3].copy(), T[:3, :3] @ MJ_TO_CV, v.f_px, v.width, v.height, v.k1)


class CalibratedSim:
    """The sim seen through a calibrated arm: readings become model joint angles, goals become the readings that
    put the links there. The gripper passes through (it has no zero error or sag here)."""

    def __init__(self, sim, arm: Arm):
        self._sim, self._arm = sim, arm

    def __getattr__(self, name):
        return getattr(self._sim, name)

    def joints(self) -> dict[str, float]:
        return self._arm.model_joints(self._sim.joints())

    def set_goals(self, goals: dict[str, float]) -> None:
        if set(goals) <= {"gripper"}:
            return self._sim.set_goals(goals)
        cmd = self._arm.command({**self.joints(), **goals})
        self._sim.set_goals({k: cmd[k] for k in goals})

    def state(self) -> dict:
        s = self._sim.state()
        return {**s, "joints": {k: round(v, 2) for k, v in self.joints().items()}}


def oracle(sim, yaw_deg: float, lens: Lens = Lens()):
    """A `perceive.locate` stand-in that answers with the true object, for checking the plumbing at no cost. The box
    keeps its starting yaw (good until it is first moved)."""
    from aspire.perceive import Found

    def locate(image, text, key, backend="oracle"):
        t = sim.truth()
        T = true_camera(t)
        if "box" in text:
            b = t["box"]
            c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
            pts = [[b["x"] + c * dx - s * dy, b["y"] + s * dx + c * dy, z]
                   for dx in (-b["w"] / 2, b["w"] / 2) for dy in (-b["d"] / 2, b["d"] / 2) for z in (0, b["h"])]
        else:
            x, y = t["target"]
            pts = [[x + dx, y + dy, 0.001] for dx in (-0.035, 0.035) for dy in (-0.035, 0.035)]
        px = lens.project(T, pts)
        if not np.isfinite(px).all():
            return Found(False, None, None, "oracle")
        l, tp = px.min(0)
        r, bt = px.max(0)
        l, tp, r, bt = max(l, 0), max(tp, 0), min(r, 639), min(bt, 359)
        return Found(r - l > 2 and bt - tp > 2, (float(l), float(tp), float(r), float(bt)), 1.0, "oracle")

    return locate


def run_episode(code: str, seed: int, variant: str, backend: str, aspire_src: Path, kind: str = "all") -> dict:
    engine, perceive, robot_mod = load_aspire(aspire_src)
    import builtins

    t0 = time.time()
    tag = TagSpec()
    sim, _ = make_sim(seed, kind, tag if variant == "tag-runtime" else None)
    fit = calibrate(seed, kind, tag)["full"] if variant == "tag-calibrated" else None
    if fit is not None:
        run_sim = CalibratedSim(sim, Arm(offsets_deg=fit.offsets_deg, compliance_deg_per_nm=fit.compliance_deg_per_nm))
        cam_arm = Arm(mount=fit.mount)  # takes model joints, already corrected
    else:
        run_sim, cam_arm = sim, Arm()
    anchors: list | None = [] if variant == "tag-runtime" else None
    lens = Lens()

    class TaggedRobot(robot_mod.Robot):
        def _capture(self):
            super()._capture()
            if anchors is not None:
                s = tags.sight(np.asarray(self._frames["wrist"][1]), tag, lens)
                if s is not None:
                    j = self._sim.joints()
                    anchors.append((np.array([j[k] for k in MOTORS[:5]]), s.T_world_cam))

    rec = robot_mod.Recorder(None, keyframes=False)
    error = None
    if backend == "oracle":
        perceive.locate = oracle(sim, episode(seed).box.yaw_deg)
    try:
        engine.check_program(code)
        robot = TaggedRobot(run_sim, rec, backend=backend, limits=robot_mod.Limits())
        robot._cams = ArmCameras(robot._cams, cam_arm, anchors)
        env = {"__builtins__": {**engine.SAFE_BUILTINS, "__import__": engine._safe_import,
                                "__build_class__": builtins.__build_class__}, "__name__": "program"}
        env["__builtins__"]["print"] = lambda *a, **k: robot.log(*a)
        exec(compile(code, "<program>", "exec"), env)
        env["run"](robot)
    except robot_mod.EpisodeLimit as e:
        error = f"limit: {e}"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    sim.step(engine.SETTLE_AFTER_S)
    grade = score(sim.truth())
    n = sum(1 for c in rec.calls if c["prim"] == "localize")
    return {"seed": seed, "variant": variant, "backend": backend, "kind": kind, "score": grade, "error": error,
            "sim_s": round(float(sim.data.time), 1), "wall_s": round(time.time() - t0, 1), "localize_calls": n,
            "tag_views": None if anchors is None else len(anchors), "log": rec.log[-8:]}
