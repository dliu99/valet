"""valet run | pose | locate | ask. `valet <command> --help` names the flags."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from . import config as conf
from .escalate import Abort as AbortDecider, AskClaude, AskPerson
from .geometry import box_to_world
from .judge import JevJudge, YesNo
from .recipe import SANDWICH, find
from .records import Records
from .runtime import Runner


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="valet")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="build the sandwich")
    run.add_argument("--config", help="valet.toml (needed for the real arm)")
    run.add_argument("--fake", action="store_true", help="fake arm and fake Jev: no hardware, no API calls")
    run.add_argument("--decider", choices=["person", "claude", "abort"], default="person", help="who settles unsure checks")
    run.add_argument("--runs", default="runs", help="where records go")

    pose = sub.add_parser("pose", help="print the arm's joints and grasp point (for down_joints, home_joints)")
    pose.add_argument("--config", required=True)

    loc = sub.add_parser("locate", help="find one recipe item from the arm's current pose and print where it is")
    loc.add_argument("item", choices=[SANDWICH.board.name, *(i.name for i in SANDWICH.layers)])
    loc.add_argument("--config", required=True)

    ask = sub.add_parser("ask", help="ask Jev a yes/no question about a saved picture (refer to it as `wrist`)")
    ask.add_argument("image")
    ask.add_argument("question")

    a = p.parse_args(argv)
    return {"run": _run, "pose": _pose, "locate": _locate, "ask": _ask}[a.cmd](a)


def _run(a) -> int:
    cfg = conf.load(a.config)
    records = Records(a.runs, SANDWICH.name)
    settings = conf.settings(cfg)
    if a.fake:
        from .fake import FakeJudge, FakeRobot, sandwich_world

        world = sandwich_world()
        robot = FakeRobot(world)
        judge = FakeJudge(world, robot)
    else:
        if not a.config:
            print("valet run: the real arm needs --config (or pass --fake)", file=sys.stderr)
            return 2
        robot = conf.so101(cfg)
        robot.connect()
        judge = JevJudge(model=cfg.get("jev", {}).get("model") or None)
    decider = {
        "person": lambda: AskPerson(show_frame=lambda f: _save(f, records.dir / "escalation.jpg")),
        "claude": lambda: AskClaude(**cfg.get("claude", {})),
        "abort": AbortDecider,
    }[a.decider]()
    try:
        outcome = Runner(robot, judge, conf.skills(cfg), decider, records, settings).build(SANDWICH)
    finally:
        if hasattr(robot, "disconnect"):
            robot.disconnect()
    print(f"{outcome}\nrecords: {records.dir}")
    return 0 if outcome == "done" else 1


def _pose(a) -> int:
    robot = conf.so101(conf.load(a.config))
    robot.connect()
    try:
        obs = robot.observe()
        print(json.dumps({"joints": [round(v, 2) for v in obs.joints.values()], "grasp_point_m": [round(v, 4) for v in obs.ee_xyz]}))
    finally:
        robot.disconnect()
    return 0


def _locate(a) -> int:
    cfg = conf.load(a.config)
    settings = conf.settings(cfg)
    item = next(i for i in [SANDWICH.board, *SANDWICH.layers] if i.name == a.item)
    robot = conf.so101(cfg)
    robot.connect()
    try:
        obs = robot.observe()
        answers = JevJudge(model=cfg.get("jev", {}).get("model") or None).ask(obs.frame, find(item))
    finally:
        robot.disconnect()
    p, box = answers[f"present:{item.name}"], answers[f"where:{item.name}"]
    z = settings.table_z + (settings.board_height_m if item.thickness_m == 0.0 else item.thickness_m)
    where = box_to_world(robot.camera, obs.T_world_cam, box, z)
    print(json.dumps({"present": round(p, 3), "box": box.__dict__, "xyz_m": where and [round(v, 4) for v in where[0]], "span_m": where and round(where[1], 4)}))
    return 0


def _ask(a) -> int:
    frame = np.asarray(PILImage.open(a.image).convert("RGB"))
    judge = JevJudge()
    try:
        print(round(judge.ask(frame, {"q": YesNo(a.question)})["q"], 3))
    finally:
        judge.close()
    return 0


def _save(frame: np.ndarray, path: Path) -> str:
    PILImage.fromarray(frame).save(path)
    return str(path)


if __name__ == "__main__":
    sys.exit(main())
