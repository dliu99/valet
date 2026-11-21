"""Record the arm's joints and its upright wrist frame together, from a running armd, without moving anything.

Each pair is what sim-to-real fitting needs: the sim renders the same joints, and the real frame says what the
camera actually saw. With the arm relaxed (`POST /relax`), move it by hand over the box and the desk."""

from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.request
from pathlib import Path


def _call(url: str, method: str = "GET", body: dict | None = None) -> dict:
    req = urllib.request.Request(url, method=method, data=json.dumps(body or {}).encode() if method == "POST" else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as res:
        return json.loads(res.read())


def capture(armd: str, out: Path, every_s: float, count: int, min_move_deg: float) -> int:
    out.mkdir(parents=True, exist_ok=True)
    saved, last = 0, None
    while saved < count:
        state = _call(f"{armd}/state")
        joints = [state["joints"][m] for m in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]
        if last is None or max(abs(a - b) for a, b in zip(joints, last)) >= min_move_deg:
            look = _call(f"{armd}/frames", "POST", {"cams": ["wrist"]})
            frame = look["frames"].get("wrist")
            if frame:
                stem = out / f"{int(time.time() * 1000)}"
                stem.with_suffix(".jpg").write_bytes(base64.b64decode(frame["data_b64"]))
                meta = {k: v for k, v in frame.items() if k != "data_b64"}
                stem.with_suffix(".json").write_text(json.dumps({"state": state, "frame": meta}, indent=1))
                saved, last = saved + 1, joints
                print(f"{saved}/{count} {stem.name}: {[round(j, 1) for j in joints]}", flush=True)
            else:
                print(f"no wrist frame: {look['missing']}", flush=True)
        time.sleep(every_s)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m valet.armd.capture", description=__doc__)
    p.add_argument("--armd", default="http://127.0.0.1:8750")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--every", type=float, default=0.5, help="seconds between looks")
    p.add_argument("--count", type=int, default=40)
    p.add_argument("--min-move", type=float, default=3.0, help="save only after some joint moved this many degrees")
    a = p.parse_args(argv)
    return capture(a.armd.rstrip("/"), a.out, a.every, a.count, a.min_move)


if __name__ == "__main__":
    raise SystemExit(main())
