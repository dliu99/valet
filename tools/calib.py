"""Fit the wrist camera (field of view, mount, barrel) so table features seen from several poses land on one spot.

    uv run --no-sync python tools/calib.py capture   # moves the arm through POSES, saves frames + joints
    uv run --no-sync python tools/calib.py fit       # ORB-match table features across views, least squares
"""

from __future__ import annotations

import base64
import itertools
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

OUT = Path(__file__).parent / "calib"
URL = "http://127.0.0.1:8750"
POSES = [
    dict(pan_deg=0, reach_m=0.16, height_m=0.22, pitch_deg=40),
    dict(pan_deg=0, reach_m=0.16, height_m=0.15, pitch_deg=40),
    dict(pan_deg=0, reach_m=0.20, height_m=0.18, pitch_deg=55),
    dict(pan_deg=12, reach_m=0.18, height_m=0.20, pitch_deg=45),
    dict(pan_deg=-12, reach_m=0.18, height_m=0.20, pitch_deg=45),
    dict(pan_deg=-4, reach_m=0.22, height_m=0.13, pitch_deg=30),
    dict(pan_deg=4, reach_m=0.14, height_m=0.24, pitch_deg=25),
]


def call(method: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(URL + path, json.dumps(body).encode() if body is not None else None, {"Content-Type": "application/json"}, method=method)
    return json.load(urllib.request.urlopen(req, timeout=30))


def capture() -> None:
    OUT.mkdir(exist_ok=True)
    call("POST", "/hold", {})
    for i, p in enumerate(POSES):
        r = call("POST", "/act", {"motion": {"kind": "to", **p}})
        time.sleep(1.5)
        st = call("GET", "/state")
        f = call("POST", "/frames", {"cams": ["wrist"]})["frames"]["wrist"]
        (OUT / f"{i}.jpg").write_bytes(base64.b64decode(f["data_b64"]))
        (OUT / f"{i}.json").write_text(json.dumps({"pose": p, "ok": r.get("ok"), "joints": st["joints"], "tip": st["tip"]}))
        print(i, r.get("ok"), r.get("stopped") or r.get("refused") or "", {k: round(v, 1) for k, v in st["joints"].items()})


def features(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.erode((gray > 10).astype(np.uint8) * 255, np.ones((25, 25), np.uint8))
    gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.SIFT_create(6000).detectAndCompute(gray, mask)


def matches(a, b) -> list[tuple[np.ndarray, np.ndarray]]:
    (ka, da), (kb, db) = a, b
    if da is None or db is None:
        return []
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.75 * n.distance]
    return [(np.array(ka[m.queryIdx].pt), np.array(kb[m.trainIdx].pt)) for m in good]


class Cam:
    def __init__(self):
        import mujoco

        from valet.armd.kinematics import MOTORS, Kinematics
        from valet.armd.simscene import WRIST_CAMERA, SceneSpec, build

        self.mj = mujoco
        self.kin = Kinematics()
        self.model = build(SceneSpec())
        self.data = mujoco.MjData(self.model)
        self.qadr = [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in MOTORS]
        self.cam = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAMERA)
        self.pos0 = self.model.cam_pos[self.cam].copy()
        self.quat0 = self.model.cam_quat[self.cam].copy()
        self.fovy0 = float(self.model.cam_fovy[self.cam])
        self.motors = MOTORS

    def set(self, x: np.ndarray) -> None:
        fovy, k1, rx, ry, rz, px, py, pz = x
        q0 = self.quat0
        r = Rotation.from_rotvec([rx, ry, rz]) * Rotation.from_quat([q0[1], q0[2], q0[3], q0[0]])
        qx, qy, qz, qw = r.as_quat()
        self.model.cam_quat[self.cam] = [qw, qx, qy, qz]
        self.model.cam_pos[self.cam] = self.pos0 + np.array([px, py, pz])
        self.model.cam_fovy[self.cam] = fovy
        self.k1 = k1

    def pose(self, joints: dict) -> tuple[np.ndarray, np.ndarray, float]:
        q = np.array([joints[m] for m in self.motors])
        self.data.qpos[self.qadr] = np.radians(q)
        self.mj.mj_kinematics(self.model, self.data)
        self.mj.mj_camlight(self.model, self.data)
        return self.data.cam_xpos[self.cam].copy(), self.data.cam_xmat[self.cam].reshape(3, 3).copy(), float(joints["wrist_roll"])

    def table(self, pose, uv: np.ndarray, w: int, h: int) -> np.ndarray:
        from valet.armd.frames import wrist_upright_turn_deg

        origin, rot, roll = pose
        aspect = w / h
        x, y = (uv[:, 0] / w - 0.5) * 2 * aspect, (uv[:, 1] / h - 0.5) * 2
        a = math.radians(wrist_upright_turn_deg(roll) + 180.0)
        x, y = x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)
        k = 1.0 + self.k1 * ((x / aspect) ** 2 + (y / aspect) ** 2)
        x, y = x * k, y * k
        f = 1.0 / math.tan(math.radians(self.model.cam_fovy[self.cam]) / 2)
        d = (rot @ np.stack([x / f, -y / f, -np.ones_like(x)])).T
        s = -origin[2] / d[:, 2]
        return origin[None, :2] + d[:, :2] * s[:, None]


def fit() -> None:
    views = []
    for meta in sorted((p for p in OUT.glob("*.json") if p.stem.isdigit()), key=lambda p: int(p.stem)):
        img = cv2.imread(str(meta.with_suffix(".jpg")))
        views.append((json.loads(meta.read_text()), img, features(img)))
    cam = Cam()
    pairs = []
    for (i, a), (j, b) in itertools.combinations(enumerate(views), 2):
        m = matches(a[2], b[2])
        if len(m) < 12:
            continue
        pa, pb = np.array([x for x, _ in m]), np.array([y for _, y in m])
        H, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 6.0)
        if H is None:
            continue
        keep = inl.ravel().astype(bool)
        print(f"views {i}-{j}: {len(m)} matches, {keep.sum()} on one plane")
        if keep.sum() >= 12:
            pairs.append((i, j, pa[keep], pb[keep]))
    h, w = views[0][1].shape[:2]
    x0 = np.array([cam.fovy0, -0.205, 0, 0, 0, 0, 0, 0])

    def residual(x):
        cam.set(x)
        poses = [cam.pose(v[0]["joints"]) for v in views]
        out = []
        for i, j, pa, pb in pairs:
            ta, tb = cam.table(poses[i], pa, w, h), cam.table(poses[j], pb, w, h)
            bad = ~np.isfinite(ta).all(1) | ~np.isfinite(tb).all(1)
            r = ta - tb
            r[bad] = 1.0
            out.append(np.clip(r, -1, 1).ravel())
        return np.concatenate(out)

    for name, x in (("start", x0),):
        r = residual(x).reshape(-1, 2)
        print(name, "median miss cm", round(float(np.median(np.linalg.norm(r, axis=1))) * 100, 2))
    lo = [30, -0.6, -0.4, -0.4, -0.4, -1e-6, -1e-6, -1e-6]
    hi = [110, 0.6, 0.4, 0.4, 0.4, 1e-6, 1e-6, 1e-6]
    best = None
    for fovy in (50, 60, 70, 84):
        x = x0.copy()
        x[0] = fovy
        sol = least_squares(residual, x, bounds=(lo, hi), loss="soft_l1", f_scale=0.005)
        miss = np.median(np.linalg.norm(residual(sol.x).reshape(-1, 2), axis=1))
        print(f"from fovy {fovy}: fovy {sol.x[0]:.1f} k1 {sol.x[1]:.3f} rot {np.degrees(sol.x[2:5]).round(2)} pos_mm {(sol.x[5:]*1000).round(1)} median miss {miss*100:.2f} cm")
        if best is None or miss < best[1]:
            best = (sol.x, miss)
    x = best[0]
    cam.set(x)
    q = cam.model.cam_quat[cam.cam]
    p = cam.model.cam_pos[cam.cam]
    result = {"wrist_fovy_deg": round(float(x[0]), 2), "barrel_k1": round(float(x[1]), 4), "wrist_pos": [round(float(v), 5) for v in p], "wrist_quat": [round(float(v), 6) for v in q], "median_miss_cm": round(float(best[1]) * 100, 2)}
    print(json.dumps(result))
    (OUT / "fit.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    {"capture": capture, "fit": fit}[sys.argv[1]]()
