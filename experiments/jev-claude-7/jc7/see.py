"""Seeing: Jev says where things are in a frame; geometry turns that into the table frame.

Jev judges image position well and distance poorly, so it is only asked for 2D boxes. Distance comes from the
camera model and the table: a box rests on the table, and the blue square lies on it. A box is fitted as a square
prism (centre, side, height, yaw) whose projection best matches the boxes Jev drew, over every view taken so far."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .camera import View
from .jev import Jev

BOX_QUESTIONS = {
    "box": {"type": "box", "instructions": "the white box standing on the table in `wrist`"},
    "top": {"type": "box", "instructions": "the top face of the white box in `wrist`"},
    "square": {"type": "box", "instructions": "the flat blue square on the table in `wrist`"},
}

EDGE = 0.02  # a Jev box edge this close to the frame edge may be the frame cutting the object off


@dataclass
class Sighting:
    view: View
    answers: dict
    frame: Image.Image | None = None
    kind: str = ""


@dataclass
class Estimate:
    xyz: np.ndarray  # box: centre of its footprint on the table; square: its centre
    size: float  # side, metres
    height: float = 0.0
    yaw_deg: float = 0.0
    residual: float = 0.0  # mean bbox-edge misfit, fraction of the image
    views: int = 0

    def summary(self) -> dict:
        d = {"x_cm": round(self.xyz[0] * 100, 1), "y_cm": round(self.xyz[1] * 100, 1), "side_cm": round(self.size * 100, 1)}
        if self.height:
            d["height_cm"] = round(self.height * 100, 1)
            d["yaw_deg"] = round(self.yaw_deg, 0)
        d["fit_error"] = round(self.residual, 3)
        d["views"] = self.views
        return d


@dataclass
class Eyes:
    """Every sighting this episode, and the estimates fitted to them."""

    jev: Jev
    sightings: list[Sighting] = field(default_factory=list)

    def look(self, frame: Image.Image, view: View, extra: dict | None = None, kind: str = "") -> dict:
        answers = self.jev.ask({"wrist": frame}, {**BOX_QUESTIONS, **(extra or {})})
        self.sightings.append(Sighting(view, answers, frame, kind))
        return answers

    def last(self, kind: str) -> Sighting | None:
        return next((s for s in reversed(self.sightings) if s.kind == kind), None)

    def box(self, since: int = 0) -> Estimate | None:
        obs = []
        for s in self.sightings[since:]:
            for key, part in (("box", "all"), ("top", "top")):
                b = s.answers[key]["box"]
                if _usable(b):
                    obs.append((s.view, part, b))
        return fit_prism(obs) if obs else None

    def square(self, since: int = 0) -> Estimate | None:
        obs = [(s.view, s.answers["square"]["box"]) for s in self.sightings[since:] if _usable(s.answers["square"]["box"])]
        return fit_square(obs) if obs else None

    def square_fused(self) -> Estimate | None:
        """The square never moves, so every look aimed at it counts. A look that disagrees with the others (the held
        box hiding part of the square, say) is dropped: keep the looks that most others agree with, within 2 cm."""
        obs = [(s.view, s.answers["square"]["box"]) for s in self.sightings
               if s.kind == "square_over" and _usable(s.answers["square"]["box"])]
        if not obs:
            return None
        casts = []
        for view, b in obs:
            c = view.on_plane((b["left"] + b["right"]) / 2, (b["top"] + b["bottom"]) / 2, 0.0)
            casts.append(None if c is None else c[:2])
        support = [0 if c is None else sum(1 for d in casts if d is not None and np.linalg.norm(c - d) < 0.02) for c in casts]
        best = max(support)
        kept = [o for o, k in zip(obs, support) if k == best]
        est = fit_square(kept)
        est.views = len(kept)
        return est

    def locate_box(self, body, guess=None, top_z: float = 0.075, tries: int = 3) -> Estimate | None:
        """Centre the box's top face under the camera (looking straight down), then look again from higher up.
        Centred, the box's unknown height barely moves the estimate; the two heights give its height and size."""
        if guess is None:
            guess = self._cast("top", top_z)
            if guess is None:
                guess = self._cast("box", top_z)
        if guess is None:
            return None
        xy = np.asarray(guess[:2], float)
        first = len(self.sightings)
        for _ in range(tries):
            view = body.look_at([xy[0], xy[1], top_z], cam_height=0.13)
            if view is None:
                break
            self.look(body.frame()[0], view, kind="box_over")
            new = self._cast("top", top_z)
            if new is None:
                break
            moved = np.linalg.norm(new[:2] - xy)
            xy = new[:2]
            if moved < 0.004:
                break
        view = body.look_at([xy[0], xy[1], top_z], cam_height=0.19)
        if view is not None:
            self.look(body.frame()[0], view, kind="box_high")
        view = body.look_across([xy[0], xy[1], 0.04])
        if view is not None:
            self.look(body.frame()[0], view, kind="box_across")
        est = self.box(since=first)
        if est is None:
            return None
        return est

    def locate_square(self, body, guess=None, tries: int = 2) -> Estimate | None:
        if guess is None:
            guess = self._cast("square", 0.0)
        if guess is None:
            return None
        xy = np.asarray(guess[:2], float)
        first = len(self.sightings)
        for _ in range(tries):
            view = body.look_at([xy[0], xy[1], 0.0], cam_height=0.20)
            if view is None:
                break
            self.look(body.frame()[0], view, kind="square_over")
            new = self._cast("square", 0.0)
            if new is None:
                break
            moved = np.linalg.norm(new[:2] - xy)
            xy = new[:2]
            if moved < 0.004:
                break
        est = self.square_fused()
        if est is None:  # no close look reached it: fall back to the overview
            obs = [(s.view, s.answers["square"]["box"]) for s in self.sightings if s.kind == "start" and _usable(s.answers["square"]["box"])]
            est = fit_square(obs) if obs else None
        return est

    def _cast(self, key: str, z: float) -> np.ndarray | None:
        """Where the centre of Jev's latest box for `key` meets the plane at height z."""
        for s in reversed(self.sightings):
            b = s.answers[key]["box"]
            if _usable(b):
                return s.view.on_plane((b["left"] + b["right"]) / 2, (b["top"] + b["bottom"]) / 2, z)
        return None


def _usable(b: dict) -> bool:
    return b["right"] - b["left"] > 0.01 and b["bottom"] - b["top"] > 0.01


def _bbox(view: View, pts: np.ndarray) -> np.ndarray | None:
    fs = [view.project(p) for p in pts]
    if any(f is None for f in fs):
        return None
    xs, ys = [f[0] for f in fs], [f[1] for f in fs]
    return np.array([min(ys), max(ys), min(xs), max(xs)])  # top, bottom, left, right


def _edges(b: dict) -> np.ndarray:
    return np.array([b["top"], b["bottom"], b["left"], b["right"]])


def _clipped_misfit(pred: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """Edge errors, ignoring an edge Jev put at the frame border when the model puts it beyond."""
    err = pred - seen
    for i, (lo_side) in enumerate((True, False, True, False)):
        at_border = seen[i] < EDGE if lo_side else seen[i] > 1 - EDGE
        beyond = pred[i] < seen[i] if lo_side else pred[i] > seen[i]
        if at_border and beyond:
            err[i] = 0.0
    return err


def prism_corners(x, y, side, h, yaw_deg, top_only=False) -> np.ndarray:
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    pts = []
    for z in ((h,) if top_only else (0.0, h)):
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            dx, dy = sx * side / 2, sy * side / 2
            pts.append([x + c * dx - s * dy, y + s * dx + c * dy, z])
    return np.array(pts)


def _huber(r: np.ndarray, k: float = 0.03) -> float:
    a = np.abs(r)
    return float(np.sum(np.where(a < k, 0.5 * a * a, k * (a - 0.5 * k))))


def fit_prism(obs: list[tuple[View, str, dict]]) -> Estimate:
    """Least-squares fit of (x, y, side, height, yaw) to Jev's 2D boxes of the whole box and of its top face."""

    def cost(p):
        x, y, side, h, yaw = p
        r = []
        for view, part, b in obs:
            pred = _bbox(view, prism_corners(x, y, side, h, yaw, top_only=(part == "top")))
            if pred is None:
                return 1e3
            r.extend(_clipped_misfit(pred, _edges(b)))
        r = np.array(r)
        prior = ((side - 0.05) / 0.03) ** 2 * 1e-4 + ((h - 0.075) / 0.05) ** 2 * 1e-4
        return _huber(r) + prior

    # start from the top-face centre seen in the latest view, cast onto a mid-height plane
    view, _, b = next(((v, p, b) for v, p, b in reversed(obs) if p == "top"), obs[-1])
    start = view.on_plane((b["left"] + b["right"]) / 2, (b["top"] + b["bottom"]) / 2, 0.075)
    x0, y0 = (start[0], start[1]) if start is not None else (0.25, 0.0)
    best = None
    for yaw0 in (-30.0, 0.0, 30.0):
        for h0 in (0.06, 0.09):
            p = _nelder_mead(cost, np.array([x0, y0, 0.05, h0, yaw0]), np.array([0.02, 0.02, 0.01, 0.02, 15.0]))
            c = cost(p)
            if best is None or c < best[1]:
                best = (p, c)
    x, y, side, h, yaw = best[0]
    n = sum(1 for _ in obs) * 4
    resid = float(np.sqrt(2 * best[1] / max(n, 1)))
    yaw = ((yaw + 45) % 90) - 45
    return Estimate(np.array([x, y, 0.0]), float(side), float(h), float(yaw), resid, len({id(v) for v, _, _ in obs}))


def fit_square(obs: list[tuple[View, dict]]) -> Estimate:
    def cost(p):
        x, y, side = p
        r = []
        for view, b in obs:
            pts = np.array([[x + sx * side / 2, y + sy * side / 2, 0.0] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
            pred = _bbox(view, pts)
            if pred is None:
                return 1e3
            r.extend(_clipped_misfit(pred, _edges(b)))
        return _huber(np.array(r)) + ((side - 0.07) / 0.05) ** 2 * 1e-4

    view, b = obs[-1]
    start = view.on_plane((b["left"] + b["right"]) / 2, (b["top"] + b["bottom"]) / 2, 0.0)
    x0, y0 = (start[0], start[1]) if start is not None else (0.25, 0.0)
    p = _nelder_mead(cost, np.array([x0, y0, 0.07]), np.array([0.02, 0.02, 0.01]))
    resid = float(np.sqrt(2 * cost(p) / max(len(obs) * 4, 1)))
    return Estimate(np.array([p[0], p[1], 0.0]), float(p[2]), residual=resid, views=len(obs))


def _nelder_mead(f, x0: np.ndarray, step: np.ndarray, iters: int = 400, tol: float = 1e-9) -> np.ndarray:
    n = len(x0)
    pts = [x0.astype(float)] + [x0 + np.eye(n)[i] * step[i] for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        centroid = np.mean(pts[:-1], axis=0)
        xr = centroid + (centroid - pts[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = centroid + 2 * (centroid - pts[-1])
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = centroid + 0.5 * (pts[-1] - centroid)
            fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0] + 0.5 * (p - pts[0]) for p in pts]
                vals = [f(p) for p in pts]
    return pts[int(np.argmin(vals))]
