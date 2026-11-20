"""Offline tests: no TypeSafe, Anthropic or local-model calls. A fake System One draws boxes from colour in the frame
and picks skills by reading the state it is given."""

from __future__ import annotations

import numpy as np
import pytest

from jev_skills import perception as P
from jev_skills import skills as S
from jev_skills.arm import View, jaw_mask
from jev_skills.models import Ledger
from jev_skills.runtime import Config, Episode


class FakeSystemOne:
    def __init__(self):
        self.ledger = Ledger()
        self.asked: list[dict] = []

    def ask(self, state, frame, questions, use):
        self.ledger.fast_calls += 1
        self.asked.append({"state": state, "questions": questions})
        out = {}
        for qid, q in questions.items():
            if q["type"] == "box":
                out[qid] = self._box(frame, "blue" in q["instructions"])
            elif q["type"] == "noul":
                out[qid] = {"noul": 0.9}
            else:
                pick = self._pick(state, q["criteria"])
                out[qid] = {"choice": pick, "confidence": 0.9, "probabilities": {pick: 0.9}}
        return out

    @staticmethod
    def _box(frame, blue):
        f = frame.astype(int)
        if blue:
            m = (f[..., 2] - f[..., 0] > 50) & (f[..., 2] - f[..., 1] > 25)
        else:
            m = (f.mean(2) > P.TOP_FACE_GRAY) & (f.max(2) - f.min(2) < 35) & ~jaw_mask(f.shape[1], f.shape[0])
        if m.sum() < 30:
            return {"box": None, "confidence": 0.1}
        vs, us = np.nonzero(m)
        h, w = m.shape
        l, r = np.percentile(us, [10, 90]) if not blue else (us.min(), us.max())
        t, b = np.percentile(vs, [10, 90]) if not blue else (vs.min(), vs.max())
        return {"box": {"left": l / w, "top": t / h, "right": r / w, "bottom": b / h}, "confidence": 0.9}

    @staticmethod
    def _pick(state, options):
        box, square = state["memory"]["white_box"], state["memory"]["blue_square"]
        jaws, over = state["arm"]["jaws"], state["arm"]["jaws_over"]
        if box.startswith("released on the blue square"):
            return "done"
        if "not located" in box or "not located" in square:
            return "look_workspace"
        if jaws.startswith("closed on something"):
            return "place" if over == "the blue square" else "carry"
        if "not measured" in box:
            return "look_box"
        return "grasp_near_top"


def test_triangulate_two_rays_meet_at_the_point():
    rot = np.eye(3)
    a = View(np.array([0.0, 0.0, 0.3]), rot, 200.0, 640, 360, 0.0)
    b = View(np.array([0.06, 0.0, 0.3]), rot, 200.0, 640, 360, 0.0)
    target = np.array([0.03, 0.01, 0.08])
    pa, pb = a.project(target), b.project(target)
    p = P.triangulate([(a, pa), (b, pb)])
    assert np.allclose(p, target, atol=1e-6)


def test_top_rect_recovers_size_and_turn():
    view = View(np.array([0.25, 0.0, 0.3]), np.eye(3), 200.0, 640, 360, 0.0)
    yaw = np.radians(20)
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    g = np.stack(np.meshgrid(np.linspace(-0.024, 0.024, 40), np.linspace(-0.024, 0.024, 40)), -1).reshape(-1, 2)
    pts = g @ R.T + [0.25, 0.02]
    px = np.array([view.project([x, y, 0.07]) for x, y in pts])
    r = P.top_rect(view, px[:, 0], px[:, 1], 0.07)
    assert abs(r["x"] - 0.25) < 0.002 and abs(r["y"] - 0.02) < 0.002
    assert 0.044 < r["da"] < 0.05 and 0.044 < r["db"] < 0.05
    assert min(abs(r["yaw"] - 20), abs(r["yaw"] - 110)) < 3


def test_skills_refuse_without_their_preconditions(tmp_path):
    jev = FakeSystemOne()
    ep = Episode(0, Config(claude=False), tmp_path, jev=jev)
    assert not S.carry(ep.ctx).ok
    assert not S.place(ep.ctx).ok
    assert "not measured" in S.grasp_near_top(ep.ctx).text


@pytest.mark.parametrize("seed", [0])
def test_an_episode_with_the_fake_decider_places_the_box(tmp_path, seed):
    jev = FakeSystemOne()
    cfg = Config(claude=False)
    out = Episode(seed, cfg, tmp_path, jev=jev).run()
    assert out["placed"] and out["finished"]
    assert out["slow_calls"] == 0 and out["fast_calls"] == jev.ledger.fast_calls
    # question ids are for code; the question text must name what it asks about
    for a in jev.asked:
        for qid, q in a["questions"].items():
            assert "`wrist`" in q["instructions"]
    assert (tmp_path / "events.jsonl").read_text().count('"kind": "chose"') >= 5
