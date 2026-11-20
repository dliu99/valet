"""The skill library. Each skill is code that runs closed-loop on fast perception (Jev boxes refined in code), joint
readings and the gripper's stall reading until its own test passes or it gives up, then reports what it measured.
Each also carries a plain-language done criterion that Jev judges on the frame afterwards.

Jev picks which skill runs next (with its parameter folded into the option name); no skill calls another in a fixed
order. The grasp, carry and place mechanics follow the aspire team's validated program and skills S01-S04
(branch claude/aspire-skills-1325c3)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from valet_sim.sim import START_JOINTS

from . import perception as P
from .arm import ARM_JOINTS, Arm, reach_angle_deg

OPEN = 100.0
HELD_MIN = 3.0  # a gripper reading above this after closing means something is between the jaws
HOVER_Z = 0.16
SQUARE_LOOK_Z = 0.18
LIFT_M = 0.06
DEFAULT_H = 0.075  # box top height before it is measured (boxes are 6-9 cm tall)


@dataclass
class Params:
    """What the hillclimb tunes in the skills themselves."""

    grip_below_top_m: float = 0.025
    grip_middle_frac: float = 0.5
    release_above_m: float = 0.012
    place_view_z_m: float = 0.05
    centre_tol_m: float = 0.012
    look_back_m: float = 0.06
    place_verify_z_m: float = 0.0  # h4 asked for 0.24: out of reach on every seed, so that move never ran
    place_look_back: bool = False  # h2: tried, lost 2 placements (the tip check misreads the oblique view)


@dataclass
class Memory:
    """What the policy has found out. Only perception and joint readings write here."""

    box_xy: np.ndarray | None = None
    box_h: float | None = None  # top-face height, metres; None until triangulated
    box_rect: dict | None = None  # yaw, da, db from the top face
    box_views: list = field(default_factory=list)  # (View, (u, v)) of the top face's centre
    square_xy: np.ndarray | None = None
    square_rough: bool = True
    held: dict | None = None  # roll, width, held_z, pitch
    placed_check: str = ""

    def facts(self) -> dict:
        def cm(v):
            return [round(100 * float(v[0]), 1), round(100 * float(v[1]), 1)]
        if self.placed_check:
            box = self.placed_check
        elif self.held is not None:
            box = "held in the jaws"
        elif self.box_xy is None:
            box = "not located yet"
        elif self.box_h and self.box_rect:
            box = f"measured from above: at {cm(self.box_xy)} cm, top {100 * self.box_h:.1f} cm high"
        else:
            box = f"located at about {cm(self.box_xy)} cm; height not measured yet"
        if self.square_xy is None:
            sq = "not located yet"
        else:
            sq = f"{'rough position' if self.square_rough else 'located'} at {cm(self.square_xy)} cm"
        return {"white_box": box, "blue_square": sq}


@dataclass
class Report:
    ok: bool  # the skill's own done test, in code
    text: str
    data: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Skill:
    key: str
    use_when: str  # shown to Jev as the option's meaning
    done: str  # the plain-language done criterion Jev judges afterwards
    run: Callable[["Ctx"], Report]


@dataclass
class Ctx:
    arm: Arm
    jev: object
    mem: Memory
    params: Params
    log: Callable[[str, dict], None]


# ---- shared pieces -------------------------------------------------------------------------------------------------

def jaws_state(arm: Arm) -> str:
    g = arm.gripper()
    goal = float(np.degrees(arm.sim.goal[-1]))
    if goal > 40:
        return "open and empty"
    return f"closed on something (reading {g:.0f})" if g > HELD_MIN else "closed on nothing"


def holding(arm: Arm) -> bool:
    return float(np.degrees(arm.sim.goal[-1])) < 40 and arm.gripper() > HELD_MIN


def go_overview(arm: Arm) -> dict:
    goals = {j: START_JOINTS[j] for j in ARM_JOINTS}
    arm.sim.set_goals(goals)
    status = arm._wait(lambda j: all(abs(j[k] - goals[k]) < 0.5 for k in ARM_JOINTS), ARM_JOINTS, 4.0)
    return {"status": status}


def _note_box(ctx: Ctx, s: P.Sighting, overhead: bool) -> dict | None:
    """Store a top-face sighting; triangulate the top's centre from every view so far (least squares); from an
    overhead view, fit the top rectangle at that height for the box's turn and size."""
    m, arm = ctx.mem, ctx.arm
    view = arm.view(s.joints)
    m.box_views.append((view, s.centre_px))
    spread = max((P.baseline(a, b) for (a, _), (b, _) in zip(m.box_views, m.box_views[1:])), default=0.0)
    p = P.triangulate(m.box_views) if spread > 0.04 else None
    if p is not None and 0.03 < p[2] < 0.12:
        m.box_h = float(p[2])
    h = m.box_h or DEFAULT_H
    rect = P.top_rect(view, s.us, s.vs, h) if overhead else None
    if rect is not None and not (0.025 < min(rect["da"], rect["db"]) and max(rect["da"], rect["db"]) < 0.08):
        rect = None
    if rect is not None and m.box_h is not None:
        m.box_rect = rect
        m.box_xy = np.array([rect["x"], rect["y"]])
    elif p is not None and m.box_h is not None:
        m.box_xy = p[:2].copy()
    else:
        c = view.to_plane(*s.centre_px, h)
        if c is not None:
            m.box_xy = c[:2]
    ctx.log("box_estimate", {"xy": None if m.box_xy is None else m.box_xy.round(4).tolist(), "h": m.box_h,
                             "tri": None if p is None else p.round(4).tolist(), "rect": rect, "views": len(m.box_views)})
    return rect


def _note_square(ctx: Ctx, s: P.Sighting, strict: bool) -> str:
    m, arm = ctx.mem, ctx.arm
    view = arm.view(s.joints)
    c, why = P.square_centre(view, s)
    if c is None:
        if m.square_xy is None and s.bbox is not None:  # a rough fix from the Jev box alone
            l, t, r, b = s.bbox
            hit = view.to_plane((l + r) / 2, (t + b) / 2, 0.0)
            if hit is not None:
                m.square_xy, m.square_rough = hit[:2], True
        return why
    if strict and m.square_xy is not None and not m.square_rough and np.linalg.norm(c - m.square_xy) > 0.025:
        return f"rejected a square sighting {100 * np.linalg.norm(c - m.square_xy):.1f} cm from the clear view"
    m.square_xy, m.square_rough = c, False
    ctx.log("square_estimate", {"xy": c.round(4).tolist()})
    return "ok"


def _wrap(a: float) -> float:
    return (a + 180) % 360 - 180


def _grasp_choices(yaw: float, da: float, db: float, phi: float) -> list[tuple[float, float, float]]:
    out = []
    for k in range(-4, 5):
        for ang, w in ((yaw, da), (yaw + 90, db)):
            rr = _wrap(ang - phi + k * 180)
            if -170 <= rr <= 10:
                out.append((w + 0.00012 * abs(rr + 90), rr, w))
    return sorted(out)


# ---- the skills ----------------------------------------------------------------------------------------------------

def look_workspace(ctx: Ctx) -> Report:
    arm, m = ctx.arm, ctx.mem
    if holding(arm):
        return Report(False, "not run: the jaws are holding the box; looking around now would only show the box")
    arm.grip(60)
    go_overview(arm)
    seen = P.look(arm, ctx.jev, ("box", "square"), "look_workspace")
    notes = []
    if seen["box"].ok:
        _note_box(ctx, seen["box"], overhead=False)
        notes.append("found the white box")
    else:
        notes.append(f"white box not found ({seen['box'].note})")
    why = _note_square(ctx, seen["square"], strict=False)
    notes.append("found the blue square" if why == "ok" else f"blue square: {why}")
    ok = seen["box"].ok and m.square_xy is not None
    return Report(ok, "; ".join(notes))


def look_box(ctx: Ctx) -> Report:
    """Two looks down at the box from behind it, 7 cm apart side to side: the top face's centre is triangulated
    across all views for its height, and each overhead view fits the top rectangle at that height."""
    arm, m, p = ctx.arm, ctx.mem, ctx.params
    if holding(arm):
        return Report(False, "not run: the jaws are already holding the box")
    if m.box_xy is None:
        r = look_workspace(ctx)
        if m.box_xy is None:
            return Report(False, "the white box is nowhere in view; " + r.text)
    arm.grip(OPEN)
    rects, misses = [], []
    for attempt in range(2):
        bx, by = m.box_xy
        phi = math.radians(reach_angle_deg(bx, by))
        u, perp = np.array([math.cos(phi), math.sin(phi)]), np.array([-math.sin(phi), math.cos(phi)])
        for lateral in (-0.035, 0.035):
            q = m.box_xy - p.look_back_m * u + lateral * perp
            mv = arm.move([q[0], q[1], HOVER_Z], [90, 85, 78, 70, 60, 50])
            if mv["status"] == "unreachable":
                misses.append("a viewpoint was out of reach")
                continue
            seen = P.look(arm, ctx.jev, ("box",), "look_box")["box"]
            if not seen.ok:
                misses.append(seen.note)
                continue
            rect = _note_box(ctx, seen, overhead=True)
            if rect is not None:
                rects.append(rect)
        if m.box_h is not None and len(rects) >= 2:
            a, b = rects[-2], rects[-1]
            if math.hypot(a["x"] - b["x"], a["y"] - b["y"]) < p.centre_tol_m:
                r = m.box_rect
                return Report(True, f"measured the box from above: top {100 * m.box_h:.1f} cm high, top face "
                                    f"{100 * r['da']:.1f} x {100 * r['db']:.1f} cm, turned {r['yaw']:.0f} deg",
                              {"attempts": attempt + 1})
    if m.box_rect is not None and m.box_h is not None:
        return Report(True, f"measured the box, but two views disagree by over {1000 * p.centre_tol_m:.0f} mm; "
                            f"top {100 * m.box_h:.1f} cm high")
    return Report(False, "could not measure the box from above" + (f" ({misses[-1]})" if misses else ""))


def look_square(ctx: Ctx) -> Report:
    arm, m = ctx.arm, ctx.mem
    held = holding(arm)
    if m.square_xy is None and not held:
        look_workspace(ctx)
    if m.square_xy is None:
        return Report(False, "the blue square is not located; look at the workspace first")
    z = (m.held["held_z"] + LIFT_M + 0.04) if held and m.held else SQUARE_LOOK_Z
    roll = m.held["roll"] if held and m.held else -90.0
    phi = math.radians(reach_angle_deg(*m.square_xy))
    q = m.square_xy - (0.0 if held else ctx.params.look_back_m) * np.array([math.cos(phi), math.sin(phi)])
    mv = arm.move([q[0], q[1], z], [90, 80, 70, 60, 50, 40], roll)
    if mv["status"] == "unreachable":
        return Report(False, "cannot look down on the square from here")
    seen = P.look(arm, ctx.jev, ("square",), "look_square")["square"]
    why = _note_square(ctx, seen, strict=held)
    return Report(why == "ok", "the whole blue square is in view and measured" if why == "ok" else why)


def _grasp(ctx: Ctx, middle: bool) -> Report:
    arm, m, p = ctx.arm, ctx.mem, ctx.params
    if holding(arm):
        return Report(False, "not run: the jaws already hold something")
    if m.box_xy is None or m.box_h is None or m.box_rect is None:
        return Report(False, "not run: the box's height and turn are not measured yet; look at the box from above first")
    h, r = m.box_h, m.box_rect
    bx, by = m.box_xy
    phi = reach_angle_deg(bx, by)
    choices = _grasp_choices(r["yaw"], r["da"], r["db"], phi)
    if not choices:
        return Report(False, "no wrist roll lines the jaws up with the box")
    _, roll, width = choices[0]
    gz = max(0.012, h * p.grip_middle_frac) if middle else max(0.012, h - p.grip_below_top_m)
    L = np.array([math.cos(math.radians(phi + roll + 180)), math.sin(math.radians(phi + roll + 180))])
    g = np.array([bx, by]) + (min(width, 0.06) / 2 - 0.008) * L
    arm.grip(OPEN)
    arm.move([g[0], g[1], h + 0.04], [90, 85, 78, 70, 60], roll)
    down = arm.move([g[0], g[1], gz], [90, 84, 78, 72], roll)
    ctx.log("grasp_descend", {"g": g.round(4).tolist(), "gz": round(gz, 4), "roll": roll, "status": down["status"],
                              "z": round(float(down["grasp_point"][2]), 4), "pitch": round(down["pitch"], 1)})
    if down["status"] == "unreachable":
        return Report(False, "the grip pose is out of reach")
    if down["status"] in ("stopped", "timeout") and down["grasp_point"][2] > gz + 0.015:
        arm.move([g[0], g[1], h + 0.06], [85, 78, 70, 60], roll)
        stale = m.box_xy.copy()
        m.box_rect = None  # the contact may have moved or turned it
        m.box_views.clear()
        m.box_xy = stale
        return Report(False, f"blocked: the jaws stopped {100 * (down['grasp_point'][2] - gz):.1f} cm above the grip height, "
                             "on top of the box; it may have moved, so measure it again from above")
    close = arm.grip(0)
    if close["gripper"] <= HELD_MIN:
        arm.grip(OPEN)
        arm.move([g[0], g[1], h + 0.06], [85, 78, 70, 60], roll)
        return Report(False, "the jaws closed on nothing: the box was not between them")
    held_z, pitch = float(down["grasp_point"][2]), down["pitch"]
    arm.move([g[0], g[1], held_z + LIFT_M], [80, 70, 60], roll)
    if not holding(arm):
        arm.grip(OPEN)
        return Report(False, f"the box slipped out while lifting (reading {arm.gripper():.0f})")
    m.held = {"roll": roll, "width": width, "held_z": held_z, "pitch": pitch, "g": g}
    m.placed_check = ""
    return Report(True, f"holding the box, lifted {100 * LIFT_M:.0f} cm (reading {arm.gripper():.0f}), gripped "
                        f"{100 * (h - held_z):.1f} cm below its top")


def grasp_near_top(ctx: Ctx) -> Report:
    return _grasp(ctx, middle=False)


def grasp_middle(ctx: Ctx) -> Report:
    return _grasp(ctx, middle=True)


def _drop_point(m: Memory) -> np.ndarray:
    sx, sy = m.square_xy
    roll, width = m.held["roll"], m.held["width"]
    L2 = math.radians(reach_angle_deg(sx, sy) + roll + 180)
    return np.array([sx, sy]) - (0.02 - min(width, 0.06) / 2) * np.array([math.cos(L2), math.sin(L2)])


def carry(ctx: Ctx) -> Report:
    arm, m = ctx.arm, ctx.mem
    if not holding(arm) or m.held is None:
        return Report(False, "not run: the jaws are not holding the box")
    if m.square_xy is None:
        return Report(False, "not run: the blue square has not been located")
    tgt = _drop_point(m)
    mv = arm.move([tgt[0], tgt[1], m.held["held_z"] + LIFT_M], [80, 70, 60], m.held["roll"])
    if mv["status"] == "unreachable":
        return Report(False, "the square is out of reach with the box held this way")
    seen = P.look(arm, ctx.jev, ("square",), "carry")["square"]
    why = _note_square(ctx, seen, strict=True)
    if not holding(arm):
        return Report(False, "the box fell out of the jaws on the way")
    off = np.linalg.norm(mv["grasp_point"][:2] - _drop_point(m))
    return Report(off < 0.02, f"holding the box above the square ({100 * off:.1f} cm from the drop point)"
                  + ("" if why == "ok" else f"; square re-check: {why}"))


def place(ctx: Ctx) -> Report:
    arm, m, p = ctx.arm, ctx.mem, ctx.params
    if not holding(arm) or m.held is None:
        return Report(False, "not run: the jaws are not holding the box")
    if m.square_xy is None:
        return Report(False, "not run: the blue square has not been located")
    tgt = _drop_point(m)
    gp, _ = arm.pose()
    if np.linalg.norm(gp[:2] - tgt) > 0.03:
        return Report(False, f"not run: the box is {100 * np.linalg.norm(gp[:2] - tgt):.0f} cm from above the square; carry it first")
    roll, held_z = m.held["roll"], m.held["held_z"]
    r = arm.move([tgt[0], tgt[1], held_z + p.release_above_m], [90, 84, 78, 72], roll)
    gp, pitch = r["grasp_point"], r["pitch"]
    if r["status"] in ("stopped", "timeout"):
        r = arm.move([tgt[0], tgt[1], gp[2] + 0.012], [pitch], roll)
        gp = r["grasp_point"]
    before = arm.gripper()
    o = arm.grip(OPEN)
    for _ in range(2):
        if o["gripper"] >= before + 25:
            break
        r = arm.move([tgt[0], tgt[1], gp[2] + 0.008], [pitch], roll)
        gp = r["grasp_point"]
        o = arm.grip(OPEN)
    arm.move([tgt[0], tgt[1], max(held_z, gp[2]) + p.place_view_z_m], [80, 70, 60], roll)
    m.held = None
    if p.place_look_back:  # step back and up so the frame shows the box standing on the square
        phi = math.radians(reach_angle_deg(*m.square_xy))
        q = m.square_xy - p.look_back_m * np.array([math.cos(phi), math.sin(phi)])
        arm.move([q[0], q[1], SQUARE_LOOK_Z], [90, 80, 70, 60, 50, 40])
    opened = o["gripper"] >= before + 25
    # look down at what was left behind
    seen = P.look(arm, ctx.jev, ("box", "square"), "place_check")
    msg = []
    ok = opened
    if seen["box"].ok and m.square_xy is not None:
        view = arm.view(seen["box"].joints)
        c = view.to_plane(*seen["box"].centre_px, m.box_h or DEFAULT_H)
        rect = P.top_rect(view, seen["box"].us, seen["box"].vs, m.box_h or DEFAULT_H)
        if c is not None:
            off = float(np.linalg.norm(c[:2] - m.square_xy))
            msg.append(f"box top seen {100 * off:.1f} cm from the square's centre")
            ok = ok and off < 0.03
        if rect is not None:
            long, short = max(rect["da"], rect["db"]), min(rect["da"], rect["db"])
            if long > 1.3 * short:
                msg.append(f"its top looks {100 * short:.0f} x {100 * long:.0f} cm, not square: it may be lying on its side")
                ok = False
    else:
        msg.append("could not see the box afterwards")
    if p.place_verify_z_m:  # rise straight up over the square: the view Jev judges "standing on the square" from
        arm.move([m.square_xy[0], m.square_xy[1], p.place_verify_z_m], [90, 85, 80, 70, 60, 50])
    text = ("released the box" if opened else "the jaws did not open fully") + "; " + "; ".join(msg)
    if opened:
        m.placed_check = ("released on the blue square: " if ok else "released, but not cleanly on the blue square: ") + "; ".join(msg)
    return Report(ok, text)


def back_off(ctx: Ctx) -> Report:
    arm, m = ctx.arm, ctx.mem
    arm.grip(OPEN)
    gp, _ = arm.pose()
    arm.move([gp[0], gp[1], max(gp[2] + 0.06, 0.15)], [80, 70, 60, 50])
    m.held = None
    return Report(not holding(arm), "jaws open and raised clear")


SKILLS: dict[str, Skill] = {s.key: s for s in [
    Skill("look_workspace", "When `memory` says the white box or the blue square is not located yet.",
          "Both the white box and the blue square are visible in `wrist`.", look_workspace),
    Skill("look_box", "When the jaws are empty and `memory` says the white box's height is not measured yet.",
          "`wrist` looks down on the white box from above, with the whole box in view.", look_box),
    Skill("look_square", "When `memory` says the blue square's position is only rough.",
          "The whole blue square is in view in `wrist`.", look_square),
    Skill("grasp_near_top", "When the jaws are empty and `memory` says the white box was measured from above: "
          "grip it about 2.5 cm below its top and lift it.",
          "The jaws are holding the white box up off the table.", grasp_near_top),
    Skill("grasp_middle", "When the jaws are empty and `memory` says the white box was measured from above: "
          "grip it halfway up its height and lift it.",
          "The jaws are holding the white box up off the table.", grasp_middle),
    Skill("carry", "When `arm` says the jaws are closed on something and they are not above the blue square yet.",
          "The jaws are holding the white box above the blue square.", carry),
    Skill("place", "When `arm` says the jaws are closed on something and they are above the blue square.",
          "The white box is standing on the blue square.", place),
    Skill("back_off", "When `arm` says the jaws are closed on nothing, or a grasp was just blocked on top of the box.",
          "The jaws are open and raised above the table, clear of the white box.", back_off),
]}
# h3: back_off was chosen 68 times in h0-h2 and never right; the best config leaves it out of Jev's menu
DEFAULT_SKILLS = ("look_workspace", "look_box", "look_square", "grasp_near_top", "grasp_middle", "carry", "place")
DONE_KEY = "done"
DONE_USE = "When `memory` says the white box was released on the blue square: stop, the task is finished."
FINAL_CRITERION = "The white box is standing on the blue square."
