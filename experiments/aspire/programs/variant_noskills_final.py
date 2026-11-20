import math
import numpy as np

PAN = (0.039, 0.0)
PITCHES = [90, 80, 70, 60, 50, 40, 30]
VIEWS = [((0.15, 0.0, 0.15), 45), ((0.12, 0.0, 0.12), 45), ((0.15, 0.0, 0.15), 30)]

def move_any(robot, xyz, roll=-90, start=90):
    r = None
    for p in PITCHES:
        if p > start:
            continue
        r = robot.move_to_pose(list(xyz), p, roll)
        if r["status"] != "unreachable":
            return r, p
    return r, None

def interior(loc, margin=4):
    l, t, r, b = loc["bbox"]
    return l > margin and t > margin and r < 640 - margin and b < 360 - margin

def top_face(robot, loc):
    frame = robot.observe()["frames"]["wrist"].astype(float)
    l, t, r, b = [int(v) for v in loc["bbox"]]
    l = max(l, 0); t = max(t, 0)
    sub = frame[t:b, l:r]
    if sub.size == 0:
        return None
    mn = sub.min(axis=2); mx = sub.max(axis=2); br = sub.mean(axis=2)
    ok = (mx - mn) < 30
    if ok.sum() < 20:
        return None
    thr = max(np.percentile(br[ok], 97) - 18, 150)
    mask = ok & (br >= thr)
    vs, us = np.nonzero(mask)
    if len(us) < 15:
        return None
    return us + l, vs + t

def tri(robot, p1, j1, p2, j2):
    r1 = robot.camera_ray(p1, j1); r2 = robot.camera_ray(p2, j2)
    o1, d1 = np.array(r1["origin"], float), np.array(r1["direction"], float)
    o2, d2 = np.array(r2["origin"], float), np.array(r2["direction"], float)
    d1 /= np.linalg.norm(d1); d2 /= np.linalg.norm(d2)
    w = o1 - o2
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    d, e = d1 @ w, d2 @ w
    den = a * c - b * b
    if abs(den) < 1e-6:
        return None, None
    s = (b * e - c * d) / den
    t = (a * e - b * d) / den
    q1 = o1 + s * d1; q2 = o2 + t * d2
    return (q1 + q2) / 2, float(np.linalg.norm(q1 - q2))

def min_rect(robot, us, vs, joints, h):
    idx = np.linspace(0, len(us) - 1, min(400, len(us))).astype(int)
    pts = []
    for i in idx:
        w = robot.pixel_to_world([float(us[i]), float(vs[i])], h, joints)
        if w is not None:
            pts.append(w[:2])
    if len(pts) < 10:
        return None
    pts = np.array(pts, float)
    best = None
    for a in range(0, 90, 2):
        ar = math.radians(a)
        e1 = np.array([math.cos(ar), math.sin(ar)]); e2 = np.array([-math.sin(ar), math.cos(ar)])
        p1 = pts @ e1; p2 = pts @ e2
        lo1, hi1 = np.percentile(p1, [2, 98]); lo2, hi2 = np.percentile(p2, [2, 98])
        area = (hi1 - lo1) * (hi2 - lo2)
        if best is None or area < best[0]:
            c = e1 * (lo1 + hi1) / 2 + e2 * (lo2 + hi2) / 2
            best = (area, a, hi1 - lo1, hi2 - lo2, c)
    area, a, w1, w2, c = best
    return float(a), c, w1, w2   # w1 along angle a, w2 along a+90

def roll_for(angle_close, x, y):
    phi = math.degrees(math.atan2(y - PAN[1], x - PAN[0]))
    roll = angle_close - phi
    while roll > 0: roll -= 180
    while roll < -180: roll += 180
    return roll

def perceive(robot, want_square):
    box1 = None; sq = None
    for xyz, p in VIEWS:
        if robot.move_to_pose(list(xyz), p)["status"] == "unreachable":
            continue
        box1 = robot.localize("the white box")
        if box1["found"]:
            break
        box1 = None
    if box1 is None:
        return None, None
    tf1 = top_face(robot, box1)
    if want_square:
        s = robot.localize("the blue square")
        if s["found"]:
            sq = s
    bx, by = box1["table_xy"]
    rr = math.hypot(bx - PAN[0], by)
    ux, uy = (bx - PAN[0]) / rr, by / rr
    d = max(rr - 0.10, 0.10)
    move_any(robot, [PAN[0] + ux * d, uy * d, 0.13], -90, 60)
    box2 = robot.localize("the white box")
    h = 0.05
    rect = None
    if box2["found"]:
        tf2 = top_face(robot, box2)
        if tf1 is not None and tf2 is not None:
            p1 = [float(np.mean(tf1[0])), float(np.mean(tf1[1]))]
            p2 = [float(np.mean(tf2[0])), float(np.mean(tf2[1]))]
            P, gap = tri(robot, p1, box1["joints"], p2, box2["joints"])
            robot.log(f"tri {P} gap {gap}")
            if P is not None and 0.01 < P[2] < 0.2 and gap < 0.02:
                h = float(P[2])
            rect = min_rect(robot, tf2[0], tf2[1], box2["joints"], h)
        if rect is None:
            w = robot.pixel_to_world(box2["center_px"], h * 0.5, box2["joints"])
            if w is not None:
                bx, by = w[0], w[1]
    if rect is not None:
        a, c, w1, w2 = rect
        return (float(c[0]), float(c[1]), h, a, w1, w2), sq
    return (bx, by, h, None, 0.04, 0.04), sq

def grasp(robot, est):
    bx, by, h, a, w1, w2 = est
    phi = math.degrees(math.atan2(by, bx - PAN[0]))
    if a is None:
        ang, ws = phi + 90, 0.04
    else:
        cands = [(a, w1), (a + 90, w2)]
        wmin = min(w1, w2)
        ok = [cw for cw in cands if cw[1] <= wmin * 1.25 + 0.003]
        ang, ws = min(ok, key=lambda cw: abs(roll_for(cw[0], bx, by) + 90))
    roll = roll_for(ang, bx, by)
    na = math.radians(phi + roll)
    off = 0.008 - ws / 2
    gx, gy = bx + off * math.cos(na), by + off * math.sin(na)
    robot.log(f"est {est} roll {roll:.1f} ws {ws:.3f} grasp {gx:.3f},{gy:.3f}")
    gz = max(0.012, h - 0.035)
    r, gp = move_any(robot, [gx, gy, h + 0.05], roll)
    r, gp2 = move_any(robot, [gx, gy, gz], roll, gp or 90)
    robot.log(f"descend {r['status']} {r['grasp_point']} pitch {gp2}")
    g = robot.set_gripper(0)
    robot.log(f"grip {g['status']} {g['gripper']}")
    move_any(robot, [gx, gy, gz + 0.06], roll, gp2 or 90)
    o = robot.observe()
    held = o["gripper"] > 3
    return held, roll, gz

def run(robot):
    robot.set_gripper(100)
    sq = None
    held = False
    for attempt in range(2):
        est, s = perceive(robot, sq is None)
        if s is not None and (sq is None or (interior(s) and not interior(sq))):
            sq = s
        if est is None:
            robot.log("box not found"); return
        held, roll, gz = grasp(robot, est)
        robot.log(f"held {held}")
        if held:
            break
        robot.set_gripper(100)
    if sq is None:
        for xyz, p in VIEWS:
            if robot.move_to_pose(list(xyz), p)["status"] == "unreachable":
                continue
            s = robot.localize("the blue square")
            if s["found"]:
                sq = s; break
        if sq is None:
            robot.log("no square"); return
    sx, sy = sq["table_xy"]
    robot.log(f"square {sx:.3f},{sy:.3f}")
    r, pp = move_any(robot, [sx, sy, gz + 0.07], roll)
    s2 = robot.localize("the blue square")
    if s2["found"] and interior(s2):
        nx, ny = s2["table_xy"]
        if math.hypot(nx - sx, ny - sy) < 0.05:
            sx, sy = nx, ny
            robot.log(f"square refined {sx:.3f},{sy:.3f}")
            r, pp = move_any(robot, [sx, sy, gz + 0.06], roll, pp or 90)
    r, pp2 = move_any(robot, [sx, sy, gz + 0.006], roll, pp or 90)
    robot.log(f"lower {r['status']} {r['grasp_point']}")
    robot.set_gripper(100)
    move_any(robot, [sx, sy, gz + 0.06], roll, pp2 or 90)
