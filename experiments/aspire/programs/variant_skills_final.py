import numpy as np, math

PL = [90, 85, 80, 75, 70, 65, 60, 55, 50]

def move(robot, xyz, pref=90, roll=-90, t=4, lo=50):
    order = sorted([p for p in PL if p >= lo], key=lambda p: (abs(p - pref), -p))
    r = None
    for p in order:
        r = robot.move_to_pose([float(v) for v in xyz], float(p), float(roll), timeout_s=t)
        if r['status'] != 'unreachable':
            return r
    return r

def top_px(frame, bbox):
    l, t, r, b = [int(round(x)) for x in bbox]
    l = max(l, 0); t = max(t, 0); r = min(r, 640); b = min(b, 360)
    c = frame[t:b, l:r].astype(float)
    g = c.mean(2); s = c.max(2) - c.min(2)
    vs, us = np.nonzero((g > np.percentile(g, 95) - 25) & (s < 40))
    return us + l, vs + t

def meet(robot, p1, j1, p2, j2):
    a = robot.camera_ray(p1, j1); b = robot.camera_ray(p2, j2)
    o1, d1 = np.array(a['origin'], float), np.array(a['direction'], float)
    o2, d2 = np.array(b['origin'], float), np.array(b['direction'], float)
    d1 /= np.linalg.norm(d1); d2 /= np.linalg.norm(d2)
    w = o1 - o2; bb = d1 @ d2; den = 1 - bb * bb
    if den < 1e-6:
        return None
    s = (bb * (d2 @ w) - (d1 @ w)) / den
    t = ((d2 @ w) - bb * (d1 @ w)) / den
    return 0.5 * (o1 + s * d1 + o2 + t * d2)

def wrap(a):
    return (a + 180.0) % 360.0 - 180.0

def footprint(robot, us, vs, h, joints):
    n = len(us)
    if n < 20:
        return None
    idx = np.linspace(0, n - 1, min(n, 400)).astype(int)
    pts = []
    for i in idx:
        p = robot.pixel_to_world([float(us[i]), float(vs[i])], h, joints)
        if p is not None:
            pts.append(p[:2])
    if len(pts) < 20:
        return None
    P = np.array(pts); m = P.mean(0); Q = P - m
    best = None
    for ang in range(0, 90):
        a = math.radians(ang)
        ea = np.array([math.cos(a), math.sin(a)]); eb = np.array([-ea[1], ea[0]])
        pa = Q @ ea; pb = Q @ eb
        a0, a1 = np.percentile(pa, [2, 98]); b0, b1 = np.percentile(pb, [2, 98])
        area = (a1 - a0) * (b1 - b0)
        if best is None or area < best[0]:
            cen = m + ea * (a0 + a1) / 2 + eb * (b0 + b1) / 2
            best = (area, cen, float(ang), a1 - a0, b1 - b0)
    _, cen, yaw, da, db = best
    return cen, yaw, da, db

def plan(cen, yaw, da, db, axis):
    if (da <= db) == (axis == 0):
        theta, wdt = yaw, da
    else:
        theta, wdt = yaw + 90, db
    phi = math.degrees(math.atan2(cen[1], cen[0] - 0.039))
    r0 = wrap(theta - phi)
    roll = min([r0, r0 + 180, r0 - 180], key=lambda x: abs(x + 90))
    ca = math.radians(phi + roll)
    cvec = np.array([math.cos(ca), math.sin(ca)])
    g = cen - (wdt / 2 + 0.009 - 0.02) * cvec
    return roll, wdt, g

def run(robot):
    robot.set_gripper(100)
    ob = robot.observe(); fr1 = ob['frames']['wrist']
    box = robot.localize("the white box")
    sq = robot.localize("the blue square")
    s0 = robot.pixel_to_world(sq['center_px'], 0.0, sq['joints'])
    sx, sy = s0[0], s0[1]
    b0 = robot.pixel_to_world(box['center_px'], 0.03, box['joints'])
    bx, by = b0[0], b0[1]
    u1, v1 = top_px(fr1, box['bbox'])
    t1 = [float(u1.mean()), float(v1.mean())]
    robot.log(f"square0 {sx:.3f},{sy:.3f} box0 {bx:.3f},{by:.3f}")

    gp0 = ob['grasp_point']
    sgn = -1.0 if by < 0 else 1.0
    move(robot, [gp0[0] - 0.02, gp0[1] + 0.07 * sgn, 0.15], pref=60)
    ob2 = robot.observe(); fr2 = ob2['frames']['wrist']
    box2 = robot.localize("the white box")
    h = None; fp = None; ptri = None
    if box2['found']:
        u2, v2 = top_px(fr2, box2['bbox'])
        p = meet(robot, t1, box['joints'], [float(u2.mean()), float(v2.mean())], box2['joints'])
        robot.log(f"tri {p}")
        if p is not None and 0.008 < p[2] < 0.16:
            h = float(p[2]); ptri = np.array(p[:2])
            fp = footprint(robot, u2, v2, h, box2['joints'])
    if h is None:
        h = 0.04
        robot.log("height fallback 0.04")
        fp = footprint(robot, u1, v1, h, box['joints'])
    if fp is not None:
        cen, yaw, da, db = fp
    else:
        cen = np.array([bx, by]); yaw = 0.0; da = db = 0.04
    robot.log(f"view2 cen {cen} yaw {yaw:.1f} da {da:.3f} db {db:.3f} h {h:.3f}")

    r = move(robot, [cen[0], cen[1], h + 0.10], pref=60, lo=55)
    if r['status'] != 'unreachable':
        ob3 = robot.observe(); fr3 = ob3['frames']['wrist']
        box3 = robot.localize("the white box")
        if box3['found']:
            u3, v3 = top_px(fr3, box3['bbox'])
            fp3 = footprint(robot, u3, v3, h, box3['joints'])
            if fp3 is not None:
                robot.log(f"view3 cen {fp3[0]} yaw {fp3[1]:.1f} da {fp3[2]:.3f} db {fp3[3]:.3f}")
                if 0.012 < fp3[2] < 0.09 and 0.012 < fp3[3] < 0.09 and np.linalg.norm(fp3[0] - cen) < 0.04:
                    cen, yaw, da, db = fp3
    if ptri is not None and np.linalg.norm(ptri - cen) < 0.03:
        cen = 0.5 * (cen + ptri)
    robot.log(f"final cen {cen} yaw {yaw:.1f} da {da:.3f} db {db:.3f}")

    gz = max(0.012, h - 0.025)
    held = False
    for attempt in range(2):
        roll, wdt, g = plan(cen, yaw, da, db, attempt)
        robot.log(f"attempt {attempt} roll {roll:.1f} wdt {wdt:.3f} g {g} gz {gz:.3f}")
        robot.set_gripper(100)
        move(robot, [g[0], g[1], h + 0.04], pref=90, roll=roll)
        r = move(robot, [g[0], g[1], gz], pref=90, roll=roll)
        gp = r['pitch_deg']
        robot.log(f"descend {r['status']} {r['grasp_point']}")
        if r['grasp_point'][2] > gz + 0.015:
            move(robot, [g[0], g[1], h + 0.05], pref=gp, roll=roll)
            continue
        rg = robot.set_gripper(0)
        robot.log(f"grip {rg['status']} {rg['gripper']} {rg['amps']}")
        if rg['gripper'] > 3:
            held = True
            break
        move(robot, [g[0], g[1], h + 0.05], pref=gp, roll=roll)
    if not held:
        return
    lz = gz + 0.05
    move(robot, [g[0], g[1], lz], pref=gp, roll=roll)

    off = (wdt / 2 - 0.02)
    phi2 = math.degrees(math.atan2(sy, sx - 0.039))
    ca2 = math.radians(phi2 + roll)
    tgt = np.array([sx, sy]) - off * np.array([math.cos(ca2), math.sin(ca2)])
    move(robot, [tgt[0], tgt[1], lz], pref=gp, roll=roll)
    r = move(robot, [tgt[0], tgt[1], gz + 0.004], pref=gp, roll=roll)
    robot.log(f"grasp pitch {gp:.1f} place pitch {r['pitch_deg']:.1f} {r['status']}")
    robot.set_gripper(60)
    move(robot, [tgt[0], tgt[1], h + 0.05], pref=r['pitch_deg'], roll=roll)
