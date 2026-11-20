
import math
import numpy as np

def log(robot, s):
    robot.log(str(s))

def move(robot, xyz, pitches, roll=-90, timeout=4):
    r = None
    for p in pitches:
        r = robot.move_to_pose(list(xyz), p, roll, timeout_s=timeout)
        if r['status'] != 'unreachable':
            return r
    return r

def ray_meet(robot, px1, j1, px2, j2):
    a = robot.camera_ray(px1, j1); b = robot.camera_ray(px2, j2)
    o1 = np.array(a['origin']); d1 = np.array(a['direction']); d1 = d1/np.linalg.norm(d1)
    o2 = np.array(b['origin']); d2 = np.array(b['direction']); d2 = d2/np.linalg.norm(d2)
    w = o1 - o2
    aa = d1@d1; bb = d1@d2; cc = d2@d2; dd = d1@w; ee = d2@w
    den = aa*cc - bb*bb
    if abs(den) < 1e-9:
        return None
    s = (bb*ee - cc*dd)/den; t = (aa*ee - bb*dd)/den
    return 0.5*((o1 + s*d1) + (o2 + t*d2))

def top_mask(frame, bbox, drop=18):
    H, W = frame.shape[:2]
    l, t, r, b = [int(round(x)) for x in bbox]
    l = max(0, l); t = max(0, t); r = min(W, r); b = min(H, b)
    crop = frame[t:b, l:r].astype(float)
    gray = crop.mean(axis=2)
    sat = crop.max(axis=2) - crop.min(axis=2)
    p97 = np.percentile(gray, 97)
    mask = (gray > p97 - drop) & (sat < 40)
    vs, us = np.nonzero(mask)
    return us + l, vs + t

def top_rect(robot, us, vs, h, joints):
    n = len(us)
    if n < 20:
        return None
    step = max(1, n // 1500)
    pts = []
    for i in range(0, n, step):
        w = robot.pixel_to_world([float(us[i]), float(vs[i])], h, joints)
        if w is not None:
            pts.append([w[0], w[1]])
    pts = np.array(pts)
    if len(pts) < 20:
        return None
    best = None
    for deg in range(0, 90):
        th = math.radians(deg)
        c, s = math.cos(th), math.sin(th)
        x = pts[:, 0]*c + pts[:, 1]*s
        y = -pts[:, 0]*s + pts[:, 1]*c
        x0, x1 = np.percentile(x, [1, 99]); y0, y1 = np.percentile(y, [1, 99])
        area = (x1 - x0)*(y1 - y0)
        if best is None or area < best[0]:
            xm = 0.5*(x0 + x1); ym = 0.5*(y0 + y1)
            cx = xm*c - ym*s; cy = xm*s + ym*c
            best = (area, deg, x1 - x0, y1 - y0, cx, cy)
    return best

def wrap(a):
    while a > 180: a -= 360
    while a <= -180: a += 360
    return a

def grasp_cands(yaw, da, db, phi):
    cands = []
    for k in range(-4, 5):
        for (ang, w) in ((yaw, da), (yaw + 90, db)):
            rr = wrap(ang - phi + k*180)
            if -170 <= rr <= 10:
                cands.append((w + 0.00012*abs(rr + 90), rr, w))
    cands.sort()
    return cands

def remeasure(robot, bx, by, h):
    robot.set_gripper(100)
    move(robot, [bx, by, 0.16], [70, 60])
    b3 = robot.localize("the white box")
    if not b3['found']:
        return None
    fr = robot.observe()['frames']['wrist']
    q = rects_from(robot, fr, b3['bbox'], h, b3['joints'])
    if q is None:
        c = robot.pixel_to_world(b3['center_px'], h, b3['joints'])
        return (c[0], c[1], None, None, None)
    _, y2, a2, d2, cx, cy = q
    log(robot, f"remeasure rect yaw {y2} dims {a2:.3f} {d2:.3f} c {cx:.3f} {cy:.3f}")
    return (cx, cy, y2, a2, d2)

def try_grasp(robot, bx, by, h, yaw, da, db):
    phi = math.degrees(math.atan2(by, bx - 0.039))
    cands = grasp_cands(yaw, da, db, phi)
    gz = max(0.012, h - 0.025)
    robot.set_gripper(100)
    choice = cands[0]
    r = None
    for attempt in range(2):
        _, roll, width = choice
        log(robot, f"try {attempt} phi {phi:.1f} roll {roll:.1f} width {width:.3f}")
        Lang = math.radians(phi + roll + 180)
        L = np.array([math.cos(Lang), math.sin(Lang)])
        off = (min(width, 0.06)/2 - 0.008)
        g = np.array([bx, by]) + off*L
        r = move(robot, [g[0], g[1], h + 0.04], [90, 85, 78, 70, 60], roll)
        log(robot, f"pre {r['status']} {r['grasp_point']} p {r['pitch_deg']}")
        r = move(robot, [g[0], g[1], gz], [90, 84, 78, 72], roll)
        log(robot, f"down {r['status']} {r['grasp_point']} p {r['pitch_deg']}")
        if r['status'] in ('stopped', 'timeout') and r['grasp_point'][2] > gz + 0.015 and attempt == 0:
            move(robot, [g[0], g[1], h + 0.06], [85, 78, 70, 60], roll)
            m = remeasure(robot, bx, by, h)
            if m is not None:
                bx, by = m[0], m[1]
                if m[2] is not None:
                    yaw, da, db = m[2], m[3], m[4]
                phi = math.degrees(math.atan2(by, bx - 0.039))
                cands = grasp_cands(yaw, da, db, phi)
            alts = [c for c in cands if abs(wrap(c[1] - roll)) > 45]
            if not alts:
                break
            choice = alts[0]
            robot.set_gripper(100)
            continue
        break
    gr = robot.set_gripper(0)
    log(robot, f"grip {gr}")
    held_z = r['grasp_point'][2]
    r = move(robot, [g[0], g[1], held_z + 0.06], [80, 70, 60], roll)
    ob = robot.observe()
    held = ob['gripper'] > 3
    log(robot, f"lift {r['status']} {r['grasp_point']} gripper {ob['gripper']} held {held}")
    return held, roll, width, g, held_z, bx, by, yaw, da, db

def rects_from(robot, frame, bbox, h, joints):
    u, v = top_mask(frame, bbox)
    return top_rect(robot, u, v, h, joints)

def run(robot):
    robot.set_gripper(40)
    ob = robot.observe()
    box = robot.localize("the white box")
    sq = robot.localize("the blue square")
    log(robot, f"box {box['bbox']} sq {sq['bbox']}")
    if not box['found']:
        return
    j1 = box['joints']
    u1, v1 = top_mask(ob['frames']['wrist'], box['bbox'])
    t1 = [float(np.mean(u1)), float(np.mean(v1))] if len(u1) > 10 else box['center_px']
    est = robot.pixel_to_world(t1, 0.06, j1)
    bx, by = est[0], est[1]
    sx = sy = None
    if sq['found']:
        s0 = robot.pixel_to_world(sq['center_px'], 0.0, sq['joints'])
        sx, sy = s0[0], s0[1]
    r = move(robot, [bx, by, 0.16], [70, 60])
    log(robot, f"hover {r['status']} {r['grasp_point']} pitch {r['pitch_deg']}")
    box2 = robot.localize("the white box")
    h = 0.06
    yaw = 0.0; da = db = 0.035
    if box2['found']:
        j2 = box2['joints']
        frame = robot.observe()['frames']['wrist']
        u2, v2 = top_mask(frame, box2['bbox'])
        if len(u2) > 10:
            t2 = [float(np.mean(u2)), float(np.mean(v2))]
            p = ray_meet(robot, t1, j1, t2, j2)
            if p is not None:
                h = float(min(0.15, max(0.02, p[2])))
                log(robot, f"top centre {p.tolist()} h {h}")
            rects = []
            ra = top_rect(robot, u2, v2, h, j2)
            if ra is not None: rects.append(ra)
            rb = top_rect(robot, u1, v1, h, j1)
            if rb is not None: rects.append(rb)
            for q in rects:
                log(robot, f"rect cand yaw {q[1]} dims {q[2]:.3f} {q[3]:.3f} c {q[4]:.3f} {q[5]:.3f}")
            if rects:
                rects.sort(key=lambda q: q[0])
                _, yaw, da, db, cx, cy = rects[0]
                bx, by = cx, cy
    held, roll, width, g, held_z, bx, by, yaw, da, db = try_grasp(robot, bx, by, h, yaw, da, db)
    if not held:
        m = remeasure(robot, bx, by, h)
        if m is not None:
            bx, by = m[0], m[1]
            if m[2] is not None:
                y2, a2, d2 = m[2], m[3], m[4]
                diff = abs(((y2 - yaw) + 45) % 90 - 45)
                if diff < 15:
                    y2 = yaw + 45
                    a2 = d2 = max(a2, d2) * 0.8
                yaw, da, db = y2, a2, d2
            else:
                yaw = yaw + 45
            held, roll, width, g, held_z, bx, by, yaw, da, db = try_grasp(robot, bx, by, h, yaw, da, db)
    if sx is None:
        return
    phi2 = math.degrees(math.atan2(sy, sx - 0.039))
    L2ang = math.radians(phi2 + roll + 180)
    L2 = np.array([math.cos(L2ang), math.sin(L2ang)])
    ho = (0.02 - min(width, 0.06)/2)
    tgt = np.array([sx, sy]) - ho*L2
    r = move(robot, [tgt[0], tgt[1], held_z + 0.06], [80, 70, 60], roll)
    log(robot, f"carry {r['status']} {r['grasp_point']}")
    sq2 = robot.localize("the blue square")
    if sq2['found']:
        bb = sq2['bbox']
        wpx = bb[2] - bb[0]; hpx = bb[3] - bb[1]
        if bb[0] > 2 and bb[1] > 2 and bb[2] < 638 and bb[3] < 358 and wpx > 30 and 0.7 < hpx / wpx < 1.4:
            c = robot.pixel_to_world(sq2['center_px'], 0.0, sq2['joints'])
            if math.hypot(c[0] - sx, c[1] - sy) < 0.025:
                log(robot, f"square refined {c} vs {sx},{sy}")
                sx, sy = c[0], c[1]
                tgt = np.array([sx, sy]) - ho*L2
    r = move(robot, [tgt[0], tgt[1], held_z + 0.012], [90, 84, 78, 72], roll)
    log(robot, f"place {r['status']} {r['grasp_point']} p {r['pitch_deg']}")
    gp = r['grasp_point']; pp = r['pitch_deg']
    if r['status'] in ('stopped', 'timeout'):
        r2 = move(robot, [tgt[0], tgt[1], gp[2] + 0.012], [pp], roll)
        gp = r2['grasp_point']
        log(robot, f"ease {r2['status']} {gp}")
    before = robot.observe()['gripper']
    o = robot.set_gripper(100)
    log(robot, f"open {o}")
    tries = 0
    while o['gripper'] < before + 25 and tries < 2:
        tries += 1
        r3 = move(robot, [tgt[0], tgt[1], gp[2] + 0.008], [pp], roll)
        gp = r3['grasp_point']
        o = robot.set_gripper(100)
        log(robot, f"reopen {tries} {r3['status']} {gp} {o}")
    r = move(robot, [tgt[0], tgt[1], max(held_z, gp[2]) + 0.05], [80, 70, 60], roll)
