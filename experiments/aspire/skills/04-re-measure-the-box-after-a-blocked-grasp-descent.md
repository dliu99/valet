---
id: "S04"
task: "benchmark"
run: "main"
round: 1
title: "Re-measure the box after a blocked grasp descent, before retrying at another roll"
before: "r0a2"
after: "r1a0"
fixed: {"107": "placed"}
verifier: "r1v1"
reproduced: ["107"]
verifier_placed: 10
before_seeds: [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
verifier_fitness_on_before_seeds: 10.0
before_fitness: 9.395
---

# Re-measure the box after a blocked grasp descent, before retrying at another roll

## Failure signature
- The grasp descent `move_to_pose` returns `stopped`. Its `grasp_point` z is far above the planned grasp z (`error_m` about 7 cm) and the pitch has sagged (about 71).
- In the observer frame, a jaw is resting on the box top, which can push or turn the box.
- The program backs off and retries at a roll 90° away, reusing the old centre and yaw.
- That grip's `set_gripper(0)` returns `timeout` or `stalled` at about 2.7 A. After the lift, `observe()['gripper']` reads about 0, so the box slipped out and was not held.
- The program's last retry then forces a diagonal yaw (+45°) and grasps the box across its corners.
- Result: lifted, not placed, with tilt about 90° and resting False. The box falls over on release.

## When to apply
- The first descent is blocked high, meaning a jaw landed on the box instead of beside it.
- The fallback grasp is chosen from the pose estimate made before that contact.

The contact may have moved the box, and the blocked descent also shows the estimated yaw or width was wrong. A grasp planned from the stale centre and yaw lands off-centre and slips. Later "rotate the yaw by 45" fallbacks then make it worse.

## Repair
1. When the descent stops more than about 1.5 cm above the grasp z, retreat upward first (top + 6 cm).
2. Open the jaws and return to the overhead viewpoint.
3. `localize` the box again and rebuild the top-face rectangle at the known height from the new frame. This gives a new centre, yaw and side lengths.
4. Recompute the reach angle and the roll candidates from the new values.
5. Pick a candidate whose roll differs from the blocked one by more than 45°.
6. Return the updated estimate to the caller so that any later retry starts from it, not from the first view.

## Code sketch
```python
def remeasure(robot, bx, by, h):
    robot.set_gripper(100)
    move(robot, [bx, by, 0.16], [70, 60])
    b = robot.localize("the white box")
    if not b['found']:
        return None
    fr = robot.observe()['frames']['wrist']
    q = rects_from(robot, fr, b['bbox'], h, b['joints'])
    if q is None:
        c = robot.pixel_to_world(b['center_px'], h, b['joints'])
        return (c[0], c[1], None, None, None)
    _, yaw, da, db, cx, cy = q
    return (cx, cy, yaw, da, db)

# inside try_grasp, after the descent:
if r['status'] in ('stopped', 'timeout') and r['grasp_point'][2] > gz + 0.015:
    move(robot, [g[0], g[1], h + 0.06], [85, 78, 70, 60], roll)
    m = remeasure(robot, bx, by, h)
    if m:
        bx, by = m[0], m[1]
        if m[2] is not None:
            yaw, da, db = m[2:]
        phi = math.degrees(math.atan2(by, bx - 0.039))
        cands = grasp_cands(yaw, da, db, phi)
    choice = [c for c in cands if abs(wrap(c[1] - roll)) > 45][0]
# return bx, by, yaw, da, db to the caller too
```
