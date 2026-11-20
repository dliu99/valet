---
id: "S01"
task: "benchmark"
run: "main"
round: 0
title: "Triangulate the top-face centroid for box height, not the bbox centre"
before: "r0a2-run2"
after: "r0a2-run3"
fixed: {"100": "placed", "101": "placed", "102": "lifted", "103": "placed", "104": "lifted", "105": "placed", "106": "placed"}
verifier: "r0v0"
reproduced: ["100", "101", "102", "103", "104", "106"]
verifier_placed: 9
before_seeds: [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
verifier_fitness_on_before_seeds: 9.0
before_fitness: 1.0
---

# Triangulate the top-face centroid for box height, not the bbox centre

## Failure signature
- The descend `move_to_pose` returns `reached`.
- The following `set_gripper(0)` returns `reached` with a reading near 0 and amps well below the stall level (about 0.6–1.7 A instead of about 2.7 A). Nothing was between the jaws.
- The lift is `reached`, but the result is "not lifted" and the box has not moved.
- The logged height is too large: the estimate came from doubling the triangulated z of two `localize` `center_px` values, and it is often clamped at its upper bound.
- The grasp z is roughly the box's real top height or above it.
- The top-face rectangle that was back-projected at that height comes out too small (about 2 cm sides), so the roll and width choice is also wrong.
- In the observer frames, the open jaws close in the air above the box.

## When to apply
Box height comes from intersecting rays through `localize` bbox centres taken from two viewpoints. A bbox centre is not a fixed 3D point. It mixes the top face with whichever side faces are visible, and that mix changes with the viewpoint. So the closest-approach point is meaningless, and "2 × z" compounds the error.

## Repair
1. In each view, segment the bright, low-saturation top-face pixels inside the bbox.
2. Take their centroid. This is the image of the top face's centre, which is one physical point seen from both views.
3. Triangulate the two centroids with `camera_ray`. Use the z of the meeting point directly as the top height. Do not double it.
4. Back-project the top-face pixels at that height to get the footprint (yaw, width, centre).
5. Set the grasp z to about 2.5 cm below that top.
6. Keep the observe frame taken right where the first `localize` was taken, so the first view's mask is available.

## Code sketch
```python
def top_px(frame, bbox):
    l, t, r, b = [int(x) for x in bbox]
    c = frame[t:b, l:r].astype(float)
    g = c.mean(2); s = c.max(2) - c.min(2)
    vs, us = np.nonzero((g > np.percentile(g, 95) - 25) & (s < 40))
    return us + l, vs + t

def meet(robot, p1, j1, p2, j2):
    a = robot.camera_ray(p1, j1); b = robot.camera_ray(p2, j2)
    o1, d1 = np.array(a['origin']), np.array(a['direction'])
    o2, d2 = np.array(b['origin']), np.array(b['direction'])
    d1 /= np.linalg.norm(d1); d2 /= np.linalg.norm(d2)
    w = o1 - o2; bb = d1 @ d2; den = 1 - bb * bb
    s = (bb * (d2 @ w) - (d1 @ w)) / den
    t = ((d2 @ w) - bb * (d1 @ w)) / den
    return 0.5 * (o1 + s * d1 + o2 + t * d2)

ob = robot.observe(); box = robot.localize("the white box")
u1, v1 = top_px(ob['frames']['wrist'], box['bbox'])
t1 = [u1.mean(), v1.mean()]
# ... move to a second viewpoint, localize again -> box2, frame2 ...
u2, v2 = top_px(frame2, box2['bbox'])
p = meet(robot, t1, box['joints'], [u2.mean(), v2.mean()], box2['joints'])
h = min(0.15, max(0.02, p[2]))   # top height, NOT 2*z
# fit the footprint from (u2, v2) with pixel_to_world(..., h, box2['joints'])
gz = max(0.012, h - 0.025)
```
