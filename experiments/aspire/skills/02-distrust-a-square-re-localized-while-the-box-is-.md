---
id: "S02"
task: "benchmark"
run: "main"
round: 0
title: "Distrust a square re-localized while the box is in the jaws"
before: "r0a2-run3"
after: "r0a2-run4"
fixed: {"102": "placed", "104": "placed", "108": "placed", "109": "placed"}
verifier: "r0v2"
reproduced: ["102", "104", "108"]
verifier_placed: 8
before_seeds: [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
verifier_fitness_on_before_seeds: 8.0
before_fitness: 6.137
---

# Distrust a square re-localized while the box is in the jaws

## Failure signature
- Lift and carry `move_to_pose` both return `reached` or `stopped` near the planned pose, and the grip stalled on the box (amps about 2.7 A).
- The `localize("the blue square")` call made at the carry pose returns a bbox that is clearly wider than tall (height/width about 0.5).
- The program accepts that bbox and logs a "refined" square centre 2–3 cm away from the first estimate, which came from the unobstructed start view.
- The place descent then goes to the shifted target.
- Result: lifted, not placed, with off_target of about 2.5–3.5 cm. The box rests flat (tilt about 0) beside the square, overlapping its edge.
- In the localize keyframe, the held box and jaws cover part of the square. The drawn bbox spans only the visible strip of the square.

## When to apply
The program re-localizes the target square after grasping, to refine the drop point. The carried box sits between the camera and the table. It hides part of the 7 cm square, so the bbox centre is biased toward the uncovered side. A taped square seen from near-overhead should project roughly square. A strongly non-square bbox, or a large jump from the first estimate, means it was occluded.

## Repair
Accept the re-localized square centre only if all of these hold:
- The bbox does not touch the frame edge.
- It is reasonably large.
- Its aspect ratio is near 1 (0.7–1.4).
- Its back-projected centre is within about 2.5 cm of the estimate from the clear first view.

Otherwise keep the first estimate. Log both values so rejections are visible in the trace.

## Code sketch
```python
sq2 = robot.localize("the blue square")
if sq2['found']:
    l, t, r, b = sq2['bbox']
    w, hh = r - l, b - t
    ok = (l > 2 and t > 2 and r < 638 and b < 358 and w > 30
          and 0.7 < hh / w < 1.4)
    if ok:
        c = robot.pixel_to_world(sq2['center_px'], 0.0, sq2['joints'])
        if c is not None and math.hypot(c[0] - sx, c[1] - sy) < 0.025:
            sx, sy = c[0], c[1]          # refine only when consistent
    robot.log(f"square2 bbox {sq2['bbox']} ok {ok} -> {sx:.3f},{sy:.3f}")
tgt = np.array([sx, sy]) - ho * L2       # recompute the drop point
```
