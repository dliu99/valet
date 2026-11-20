---
id: "S03"
task: "benchmark"
run: "main"
round: 1
title: "Descend to grasp at vertical pitch, not at the hover's fallback pitch"
before: "r1a0-run2"
after: "r1a0-run3"
fixed: {"100": "placed", "108": "placed"}
verifier: "r1v0"
reproduced: ["100", "108"]
verifier_placed: 10
before_seeds: [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
verifier_fitness_on_before_seeds: 10.0
before_fitness: 8.757
---

# Descend to grasp at vertical pitch, not at the hover's fallback pitch

## Failure signature
- The pre-grasp hover `move_to_pose` gets `unreachable` at pitch 90, 85 and 78, then succeeds near pitch 70.
- The descend `move_to_pose` is then tried first at that inherited pitch (about 72). It returns `reached` or `stopped` with `pitch_deg` around 72, and pitch 90 is never tried even though it is reachable lower down.
- `set_gripper(0)` stalls on the box (about 2.7 A, reading about 29). Lift and carry go fine.
- The place descent returns `pitch_deg` near 88–90.
- Result: lifted, not placed. off_target is about 3 cm, tilt about 0, resting, released. The box sits flat but beside the square.

## When to apply
The grasp descent reuses the pitch the hover ended at, while the place descent is near vertical. A gripped box hangs rigidly below the grasp point. Changing pitch by Δ between grasp and place swings the box about the grasp point. Its footprint moves by about (grasp height above the box bottom) × sin Δ, which is 1.5–2 cm for a tall box at Δ ≈ 18°. That is enough to miss the 3 cm tolerance even when the square estimate is right.

A high hover often cannot reach vertical pitch, but the grasp height usually can. So the hover's fallback pitch must not carry over to the descent.

## Repair
1. Give the grasp descent its own pitch list, starting at 90: `[90, 84, 78, 72]`. Do not seed it with the hover's `pitch_deg`.
2. Place with the same pitch preference (90 first), so grasp and place pitch match.
3. Log both pitches. If they still differ by more than about 5°, either place at the grasp pitch or shift the place target by the swing.

## Code sketch
```python
r = move(robot, [g[0], g[1], h + 0.04], [90, 85, 78, 70, 60], roll)  # hover: any pitch
r = move(robot, [g[0], g[1], gz], [90, 84, 78, 72], roll)            # descend: vertical first
grasp_pitch = r['pitch_deg']
# ... grip, lift, carry ...
r = move(robot, [tgt[0], tgt[1], held_z + 0.012], [90, 84, 78, 72], roll)
robot.log(f"grasp pitch {grasp_pitch:.1f} place pitch {r['pitch_deg']:.1f}")
```
