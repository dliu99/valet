# valet-sim

A MuJoCo SO-101 on a table with a box to pick up and set down on a blue square. It is the shared testbed for
valet's control experiments: every approach runs here first, on the same seeded scenes, before it touches the arm.

```bash
cd sim && uv sync
uv run pytest
uv run python -m valet_sim.expert          # the scripted reference on the 20 benchmark seeds
uv run valet-sim --port 8770 --seed 0      # the HTTP API, in real time
```

## Driving it

Python, simulated time (as fast as the CPU allows):

```python
from valet_sim.sim import Sim, SimSpec, with_box
from valet_sim.task import episode, score
from valet_sim.ik import IK

ep = episode(3)
sim = Sim(with_box(SimSpec(), ep.box, ep.target_xy))
sim.set_goals(IK(sim.model).solve([0.25, 0.0, 0.08], pitch_deg=60))   # or raw joint goals in degrees
sim.step(2.0)
image = sim.frame("wrist")          # PIL image, upright, as the real camera pipeline delivers it
print(sim.state(), score(sim.truth()))
```

HTTP, real time, for harnesses in any language: see the routes at the top of `src/valet_sim/server.py`
(`/state`, `/goals`, `/move`, `/torque`, `/frames`, `/reset`, `/score`). Give each experiment its own `--port`.

## The benchmark

- Episodes are `task.episode(seed)`; the benchmark is seeds 0-19. Box 4-5.5 cm square, 6-9 cm tall, yawed up to
  30 degrees, 20-27 cm from the pan axis within 40 degrees of straight ahead; the target square is at least 12 cm away.
- Placed: box centre within 3 cm of the square, tilted under 15 degrees, resting on the table, jaws off it.
  Lifted: held with its bottom at least 2 cm off the table at some point.
- Report placed and lifted out of 20, median seconds per episode, and model calls and cost per episode.
- A policy sees only `state()` (or `/state`) and frames. `truth()`, `score()` and `/score` are for grading.
- Record every benchmark episode with `valet_sim.scoreboard.record()` to `experiments/<name>/results/episodes.jsonl`:
  placed, lifted, seconds, `slow_calls` (Claude or any model slower than about a second) and `fast_calls` (Jev,
  a local model, a segmenter). `python -m valet_sim.scoreboard <dirs>` prints every config side by side. A config
  with more than 5 slow calls per episode is marked not scalable: iterate with any model, but the goal is a policy
  that does not lean on a slow LLM.
- Reference: `python -m valet_sim.expert` reads the true box pose and places 13/20 (lifted 15/20). It proves the
  scenes are solvable and can generate demonstrations; it is a floor to beat, not a ceiling.

## Opt-in: a table tag and the real arm's errors

Both are off by default, so results stay comparable across teams; switch them on per experiment.

```python
from valet_sim.scene import SceneSpec, TagSpec, tag_corners
from valet_sim.sim import Errors, Sim, SimSpec

spec = SimSpec(scene=SceneSpec(tag=TagSpec()),                         # a 5 cm AprilTag 36h11 beside the base
               errors=Errors(joint_offsets_deg=(4, -5, 3, -4, 5, 0),   # readings and goals = true angle + offset
                             compliance_deg_per_nm=4.0,                # ~1.5 cm of sag at the far edge
                             mount_shift_m=(0.004, 0, 0), mount_tilt_deg=(0, 2, 0)))  # camera off its `Look` pose
```

- `TagSpec` places the tag clear of every box and target the benchmark draws; `tag_corners` gives its corners in the
  world in OpenCV's detection order. Building a scene with a tag needs OpenCV: `uv sync --extra tag`.
- With `Errors`, `state()`, `joints()` and `grasp_point()` report what the arm believes (forward kinematics of its
  readings); `truth()` adds the true joints, grasp point and wrist camera pose, for measuring error.
- With a mount error, `sim.model` holds the true camera pose, because it renders the frames. Build a policy's camera
  model from `build(SceneSpec())`, or the policy is handed the very calibration the error takes away.
- `experiments/apriltag/REPORT.md` measures what the tag is worth under these errors.

## How close it is to the real arm

| | Matched to the real SO-101 | Not matched |
|---|---|---|
| Kinematics | Menagerie's model; joints are LeRobot calibrated degrees, same convention as `so101_new_calib.urdf` | the real joint zeros are about 5 degrees off and the arm sags ~1.5 cm (opt-in: `Errors`) |
| Servos | kp 80 instead of Menagerie's 998: 0.4 A at 30 deg/s, against 0.1-0.25 A measured at 8-25 deg/s; goals slew at 60 deg/s | backlash, overload latching, encoder glitches |
| Gripper | jaws stall on the box and hold it by friction; 0-100 taken as degrees of the jaw joint | the real 0-100 may be a normalised range |
| Wrist camera | fitted to the real fixed jaw's outline (IoU 0.16 -> 0.64), level at roll -90, ~84 deg vertical FOV, barrel distortion, noise, blur, vignette | lens model is approximate; fitted from two real frames |
| Scene | a table, a wall, soft light, shadows | the real desk's clutter, lighting and textures |

Frames are upright: at wrist roll -90, image up is away from the base and image right is the arm's right; with the
gripper open the fixed jaw enters from the bottom left. Jev reads these frames sensibly: in a 10-frame check its
box tracked the true box offset in order.

`src/valet_sim/model/` is MuJoCo Menagerie's `robotstudio_so101` (Apache-2.0, see `model/LICENSE`).
