# The primitive API

A program is Python source that defines `run(robot)`. The engine calls it once per scene, lets the scene settle
for one second after it returns, and grades it. `robot` is the only way to sense or act. Programs may import
`numpy`, `math`, `itertools`, `functools`, `collections`, `statistics`, `dataclasses` and `typing`, and may not
use names that start with an underscore. `print` and `robot.log` write to the trace.

The primitives are what the real SO-101 has: joint readings and servo currents, a wrist camera, inverse
kinematics and a gripper. They contain no strategy: no grasp planner, no retries, no fallbacks.

## The task

A white box stands somewhere on the table in front of the arm. A blue square, 7 cm on a side, is taped to the
table elsewhere. Pick up the box and set it down on the square.

- **Placed**: the box's centre ends within 3 cm of the square's centre, tilted less than 15 degrees, resting
  on the table, with neither jaw touching it.
- **Lifted**: at some point the box was held in the jaws with its bottom at least 2 cm off the table.

Neither the box's pose and size nor the square's position is given; find them with the camera.

## Frames and units

World frame: origin at the arm's base on the table, **x** forward, **y** to the arm's left, **z** up. Metres and
degrees. The table is the plane z = 0. The shoulder's pan axis is at x = 0.039, y = 0.

**Pitch** is the angle of the jaws below horizontal: 90 points them straight down, 0 straight out. **Roll** is the
wrist roll joint. At roll -90 the jaws close across the reach direction (left-right as seen from the base); at roll
0 they close along it; changing roll by +d turns the closing direction by about +d (counter-clockwise seen from
above) when the jaws point down.

The **grasp point** is a point between the jaws, about 2 cm from the fixed jaw's inner face toward the moving jaw.
At roll -90 the fixed jaw is on the arm's left of the grasp point and the moving jaw closes toward it from the
right.

## Primitives

### `robot.observe() -> dict`

Reads the robot. Does not move it.

- `t`: simulated seconds since the start
- `joints`: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`, in degrees
- `gripper`: the jaw reading, 0 closed to about 100 wide open (degrees of the jaw joint)
- `amps`: each servo's current in amps
- `grasp_point`: `[x, y, z]` of the grasp point, from forward kinematics of the joint readings
- `pitch_deg`: the jaws' pitch below horizontal
- `frames`: `{"wrist": array}`, 360 x 640 x 3 uint8 RGB

### `robot.localize(text) -> dict`

Asks a vision model for the tight bounding box of `text` (for example `"the white box"`) in the wrist camera's
current frame. Only what is in view can be found; move the arm to look elsewhere.

- `found`: bool
- `bbox`: `[left, top, right, bottom]` in pixels, or None
- `center_px`: `[u, v]`, the box's centre
- `table_xy`: `[x, y]` where the ray through `center_px` meets the table plane z = 0
- `joints`: the joint readings the frame was taken at (for the geometry helpers)

At most 10 calls per scene.

### `robot.move_to_pose(xyz, pitch_deg, roll_deg=-90, timeout_s=4) -> dict`

Solves inverse kinematics for the grasp point at `xyz` with the given pitch and roll, starting from the current
joints, and drives the five arm joints there in joint space (each servo slews at up to 60 deg/s; the path between
poses is not a straight line). The gripper keeps its current goal. Returns once the arm settles or stops moving or
`timeout_s` passes.

- `status`: `reached` (within half a degree of every joint goal), `stopped` (stopped moving short of the goal:
  gravity sag or contact), `timeout`, or `unreachable` (no IK solution; the arm does not move)
- `grasp_point`, `pitch_deg`: where it ended up
- `error_m`: distance from `xyz` to where the grasp point ended up
- `goal_joints`: the IK solution

### `robot.set_gripper(pct, timeout_s=3) -> dict`

Drives the jaw to `pct` (0 closed, about 100 wide open). Returns when it gets there, stalls, or `timeout_s` passes.

- `status`: `reached`, `stalled` (stopped short of `pct`, for example on something between the jaws), or `timeout`
- `gripper`: the jaw reading
- `amps`: the gripper servo's current

## The wrist camera

The arm's only camera rides on the gripper: RGB, no depth, 640 x 360 frames, about 84 degrees vertical field of
view, with barrel distortion. Pixel `(u, v)` is (column, row), origin at the top left. At roll -90, image up is away
from the base and image right is the arm's right; with the gripper open the fixed jaw shows at the bottom left.
Things closer than 4 mm to the lens are not drawn.

These helpers use the calibrated camera model (distortion included) and the camera pose that forward kinematics
gives for a joint reading. They do not move the arm and are not traced.

- `robot.pixel_to_world(px, z, joints=None) -> [x, y, z] | None`: where the ray through pixel `px` meets the
  horizontal plane at height `z`.
- `robot.world_to_pixel(xyz, joints=None) -> [u, v] | None`: where a world point appears (None if it cannot).
- `robot.camera_ray(px, joints=None) -> {"origin", "direction"}`: the ray through `px`, in the world frame.

`joints` defaults to the current reading; pass the `joints` a `localize` result carries to reason about the frame
it was taken from.

## Limits

Per scene: 60 primitive calls, 10 `localize` calls, 120 s of simulated time, 300 s of wall time. Past a limit the
program is stopped (catching exceptions does not prevent it) and the scene is graded as it stands.

## Traces

Every primitive call is recorded: inputs, outputs, status, the state before and after, and the frames after it.
Besides the wrist frame, the trace keeps a frame from an observer camera beside the table (about 0.55 m to the arm's
right, looking across the workspace). It is there for debugging only: programs cannot see it, as the real arm has no
such camera. `move_to_pose` frames are overlaid with the goal (red cross) and where the grasp point ended up (green
cross); `localize` records the frame it looked at with the bounding box drawn.
