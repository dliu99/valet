# SAM 3 on the wrist camera: segment, fit, reach

**Verdict.** SAM 3 gives the arm fast, precise perception from the wrist camera alone. With the pixel-to-metre step
done in code, it places **18/20 (lifted 20/20)** on seeds 0-19 with **0 slow calls** (no Claude) and 3.0 SAM 3
calls per episode, each ~0.53 s on an M1 Max. For comparison, the expert that reads the true pose places 13/20, and
Claude driving the arm placed 11/20 with 21 Claude calls per episode. In the final run the box estimate the arm
grasps from was within 0.10 cm of the truth in every episode, and the square within 0.13 cm. Both misses happened
after a good estimate: the jaws closed on a yawed box's corners. Worth running on the real arm, once the camera pose
is calibrated (section 4).

**Against main** (section 5, rebased on main of 2026-11-21, in main's own `armd` sim, on main's task):
- Locating, on the same frames: the two-view mask fit puts the box centre 0.03 cm off (median; 0.09 cm p90). Main's
  `samAxis` is 1.1-1.3 cm off per view with a 7-14 cm tail, and its `find` is 0.9 cm off on the 12 of 30 poses
  where two views agree.
- On main's "move the box 10 cm left" goal, this policy places 7/10 with no LLM. Main's brain with Jev and no Claude
  places 0/10. There, main's grasp fails before locating matters: armd's 0.9 A gripper stop trips at first touch in
  its sim.

## 1. Running SAM 3 locally

- `facebook/sam3` is gated on Hugging Face and this account is not on its list (403). `mlx-community/sam3-bf16` is
  Meta's SAM 3 weights converted to MLX (same model, SAM License, open download) and runs through `mlx-vlm` 0.8
  (`mlx_vlm.models.sam3`). So this is SAM 3 itself, not a substitute. `sam3p/seg.py`.
- 1.7 GB of bf16 weights, 3.1-3.8 GB peak, 3 s to load. One backbone pass per frame serves every prompt.
- M1 Max, one 640x360 wrist frame, "white box" + "blue square" (median of 100 frames, `results/perception.jsonl`):

| SAM 3 input size | 336 | 504 | 672 | 1008 (default) |
|---|---|---|---|---|
| seconds per frame, both prompts | 0.26 | **0.54** | 0.74 | 1.99 |

The policy runs at 504: under the 1 s line that makes a call "slow", and its fitted positions match 1008's within
noise (table below). 336 is twice as fast and only slightly worse.

## 2. Perception against the true pose

100 frames from non-benchmark seeds 20-39, five views each: the start pose, 22 cm above the box, 15 cm above the box,
22 cm above the square, and the box set down on the square seen from above (`python -m sam3p.measure`). Truth
masks come from MuJoCo's segmentation render through the same lens resampling as the delivered frame
(`sam3p/render.py`). Only objects wholly inside the frame count. Centre errors are on the table, in cm.

- **fit**: an upright square-based box resting on the table (x, y, yaw, side, height) or the 7 cm square, fitted
  to the mask through the camera model (pose from the joints, pinhole + fitted barrel distortion), maximising the
  soft IoU of the projected silhouette (`sam3p/fit.py`). Away from the start view the box fit takes the start
  view's side and height (what the first policy did).
- **bbox centre**: the detector alone: the centre of its bounding box back-projected to a plane at 3.75 cm for the
  box and the table for the square. This is the only way to use Jev's answer, and the fair comparison.

| method | s / frame | box found | box mask IoU | box bbox IoU | box centre, fit (med / p90) | box centre, bbox centre (med / p90) | box yaw err deg | box height err cm (start view) | square mask IoU | square centre, fit (med / p90) | square centre, bbox centre (med / p90) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fit on true masks | - | 86/86 | 1.00 | 1.00 | 0.03 / 0.70 | 0.85 / 1.80 | 0.16 | 0.08 | 1.00 | 0.02 / 0.05 | 0.37 / 1.32 |
| SAM 3, 336 | 0.26 | 82/86 | 0.96 | 0.93 | 0.16 / 1.33 | 0.94 / 1.80 | 0.30 | 0.30 | 0.95 | 0.07 / 0.14 | 0.33 / 1.76 |
| SAM 3, 504 | 0.54 | 83/86 | 0.97 | 0.95 | 0.14 / 1.72 | 0.96 / 1.94 | 0.21 | 0.26 | 0.98 | 0.03 / 0.07 | 0.33 / 1.71 |
| SAM 3, 672 | 0.74 | 84/86 | 0.98 | 0.97 | 0.10 / 1.55 | 0.87 / 1.96 | 0.27 | 0.22 | 0.98 | 0.03 / 0.05 | 0.34 / 1.37 |
| SAM 3, 1008 | 1.99 | 85/86 | 0.99 | 0.98 | 0.09 / 2.08 | 0.91 / 2.02 | 0.16 | 0.16 | 0.98 | 0.02 / 0.05 | 0.36 / 1.56 |
| SAM 3.1, 504 | 0.50 | 84/86 | 0.97 | 0.95 | 0.17 / 2.23 | 0.97 / 2.19 | 0.27 | 0.31 | 0.98 | 0.03 / 0.07 | 0.33 / 1.71 |
| SAM 3.1, 1008 | 1.88 | 86/86 | 0.99 | 0.98 | 0.09 / 2.06 | 0.90 / 2.25 | 0.18 | 0.18 | 0.98 | 0.02 / 0.05 | 0.37 / 1.51 |
| Jev box question | 0.11 | 86/86 * | - | 0.83 | - | 0.93 / 3.14 | - | - | - | - | 0.59 / 1.17 |

\* Jev always returns a box. Per view and per seed: `results/perception.md`, `results/perception.jsonl`;
SAM 3.1 (`mlx-community/sam3.1-bf16`, same frames): `results/perception_sam31.jsonl`.

What the table says:
- SAM 3.1 is no better than SAM 3 on single frames: same mask IoU, a slightly wider tail at 504, 7% faster. Its
  headline change (Object Multiplex) speeds up tracking across video frames, which this per-frame pipeline does not
  use. The policy stays on SAM 3.
- SAM 3's masks are nearly as good as the truth (box IoU 0.97, square 0.98 at 504). Jev's boxes are looser
  (IoU 0.83) and its worst start-view box centre is 22 cm off (p90 over all views 3.1 cm).
- The gain is in the fit, not the detector: given only a bounding-box centre, SAM 3 and Jev land the box about
  0.9 cm off alike (a box's pixel centre is not over its footprint when seen at an angle). A mask lets code fit
  the whole box: 0.14 cm median, yaw within 0.2 degrees, height within 0.3 cm.
- The p90 tail (1.7 cm) has one cause: on 3 of 20 start views the box sits far left, its shaded side face reads dark
  grey, and the "white box" mask covers only the lit faces. A single-view fit then makes the box short and shifted,
  and that size was carried into later views. **Fitting one box to the start view and the look from above together
  removes the tail**: centre max 0.09 cm and height max 0.22 cm over 20 seeds (`results/two_view.md`).

## 3. The policy

`sam3p/policy.py`. Code owns every step; no model chooses anything. Per episode:

1. Survey from the start pose: SAM 3 on "white box" and "blue square", fit both. If either is missing, look from
   above three fixed spots.
2. Move the camera 22 cm above the box estimate, segment again, fit (one box to both views, once kept).
3. Grasp straight down with the box in the middle of the open jaws, 2.5 cm below its fitted top; the gripper reading
   says whether anything is held (point-and-reach's mechanics: with true pointing it placed 20/20).
4. Carry to the square's fitted centre, offset by where the box sits against the fixed jaw; lower, open, rise.
5. Look from above the square: segment, fit the box with its known size; if it is more than 2 cm off and upright,
   grasp and place again (once).

With masks from the truth instead of SAM 3 (debugging only) the same code places 20/20: the control and the fit are
not the bottleneck.

### Hillclimb

Seeds 0-4 saturate at once (5/5 with the first version), so changes were judged on non-benchmark seeds 20-39 too,
one at a time, kept only if placed went up. These dev runs are not on the scoreboard.

| step | change | seeds 0-4 placed | dev seeds 20-39 placed | kept | what it fixed |
|---|---|---|---|---|---|
| 0 | first version: start-view size, "white box" | 5/5 | 18/20 | - | - |
| 1 | fit the box to the survey and the look from above together | - | 19/20 | yes | seed 36: the shaded-face mask made the box 2.5 cm tall, so the grip went 2.6 cm wide and low |
| 2 | if "white box" finds nothing, ask "white cube" on the same frame (reuses the backbone pass) | - | 20/20 | yes | seed 37: from straight above the arm's shadow crosses the box top and "white box" scores under 0.1 |
| 3 | roll the wrist so the jaws close square to the box's fitted yaw | - | 20/20 | no | grips go flat (gripper 21-31 instead of up to 46), but a rolled wrist at the 68-78 degree pitch that reaches the square holds the box tilted, and it shifts 1-1.7 cm when released: off-target grows (true masks, dev 20-31: 11/12 placed vs 12/12) |
| ref | true masks instead of SAM 3 (debugging only) | 5/5 | - | - | 20/20 on seeds 0-19 |

Gentler releases (crack the jaws open and rise first; open slowly) and a vertical place (lowering the carry until
88 degrees reaches) did not remove step 3's shift, so it stays off (`Config.align_roll`).

The prompt wording was chosen on seeds 20-39 too (`python -m sam3p.prompts`, 40 frames): "white cube" gives
mask IoU < 0.8 on 2/40 frames, "white box" 4/40, "white block" 8/40, "cube" 11/40, "box" 14/40. The policy keeps
"white box" (the task's own words) and falls back to "white cube".

### Seeds 0-19

Config `sam3-504-twoview-orwhitecube` (`--config twoview --fallback "white cube"`), recorded with
`valet_sim.scoreboard.record()` in `results/episodes.jsonl`; per-episode logs with every look, fit and grade in
`results/logs/sam3-504-twoview-orwhitecube/`.

| | placed | lifted | slow calls / ep | fast calls / ep (SAM 3 frames) | s / SAM 3 frame | median wall s / ep | median simulated s / ep | $ / ep |
|---|---|---|---|---|---|---|---|---|
| this policy | **18/20** | **20/20** | **0** | 3.0 (max 4) | 0.53 | 3.3 | 11.1 | 0 |
| first version, seeds 0-4 | 5/5 | 5/5 | 0 | 3.2 | 0.59 | 4 | - | 0 |
| expert (reads the true pose) | 13/20 | 15/20 | - | - | - | - | - | - |
| Claude with tools (claude-policy) | 11/20 | 20/20 | 21.4 | - | - | 79 | - | 0.29 |

Off-target on the 18 placed boxes: 0.1-1.6 cm (median 1.0). In-episode error of each estimate against the truth
(cm, median / max over 20 episodes):

| estimate | box | square |
|---|---|---|
| survey (start pose, one view) | 0.19 / 3.81 | 0.04 / 0.13 |
| look from above (one view, start-view size) | 0.08 / 1.81 | 0.08 / 1.50 |
| **both views fitted together (what the grasp uses)** | **0.03 / 0.10** | - |
| after placing | 0.02 / 3.25 | 0.04 / 0.07 |

Slow calls per episode: **0** (no Claude, no model over 1 s). Fast calls: SAM 3 frames, ~0.53 s each. The policy
calls no Jev. The whole experiment spent $0 on Claude and $0.004 on Jev (100 box questions for the table).

**Where it fails.**
- Seeds 1 and 7 are corner grips. Both boxes are yawed 27 and -17 degrees off the jaws' closing direction. The jaws
  closed on the corners (gripper 38-39, where a flat grip of that box reads ~28). Both boxes came loose before reaching the
  square: seed 1's lies on its side 11.6 cm away, seed 7's stands 15.7 cm away. Perception was right
  in both (box 0.01 cm). The contact is a knife edge: seed 1 gripped flat (29) and placed in the first run, and the
  true-mask runs place both. A corner grip shows in the gripper reading: 11-16 units above what the fitted side
  predicts (reading ≈ 20 + 8.7 × (side cm - 4.0) for a flat grip, across every run here). The next step is to set
  the box down and re-grasp when that happens. Rolling the wrist fixes the grip but costs placement accuracy (step 3).
- The policy's own placement check calls a box "tipped" when the upright fit's IoU is under 0.85. It did so for both
  failures (correctly) and once on a dev seed for a box that was upright and placed. That ended the episode, which
  cost nothing there.
- Perception failures the hillclimb fixed, and that would recur without its two changes:
  - shaded faces dropping out of the "white box" mask;
  - the arm's shadow across the box top hiding it from straight above.
- Not tested here: clutter, other white objects, a box lying on its side at the start, a square partly under the
  box at the start.

## 4. Is SAM 3 worth running on the real arm?

Yes, as the perception layer, with three conditions.

- **Speed and memory fit.** At 504 px SAM 3 takes 0.5 s per frame on the M1 Max with 3-4 GB, and the task needs about 3
  frames, because positions are fixed once in table coordinates and the arm moves open-loop between looks. 336 px
  halves the time for slightly wider error (0.16 cm median against 0.14). SAM 3.1 gives the same accuracy.
- **The precision comes from geometry, so the geometry has to be right on the real arm.** The fit is exact only if
  the camera pose from the joints is. The sim README says the real joint zeros are about 5 degrees off. At 25 cm, 5
  degrees is about 2 cm on the table, far more than SAM 3's error. Main now has that calibration: `tools/calib.py`
  fitted the real wrist camera's mount, field of view and barrel (`REAL_CAMERA` in `src/valet/armd/locate.py`) so
  table features seen from seven poses land within 0.6 cm of each other. That 0.6 cm, not SAM 3, sets the floor on
  the real arm. Load it into `sam3p/camera.py`'s `Lens` and mount, then check the fit on a few real frames against a
  ruler before trusting it.
- **Real images help SAM 3, and lighting is the risk.** SAM 3 is trained on real photos, so the sim's flat-shaded
  faces are if anything harder than a real box. The two failure modes found here, shaded faces and the arm's shadow,
  are lighting effects, and both have fixes in place: the two-view fit and the fallback wording. A white box on a
  white desk, or clutter, are untested.
- **Jev's role.** For where things are, SAM 3 plus the two-view fit puts the box centre within 0.03 cm (median), against
  0.93 cm from Jev's box answer. Jev stays useful for yes/no checks SAM 3 cannot answer.

Also measured: Jev box questions at 0.11 s each give 0.9 cm median and a 3.1 cm p90 on the box. That is enough to
centre a look, not to grasp.

## 5. Against main's approach, in main's sim

Rebased on main of 2026-11-21 (dliu99/valet#13-#15). On main, `armd` has its own MuJoCo backend (`valet.armd.sim`)
and already runs SAM 3 (`armd/segment.py`). The brain (`brain/skills.ts`) puts the mask on the table its own way,
and Jev picks every skill. Main's sim shares valet_sim's arm model, wrist camera pose, field of view, barrel
distortion and optics: the camera pose matches exactly at random joints. So this experiment's code runs there
unchanged. Harness `sam3p/versus.py`, results in `results/versus/` (`summary.md`, per-episode JSONL, main's logs).

### Locating the box on the same frames

30 box poses: the 6×6×9 cm white box at a random place and yaw (seed 1). Each pose gets two wrist frames, from the
poses main's `find` uses: the start pose, then `lookAt` its first answer, or main's first scan pose where that is
out of reach. One SAM 3 pass per frame at 504 px feeds both methods.

- **main**: `samAxis` on armd's `describe()` of the best "white box" mask.
  - Looking steeply down, it takes the mask's centroid on a plane 5 cm up.
  - Otherwise it takes the mask's lowest pixel on the table, plus a 3.2 cm radius away from the arm.
  - `find` keeps two views that agree within 2 cm.
- **this**: the jaw-filtered mask with the "white cube" fallback, and an upright box fitted to it through the camera
  model.

| method | views | positions | centre error cm, median / p90 / max |
|---|---|---|---|
| main `samAxis`, start view | 30 | 23 | 1.34 / 10.59 / 13.79 |
| main `samAxis`, `lookAt` view | 30 | 25 | 1.12 / 6.75 / 7.87 |
| main `find`: both views, agreeing within 2 cm | 30 | 12 | 0.94 / 1.18 / 1.33 |
| mask fit, start view | 30 | 30 | 0.28 / 1.63 / 3.34 |
| mask fit, `lookAt` view (size from the start view) | 30 | 30 | 0.23 / 1.81 / 2.64 |
| **mask fit, both views together** | 30 | 30 | **0.03 / 0.09 / 0.32** |

- **Main's large misses (7-14 cm, a fifth of the start views) are all oblique "foot" readings of boxes on the arm's
  left.** `samAxis` assumes the mask's lowest pixel is the base's near edge. For a 9 cm box seen from beside
  through the wide lens, the lowest pixel is a top corner instead. On pose 3 it sits 70 px outside the projected base
  corners and lands 11-14 cm away on the table.
  - The mask itself was good. On these views the top "white box" instance was the box; the jaw's own instance
    scored 0.1-0.3.
  - The fit models the whole box, so it is not fooled.
- Main's agreement check rejects those misses, so `find` was within 1.3 cm whenever two views agreed (12 of 30).
  Otherwise it goes on looking, up to 4 views.
- In 11 of 60 views main had no position because no mask scored 0.5, its cut. The fit takes the best mask over 0.1
  that is not on the jaw, and asks "white cube" when "white box" finds nothing.

### The task

Main's own goal from `tools/batchbox.sh`: "Pick up the white box, move it about 10 cm to the left, and set it down
standing upright."
- 10 box poses (seed 0), the same for every run, all at yaw 0 because `/fake/reset` cannot turn the box.
- Graded from the sim's truth. **Placed**: upright, released, and within 2 cm of the spot 10 cm to the arm's left of
  where the box stood (main's `sideways`). **Lifted**: the box rose 1 cm.

How each side runs:
- **This policy** runs in-process on main's armd app, through its Guard and Mover (the code behind `POST /act`), on
  a clock that runs as fast as the physics computes.
  - It takes the target spot from its own fit.
  - It closes with its own rule, until the jaw stops moving. armd's close stopped at first touch here too, and lifted
    nothing in a smoke test.
- **Main's brain** runs as it ships: `valet do --no-claude` against `valet-armd serve --sim` on port 8771.
  - Jev picks every skill.
  - Claude's line-up and checkpoints are off, to keep Claude spend at $0.

| | placed | lifted | LLM calls / ep | Jev calls / ep | SAM 3 frames / ep | arm time / ep (median) |
|---|---|---|---|---|---|---|
| **this policy** | **7/10** | **9/10** | 0 | 0 | 3 | 22 s simulated (3 s wall) |
| main as it ships | 0/10 | 1/10 | 0 | 14 | - | 36 s (sim runs in real time) |
| main, armd's gripper stop raised from 0.9 A to 1.5 A | 0/10 | 5/10 | 0 | 16 | - | 49 s |

Where each fails:
- **Main as it ships: the close.** 21 closes over 10 episodes "stopped part way".
  - armd's grip stops at 0.9 A of gripper current. In the sim the jaw draws 0.78 A just moving.
  - At main's 60° tilt, the moving jaw touches the box at 74 % open and reads 0.94 A, before it has pushed the box
    onto the fixed jaw. This was replayed in-process from main's own recorded joints.
  - `holds()` then passed two of these empty grips (the box shifted 0.057 against a 0.06 limit, and scaled 0.86
    against 0.85). The arm carried nothing and Jev chose done.
  - Jev also chose `line_up` 18 times, which does nothing without Claude.
- **Main with a 1.5 A stop: the grip and the carry.** Closes went further (to 59 % open in the replay), and 5 boxes
  left the table, but none arrived at the spot.
  - Four ended 6.9-9.9 cm from the spot.
  - One landed on its side 1.9 cm from the spot.
  - With no current stop at all, the replayed close clamps the box. But the soft sim servos (kp 80) let the arm ride
    up 4 cm and tilt 11° as the jaws close on the box at 60°.
- **This policy: three misses, each with a box estimate within 0.02 cm of the truth.** All three boxes were 19-24°
  to the side of the arm's line, where yaw 0 on the table turns the box that much against the jaws. That is the
  setup of section 3's corner grips; this harness did not log the gripper readings that would confirm it.
  - One was not lifted.
  - One fell out in the carry.
  - One was set down 2.7 cm off.

What this says for main:
- **Locating.** The mask fit is a drop-in gain for `find`: 0.03 cm from two views, against main's 0.9-1.3 cm when its
  views agree, and no 7-14 cm single-view misses. Section 2 shows the same against Jev's box answers.
- **Grasping, at least in its sim, fails before locating matters.**
  - The 0.9 A stop is probably right on the real arm, where moving joints read 0.1-0.25 A (the comment on
    `armd/sim.py`'s `ServoModel`). It is the sim's gripper that draws too much current.
  - So the sim's gripper current needs to come down before the sim can judge main's grasps.
- **Main's design leans on Claude** to line up the jaws and confirm the grip. This run turned that off, so main with
  Claude is untested here.

## Reproduce

```bash
cd experiments/sam3 && uv sync --extra sam
uv run python -m sam3p.measure --seeds 20-39 --jev       # the perception table (needs TYPESAFE_API_KEY for Jev)
uv run python -m sam3p.measure --seeds 20-39 --sam 3.1 --res 504 1008 --no-truth --out results/perception_sam31.jsonl
uv run python -m sam3p.tables                            # the compact tables above
uv run python -m sam3p.bench --seeds 0-19 --config twoview --fallback "white cube" --record
uv run python -m sam3p.bench --seeds 0-19 --oracle       # same policy, true masks
# section 5, against main (armd's code is imported from ../../src)
PYTHONPATH=.:../../src uv run python -m sam3p.versus frames --poses 30 --seed 1
PYTHONPATH=.:../../src uv run python -m sam3p.versus task --episodes 10 --seed 0
# main's brain: serve armd's sim on a spare port (never 8750), then drive it from the repo root's `npm run valet`
(cd ../.. && PYTHONPATH=src experiments/sam3/.venv/bin/python -m valet.armd serve --sim --allow-motion --config <toml with [armd] url = "http://127.0.0.1:8771">)
PYTHONPATH=.:../../src uv run python -m sam3p.versus main --episodes 10 --seed 0 --armd http://127.0.0.1:8771
PYTHONPATH=.:../../src uv run python -m sam3p.versus summary
```
On the shared Mac, wrap anything that loads SAM 3 in `python3 ~/.valet/slot.py --name <team> --heavy --`.
