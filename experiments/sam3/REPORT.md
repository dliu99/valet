# SAM 3 on the wrist camera: segment, fit, reach

**Verdict.** SAM 3 gives the arm fast, precise perception from the wrist camera alone, and with the
pixel-to-metre step done in code it lifts placement to FINAL_PLACED/20 (lifted FINAL_LIFTED/20) on seeds 0-19 with
**no Claude calls** and about FINAL_FAST SAM 3 calls of ~0.55 s per episode. The expert that reads the true pose
places 13/20; Claude driving the arm placed 11/20 with 21 Claude calls per episode.

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

HILLCLIMB_TABLE

### Seeds 0-19

FINAL_TABLE

Slow calls per episode: **0** (no Claude, no model over 1 s). Fast calls: SAM 3 frames (~0.55 s each). Jev: 0.

FAILURES

## 4. Is SAM 3 worth running on the real arm?

REAL_ARM

## Reproduce

```bash
cd experiments/sam3 && uv sync --extra sam
uv run pytest                                            # camera model and fits, no SAM 3
uv run python -m sam3p.measure --seeds 20-39 --jev       # the perception table (needs TYPESAFE_API_KEY for Jev)
uv run python -m sam3p.bench --seeds 0-19 --config twoview --record
uv run python -m sam3p.bench --seeds 0-19 --oracle       # same policy, true masks
```
On the shared Mac, wrap anything that loads SAM 3 in `python3 ~/.valet/slot.py --name <team> --heavy --`.
