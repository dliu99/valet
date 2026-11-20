# valet

A robot harness for an SO-101 arm with one wrist camera. It assembles a sandwich.

Code owns the task. Between skills it asks Jev ([TypeSafe](https://docs.typesafe.ai)) narrow
yes/no questions about the wrist camera's picture ("is the gripper holding the cheese?",
"is the cheese on top of the sandwich?") and acts on the probabilities. Only an unsure
answer, or repeated failure, goes to a decider: a person at the terminal, or Claude. Nothing
in the loop generates code or picks its own next action.

```
recipe.py      the layers, in order, and the questions about each      (a coding model edits this)
   │
runtime.py     find → pick → check held → place → lift → check on top
   │    ├──► Jev (judge.py)          ~1 request per check, many questions at once
   │    └──► decider (escalate.py)   only when Jev is unsure: person or Claude Opus 5.5
skills.py      scripted moves  ·  policy.py: a trained LeRobot policy (ACT, SmolVLA)
   │
hardware/so101.py   SO-101 through LeRobot: IK, interpolated motion, wrist camera
```

## Try it without hardware

```bash
uv sync --extra claude
```

```bash
uv run valet run --fake --decider person
```

```bash
uv run pytest
```

`--fake` uses a simulated table, arm and Jev (`fake.py`); it makes no API calls. Records go to
`runs/`.

## On the arm

1. Install LeRobot from your checkout into this environment:

   ```bash
   uv pip install -e "../lerobot[feetech,placo-dep]"
   ```

2. `cp valet.example.toml valet.toml` and work through it. The header of
   `src/valet/hardware/so101.py` lists the calibration steps: LeRobot calibration, the URDF, the
   gripper-down pose (`valet pose`), camera intrinsics and where the camera sits on the gripper.
   Check that last one with `valet locate cheese --config valet.toml` against a ruler.
3. Set `TYPESAFE_API_KEY`, and for `--decider claude` an Anthropic credential
   (`ANTHROPIC_API_KEY` or `ant auth login`).
4. Set `allow_motion = true`, then:

   ```bash
   uv run valet run --config valet.toml --decider person
   ```

Every target is checked against the `[workspace]` box before anything moves, and LeRobot's
`max_relative_target` clips each joint command. Keep a hand near the power switch for the
first runs.

## Skills

The recipe names a skill per action (`pick_flat`, `place_on_stack`); `[skills]` in the config
binds each name to a scripted skill or a LeRobot policy. Scripted picks go straight down and close,
which suits stiff items. Thin, floppy slices usually need a learned skill: record teleoperated demos
with the leader arm for one narrow action (pick a slice from the table), train ACT or SmolVLA, and
bind it. The policy's camera name and resolution must match its training dataset.

## Tuning from records

Each run writes `runs/<recipe>-<time>/`:

- `checks.jsonl`: every question, Jev's answer, how it was read (yes, no, unsure) and the frame.
- `events.jsonl`: moves, skills, decisions and who made them.
- `frames/`: every picture a question was asked about.

Add `"actual": true|false` to check lines by hand, then pick per-question thresholds from how
often each probability band was right. TypeSafe suggests 0.7 / 0.3 as a starting point.

## Limits

- Untested on hardware. The SO-101 adapter and policy skill follow LeRobot's examples and
  interfaces at the checkout this was written against.
- Jev's image and Box questions are labelled "Dreamer-ready" by TypeSafe: prototype quality,
  subject to change. Box needs Jev 1.14.
- Depth comes from the table plane: an item's height is taken from the recipe, not measured.
