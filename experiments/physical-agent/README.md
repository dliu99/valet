# physical-agent

Claude drives the simulated SO-101 only through one tool, `execute_absolute_eef_sequence`, the interface of
[Physical Agent](https://physical-agent-project.github.io/): up to 8 absolute end-effector targets per call, two
camera views per turn, a result that only says the call ran. Results and the recommendation are in
[REPORT.md](REPORT.md).

```bash
cd experiments/physical-agent && uv sync
uv run pytest                                                     # no API calls
uv run python -m physical_agent.demo --kind task --out runs/demos/task   # the in-context demo (expert, seed 101)
uv run physical-agent --condition zero --seeds 0-4                # also: demo, distractor; --model, --effort, --turns
uv run python -m physical_agent.analyze <run> [<run> ...]         # the report's table
```

`ANTHROPIC_API_KEY` is read from the nearest `.env` above this folder. Each run writes `runs/<name>/` (ignored by
git): per seed, `events.jsonl` (reasoning, thinking summary, call, result, state, and the true box pose for grading),
the frames the model saw, and `result.json`; `summary.json` per run. `results/` keeps the summaries the report cites.

## The interface

- **Target**: `x, y, z` (m, base frame: x forward, y left, z up) of the point between the jaw tips, `pitch_deg`
  (jaws below horizontal, 90 = straight down), `roll_deg` (the wrist roll joint; -90 holds the wrist camera level),
  `gripper` (0 closed - 100 open), `control_steps` (0.1 s ticks for the move, 1-50).
- **Yaw is dropped.** The arm has 5 joints: the shoulder pan points it at the target's x, y, so the jaws' heading
  about vertical is not free. A yawed box is matched with roll at a steep pitch, as the expert does.
- **Execution**: each target is solved with `valet_sim.ik` (seeded from the previous solution), joint goals are
  interpolated over `control_steps` ticks, then the arm gets up to 3 s to settle. An unreachable target stops the
  sequence; the result names it (`executed 2 targets; target 3 failed: no joints put the grasp point at ...`).
  Otherwise the result is only `executed N targets`.
- **Each turn** the model gets the scene (side) camera, the wrist camera and the joint/EEF state as text; it writes
  reasoning and makes one call. History, including thinking, stays in context. It ends the episode by replying
  without a call; the budget is 15 turns.
- **Demos** are `valet_sim.expert`'s plan run through the same tool on seed 101 (outside 0-19): three observations,
  each followed by the call executed from it. The distractor is the paper's control, a different task (lift the box
  and set it back where it was) from the same start.
- The wrist camera's near clip plane is moved from 3.2 cm to 4 mm (`scene.py`), so the box between the jaws stays
  visible as it would on the real camera.
