# jev-claude-7: Jev measures, Claude plans and learns

A valet-sim pick-and-place policy that uses the wrist camera only. See [REPORT.md](REPORT.md) for the design,
results and comparison.

```bash
cd experiments/jev-claude-7 && uv sync
uv run pytest                                              # offline: no TypeSafe or Anthropic calls
uv run python -m jc7.run --seeds 0 1 2 3 4 --name try      # parallel, no lessons
uv run python -m jc7.run --seeds $(seq 0 19) --name full --lessons runs/full/lessons.md   # in order, with lessons
uv run python -m jc7.run --seeds 0 1 2 3 4 --name local --fast h2o --lessons runs/local/lessons.md   # H2O-Lightning-4B at 127.0.0.1:8741 instead of Jev
uv run python show.py runs/try/seed-03                     # one episode's plans, steps and grade
uv run python analyze.py runs/full                         # measurement error and outcomes, graded after the fact
```

Wrap runs in `python3 ~/.valet/slot.py --name team-7 -- ...` on the shared Mac. Every episode is appended to
`results/episodes.jsonl` for `python -m valet_sim.scoreboard`; `--model` picks the Claude planner and `--max-usd`
stops a run once Claude spend reaches it.

Keys come from `TYPESAFE_API_KEY` and `ANTHROPIC_API_KEY`, read from the repository's `.env` when unset.
Each episode writes `runs/<name>/seed-NN/` (`events.jsonl`, `result.json`, the frames Claude and Jev saw) and the
run writes `runs/<name>/summary.json`. `runs/` is not committed; `results/` holds the scoreboard rows, the full runs'
summaries and the lessons each run wrote.

| File | What it does |
|---|---|
| `jc7/camera.py` | the wrist camera from forward kinematics: pixel rays, projection, the lens's barrel distortion |
| `jc7/see.py` | Jev's 2D boxes to table positions: centring looks, a two-height look, an oblique look, a prism fit |
| `jc7/body.py` | moves of the grasp point by IK, gripper drive with stall detection, camera poses for looking |
| `jc7/planner.py` | the Claude prompt and the plan schema; plan, replan, lesson |
| `jc7/episode.py` | one episode: measure, plan, run with checks, replan, self-check, lesson, grade |
| `jc7/run.py` | runs seeds and summarises them |
