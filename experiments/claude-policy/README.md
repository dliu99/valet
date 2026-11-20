# claude-policy

Claude runs the arm. Claude Opus 5.5 gets one goal sentence and three tools, and plans and judges everything else on
the valet-sim pick-and-place benchmark. Results are in [REPORT.md](REPORT.md).

```bash
cd experiments/claude-policy && uv sync
uv run pytest                                                    # stub Claude and Jev, no API calls
uv run claude-policy --seeds 0 1 2 3 4 --overlay --tag try       # needs ANTHROPIC_API_KEY (read from the nearest .env)
uv run python -m claude_policy.show runs/<run>/seed03/events.jsonl
uv run python -m claude_policy.table label=runs/<run> ...
```

| File | What it does |
|---|---|
| `agent.py` | the goal, system prompt, tool definitions and the append-only Claude loop; cost accounting |
| `arm.py` | the tools on the sim in simulated time: frames, joint state, IK moves (errors say which pitches reach), gripper |
| `overlay.py` | a table grid in base-frame cm and the grasp point, drawn on frames from joint state and camera calibration |
| `jev.py` | one yes/no question to Jev about the current frames |
| `run.py` | runs seeds in parallel processes and grades each with the sim's own `score(truth())` |
| `jev_eval.py` | scores Jev's end-of-episode judgement against the grader |

A policy reads only `Sim.state()` and frames; `truth()` and `score()` are used only by `run.py` after the episode.
The wrist camera is the only camera by default (`--cams wrist side` adds the side view for comparison). Each run writes
`runs/<time>-<tag>/` (git-ignored) with every request's usage, every tool call, every frame Claude saw, and the result.
