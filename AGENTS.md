# Agent instructions

- Python 3.12+, managed with uv. `uv run pytest` must pass before a change is done; tests use
  `valet.fake` and stub clients and never call TypeSafe, Anthropic or hardware.
- Verify a runtime change by running it: `uv run valet run --fake --decider abort`, then read
  the newest `runs/*/events.jsonl` and `checks.jsonl`.
- Code owns control flow. Jev answers narrow questions about one frame; the runtime acts on
  yes and no and escalates unsure. Do not add steps where a model picks the next action.
- Question ids are `<kind>:<item>`. They are not sent to the model, so the question text must
  carry its whole meaning. `fake.FakeJudge` answers by kind; add a kind there when you add one.
- Never set `allow_motion = true` or run against the real arm unless the person says so.
- Imports of LeRobot, torch and anthropic stay inside the functions that need them, so the
  core runs without them.
