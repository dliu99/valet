# Agent instructions

- Jev picks every step and how far it goes (`valet do`, `brain/director.ts`) from small code skills
  in `brain/skills.ts`; Claude writes the skills. Each skill does its own careful motion and has a
  `ready(facts)` precondition, so Jev is only offered steps that can run; when one step is possible
  it runs without asking. Add capability as a new small skill, not a longer script. Boxes from Jev
  locate an object coarsely; the last centimetres are Jev's nudges. `armd` is the only process on
  the servo bus; it moves where it is told and only an explicit POST /relax turns torque off.
- Python 3.12, managed with uv. `uv sync` and plain `uv run` uninstall a hand-installed LeRobot,
  so use `uv run --no-sync`.
- Verify an armd change by running it: `uv run --no-sync valet-armd serve --fake --allow-motion`
  and drive it over HTTP, or `uv run --no-sync valet-armd check --fake --allow-motion`.
- The brain is TypeScript on Node 22.19+, run directly with no build step. `npm run check` must
  pass before a brain change is done. Verify a brain change on the real arm, with the person's
  go-ahead, and look at the run in `valet ui`.
- Do not write unit tests, in Python or TypeScript, and do not add test files, test doubles,
  fixtures or a test runner. Verify a change by running the real thing (armd over HTTP with
  `--fake`, the sim, or the arm with the person's go-ahead).
- The planner and the chat decider go through pi-ai. Jev and h2o are still called on System One's
  wire format directly (`brain/deciders/systemone.ts`). Every decision is recorded with its exact wire request and
  response. Keep it that way: the debug UI and replay depend on it.
- Question ids are never sent to the model, so the question text must carry its whole meaning.
- pi-ai and pi-durable are pinned to 1.5.2, because their APIs change without notice.
- Never set `allow_motion = true` or run against the real arm unless the person says so.
- Imports of LeRobot and placo stay inside the functions that need them, so armd runs without them.
