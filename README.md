# valet

An SO-101 arm driven by Jev ([TypeSafe](https://docs.typesafe.ai)). Claude gives Jev a goal; Jev looks
through the wrist camera and picks every move and how far it goes.

- **armd** (Python, `src/valet/armd/`): the only process on the servo bus. HTTP on `127.0.0.1:8750`.
- **the brain** (TypeScript, `brain/`): the tick loop, the recorder and the Claude planner.

## Setup

```bash
uv sync                                                   # once; afterwards always --no-sync
uv pip install -e "../lerobot[feetech,kinematics]"
cp valet.example.toml valet.toml                          # port, robot_id, allow_motion
npm install
```

SAM 3 masks (`POST /segment`, `segment()` in programs) need mlx-vlm; the `sam` extra in `pyproject.toml`
says what to install without disturbing LeRobot. The model (1.7 GB) loads on the first request.

Keys come from `.env`: `TYPESAFE_API_KEY` for Jev, `ANTHROPIC_API_KEY` for Claude.

## Run

```bash
.venv/bin/python -m valet.hardware.camfeed --name Innomaker --out runs/live/wrist.jpg   # a terminal with camera permission
uv run --no-sync valet-armd serve
npm run valet -- run "pick up the bottle and move it left" --ui
```

`valet ui` shows every tick: the frame, Jev's answers, the move and its result, and the exact wire
request and response; any tick can be re-asked with an edited goal. Runs are kept in
`runs/valet.sqlite`, frames under `runs/frames/<sha256>.jpg`.

## Checks

```bash
npm run check && npm test
uv run --no-sync pytest
```
