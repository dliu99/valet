# Tools

Helpers used while tuning valet on the real arm. Run them from the repo root; their output goes under the ignored `runs/`.

## Runs on the arm

- `batch.sh right left ...`: one pick, move and place goal per side for the white supplement bottle, with Claude's checks; prints one line per run.
- `batchbox.sh right left ...`: the same for the white box.
- `mixed.sh "white box:right" "glass pill jar:left" ...`: one goal per `object:side` pair, after any running goal ends.
- `lying.sh`: picks up a jar that may be lying on its side, then sets it down to the right and to the left.
- `wait.sh N [seconds]`: waits until the latest run has more than N programs or has ended.

## Arm programs

- `node tools/prog.ts tools/programs/park.js [seconds]`: runs one arm program with the same helpers skills use (`moveTo`, `move`, `grip`, `locate`, `segment`, `find`, `check`, `log`). It saves the last wrist picture to `runs/prog/prog.jpg`.
- `tools/programs/` holds the programs used along the way. `park.js` parks the arm. Some call skills that no longer exist (`pick`, `place_right`).

## Reading runs

- `node tools/checkpoints.ts <run>`: Claude's checks in a run, with frame hashes and any jaw move.
- `node tools/whys.ts <run> ...`: Claude's found, gripped and done verdicts for several runs.
- `node tools/progs.ts [run] [from] [--talk]`: a run's programs and logs, and optionally Claude's replies.
- `node tools/endat.ts <run>`: when a run ended.

## Camera and geometry

These are Python tools. Run them with `uv run --no-sync python tools/<name>.py`.

- `calib.py capture | fit`: moves the arm through set poses and saves frames to `tools/calib/`. It then fits the wrist camera from table features seen across views; `REAL_CAMERA` in `src/valet/armd/locate.py` came from it.
- `validate.py`: compares where each camera model puts the cap seen in earlier views with where the grasp found the bottle.
- `tipcheck.py`: compares where the camera model puts the grasp point with where the fixed jaw tip shows in the picture.
- `pts.py <height_cm> u,v ...`: table-plane points for pixels of the wrist picture, from the running arm server.
- `size.py l t r b`: a bottle's size and axis from one box.
- `grab.py image l t r b out.jpg`: GrabCut inside a loose box.
- `reach.py`: which pitches reach each reach and height, within the arm's real joint limits.
