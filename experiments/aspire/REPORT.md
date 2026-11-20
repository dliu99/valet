# ASPIRE-style code-as-policy on valet-sim

Can coding agents that debug from per-primitive traces, and keep a skill library they verified themselves, write a
pick-and-place program that generalizes? On this benchmark, yes. Claude agents wrote and debugged a program on debug
seeds 100-109 only. It places the box on **19 of 20 held-out scenes and lifts it on all 20**. The scripted expert,
which reads the true box pose, places 13. Nobody hand-wrote the grasp, the perception or the four skills: they came
out of the agents' own debugging.

The program needs Claude only to write it. At run time its only model use is about four box questions per episode
(`localize`). With **Jev** answering them it places **15/20** (17/20 after a one-line fix to how my `localize`
read Jev's confidence; see below). With **H2O-Lightning-4B running on this Mac** it places **20/20**, with no model larger than 4B and no API at run time.

Everything ran in simulation (`sim/`, valet-sim); nothing touched the arm.

## Held-out result (seeds 0-19, evaluated once per configuration)

| Configuration | Placed | Lifted | Model calls per episode at run time | Run-time $ per episode |
|---|---|---|---|---|
| Final program, `localize` by Claude Opus 5.5 | **19/20** | **20/20** | 4 slow (Claude) | $0.014 |
| Final program, `localize` by Jev, as first run | 15/20 | 16/20 | 4 fast (Jev) | ~$0.009 (est.) |
| Same, after the `found` fix (post hoc) | 17/20 | 19/20 | 4 fast (Jev) | ~$0.009 (est.) |
| Final program, `localize` by H2O-Lightning-4B, local | **20/20** | **20/20** | 4 fast (local) | $0 |
| Scripted expert (reads the true pose) | 13/20 | 15/20 | 0 | 0 |

Every row is on the shared scoreboard: `results/episodes.jsonl`, written with `valet_sim.scoreboard.record()`.
Claude calls count as `slow_calls`; Jev and H2O calls count as `fast_calls`. The median episode is 10.6-10.9 s of
simulated time. The program's only miss with Claude perception, seed 7, comes from a top-face fit that
underestimated one side (3.1 cm for about 4.6 cm), which led to an along-reach grasp; the box slid out during the
carry and landed on its side.

**Jev and the `found` fix.** The first Jev run (pre-registered) placed 15/20. Three of its five misses (seeds 0, 1,
10) were not perception errors. Jev returned a sensible box at confidence 0.27-0.28, but my Jev backend reported
`found: False` below an arbitrary 0.3 cutoff, so the program stopped at once. The Claude backend has no such cutoff.
The fix makes `localize` report `found` whenever the model returns a box, and passes Jev's confidence through to the
program. I found it by looking at held-out failures, so the 17/20 rerun is post hoc; the 15/20 is the clean number.
The other two misses are genuine. On seed 17, a looser box put the jaws beside the box. On seed 18, the box was
placed 3.5 cm off. On five reference frames, Jev's boxes have an IoU of 0.67-0.95 against truth and Claude's
0.92-0.98; on the one frame measured, H2O's are 0.87-0.88 (`python -m aspire.localize_check`).

**Do debug scores predict held-out ones?** All five programs that placed at least 9/10 debug scenes were also run
on the held-out seeds afterwards, as a diagnostic not used for selection. They placed 17, 18, 19, 19 and 20 of 20.
Programs that pass the debug scenes generalize; the final pick was not a lucky one.

## How it works

| Piece | Where | What |
|---|---|---|
| Primitive API | `api.md`, `src/aspire/robot.py` | `observe()`, `localize(text)`, `move_to_pose(xyz, pitch, roll)`, `set_gripper(pct)`, plus camera-geometry helpers (`pixel_to_world`, `world_to_pixel`, `camera_ray`). Wrist camera only, as on the real arm. No grasp planner, no retries, no strategy. |
| Perception | `src/aspire/perceive.py`, `camera.py` | `localize` asks one model for a bounding box (Claude, Jev or H2O) and back-projects its centre to the table with the calibrated wrist model. The camera model matches MuJoCo to sub-pixel, distortion included. |
| Execution engine | `src/aspire/engine.py` | Runs a program on a seed in a sandbox: no imports beyond numpy and math, no `_` names, no `truth`. It records every primitive call (inputs, outputs, status, state before and after, wrist and observer frames with goal and grasp-point overlays) and grades with `task.score` afterwards. |
| Loop | `src/aspire/evolve.py`, `agents.py` | K=3 actors per round on the Claude API (Opus 5.5, effort medium, at most 10 tool calls each): `run_program` on debug seeds, `view_keyframes`, `submit`. Round 0 starts from scratch. Each later actor gets one of the top two programs, its failure traces and the other top program, starting from a different failing seed. Selection is fixed in advance: fitness = placed + 0.3 × lifted-only, then the earliest round. |
| Skills | `skills/*.md` | A fix is a run that places or lifts scenes its predecessor failed on the same seeds. It can come between rounds (child against parent) or inside one actor's session (run k+1 against run k). A writer turns the best few fixes per round into an entry: failure signature, when to apply, repair, code sketch. A **fresh verifier** gets only the failing program, its traces and the entry, never the fixed code. The entry is admitted only if the verifier's program reproduces at least half the fixes without lowering the score. |

The observer side camera appears in traces for the debugging agents only. Programs cannot read it.

## Debugging rounds

The loop ran 2 rounds on seeds 100-109 and stopped once a program placed 10/10. Seeds 0-19 were never run before
the final evaluation.

| Round | Actors (placed of 10 on debug) | Best |
|---|---|---|
| 0, from scratch | 9, 5, 9 | r0a2: 9 placed, 10 lifted |
| 1, from r0a2 and r0a0 | 10, 10, 9 | **r1a0: 10/10 (the final program, `programs/benchmark_final.py`)** |

The verifiers' programs also joined the archive (9, 8, 10 and 10 of 10). The final program is 238 lines. It:
- localizes the box and square from the start pose;
- triangulates the height of the box's top face from two wrist views;
- segments the top face and fits a minimum-area rectangle to get the yaw and side lengths;
- closes the jaws across the narrow side, offset for the fixed jaw;
- descends straight down;
- checks the grip by the jaw reading, and re-measures and retries after a miss;
- rejects a re-localized square if the box in the jaws occludes it;
- places at the grasp pitch and nudges a stuck release.

## The skill library (all four discovered and verified by the agents)

| Skill | What it fixed (debug seeds) | Verifier, from the entry alone |
|---|---|---|
| [S01 Triangulate the top-face centroid for box height, not the bbox centre](skills/01-triangulate-the-top-face-centroid-for-box-height.md) | Grasps closing in the air above the box, because two bbox centres are not one 3D point. 7 scenes fixed. | 6 of 7 reproduced; 1/10 → 9/10 |
| [S02 Distrust a square re-localized while the box is in the jaws](skills/02-distrust-a-square-re-localized-while-the-box-is-.md) | Boxes set down 2.5-3.5 cm off, because the held box hid part of the square and skewed its bbox. 4 fixed. | 3 of 4; 5/10 → 8/10 |
| [S03 Descend to grasp at vertical pitch, not at the hover's fallback pitch](skills/03-descend-to-grasp-at-vertical-pitch-not-at-the-ho.md) | About 3 cm placement error from the box swinging when grasp and place pitch differ. 2 fixed. | 2 of 2; 8/10 → 10/10 |
| [S04 Re-measure the box after a blocked grasp descent](skills/04-re-measure-the-box-after-a-blocked-grasp-descent.md) | Retrying from a stale pose after a jaw landed on the box and moved it. 1 fixed. | 1 of 1; 9/10 → 10/10 |

Two more fixes went to the writer, which answered NONE for both (already covered by the library, or not a general
repair). No candidate
failed verification. Each entry carries its provenance: run, round, the before and after programs, the seeds fixed,
the verifier and the scores.

## Transfer: does the library help on a new task?

The variant task (`tasks.variant_episode`) has thinner, taller boxes (3.2-4.2 cm square, 8.5-10 cm tall). Each box
starts on the arm's right and the square is on its left, so every carry crosses in front of the base. Debug seeds
are 200-209, held-out 300-319. The expert places 4/10 on the variant debug seeds and 10/20 on held-out.

- **The benchmark program, unchanged:** 7/10 on variant debug and **18/20 on variant held-out**. The program itself
  transfers.
- **From scratch, with and without the library.** Six actors per arm, the same prompt and budget (10 tool calls),
  differing only in the library:

| Arm | Actors' debug placed (of 10) | Actors reaching 10/10 | Mean $ per actor | $ per 10/10 program | Held-out of the arm's pick |
|---|---|---|---|---|---|
| With the 4 skills | 10, 10, 10, 10, 10, 9 | 5 of 6 | $0.92 | **$1.10** | 19/20 |
| Without | 5, 10, 4, 8, 0, 6 | 1 of 6 | $1.12 | **$6.74** | 20/20 |

The skills transfer, as reliability rather than a higher ceiling. All six skills-arm programs segment the top face
by brightness and saturation before triangulating it (S01), and try a vertical descent first (S03). Of the six
no-skills programs, one uses saturation and two descend vertically first. Some actors cite the library in their
notes ("the skill-library rule against using the bbox centre was applied from the start"). Without the library, most actors
spent their budget rediscovering those failures. The best program from each arm is about as good (19 vs 20 of 20);
the library mostly cuts the search cost per working program, about 6x here.

**Primitives or search?** The 0/10 no-skills actor tried four versions in 10 calls. In them, it:
- estimated box height by assuming depth equals height;
- let its pixel mask latch onto the white jaw;
- approached below the tops of the tall boxes.

These are the failures S01 and S04 describe. Other actors with the same primitives placed 10/10, so the bottleneck
is search (budget and starting hypotheses), not the primitives.

## Spend

My ledger (`state/ledger.jsonl`, every API call) totals **$29.66** for this experiment. Offline spend covers
writing, debugging, verifying and evaluating; run-time cost is what one episode of the finished program costs.

| Phase | $ | Of which |
|---|---|---|
| Setup: smoke tests, localizer comparison | 0.59 | |
| Benchmark loop: 2 rounds, 6 actor and 4 verifier sessions | **9.43** | actors 6.05, verifiers 1.54, skill writer 0.89, Claude localize during debugging 0.94 |
| Benchmark held-out and zero-shot variant evaluations | 0.70 | Claude localize |
| Transfer study: 12 actors, diagnostics, variant held-out | 18.77 | actors 15.61, Claude localize 3.16 |
| Run-time perception runs (Jev, H2O) | 0.18 | Jev (estimated price), H2O free |

- **Tokens:** 1.21M input, 0.66M output, 7.21M cache reads, 2.00M cache writes.
- **Lost to shutdowns:** about $3.8 of actor spend went to sessions killed when the machine went down (three times).
- **Run time:** $0.014 per episode with Claude perception (4.3 calls × $0.0032). With H2O it is $0, and nothing in
  the loop needs a large model.
- **Jev pricing:** Jev calls are counted at a placeholder $0.002 each; I don't have TypeSafe's actual price.
- **Comparison:** the whole benchmark loop, skills included, cost $9.43 of offline spend, once. The
  Claude-drives-the-arm team ([dliu99/valet#5](https://github.com/dliu99/valet/pull/5)) pays per episode instead,
  and placed 11/20.

## What this says about the doubt

**"Code-as-policy written by hand doesn't generalize."** Here, the program written by agents from traces does: 10
debug scenes were enough for 19/20 held-out, and the same program places 18/20 on a task variant it never saw.

**"There is no right grasp; it should be learned."** The grasp here was not given. The primitives contain no grasp
logic. The agents found from traces that they had to:
- measure the box top by triangulation;
- align the jaws to the narrow side;
- offset for the fixed jaw;
- keep the grasp and place pitch equal.

They wrote that down as verified skills, and those skills transferred.

**"Get to where a small local model can approximate it."** This is where code-as-policy fits. The expensive model
works offline, once per task, about $10 here. At run time the program asks a few box questions that Jev or a local
4B model can answer. The program depends on box quality, not on which model drew the box: Claude 19/20, Jev 15-17/20,
local H2O 20/20. With 20 scenes, the 19 and 20 are a tie; Jev's looser boxes cost two to four scenes.

## Caveats

- **One task family, simulation only.** The sim's servos, friction and camera are fitted to the real arm but not
  identical (see `sim/README.md`). The debug set is 10 scenes.
- **Placement margin is thin.** Placed boxes land 1-2.5 cm from the square's centre against a 3 cm limit. On the
  real arm, calibration error eats into that.
- **The transfer comparison is small.** It is 6 actors per arm and one variant. The difference in reaching 10/10
  (5/6 against 1/6) is large, but the held-out ceiling is the same.
- **The Jev `found` fix is post hoc**, as described above. The H2O run uses the fixed semantics.
- **Localize answers are cached** by exact frame. Identical frames give identical answers, which also makes runs
  deterministic (re-running the final program after merging `main` reproduced 19/20 with every answer from cache).
  Costs are reported for calls actually made.
- **H2O latency.** In the held-out run, H2O's box questions took a median 3.2 s (p90 3.5 s; bf16 weights, one
  image per question), slower than the 0.4-0.9 s quoted for text. An episode spends about 14 s waiting on
  perception. Results don't depend on latency here, because simulated time stops while a question is answered; on
  the real arm, the arm would wait.

## Reproduce

```bash
cd experiments/aspire && uv sync
uv run pytest
uv run aspire eval programs/benchmark_final.py --task benchmark --seeds heldout --backend jev --out state/heldout/x
uv run aspire evolve --name mine --k 3 --rounds 5             # the loop (Claude API; writes skills/)
uv run aspire evolve --name t --task variant --freeze-skills  # the transfer arm; --no-skills for the control
uv run python -m aspire.reference --task variant --seeds heldout   # the expert on any seed set
uv run python -m valet_sim.scoreboard results
```

Keys come from `/Users/devin/projects/valet/.env` (`ANTHROPIC_API_KEY`, `TYPESAFE_API_KEY`) and are never printed.
`state/` (traces, keyframes, sessions, ledger) is git-ignored. `results/` keeps the archives, the evaluation
summary (`summary.json`), the expert reference and the scoreboard episodes.
