# Jev picks closed-loop skills

Can Jev drive the arm if it picks among a few closed-loop skills instead of low-level moves? On valet-sim seeds
0-19, wrist camera only, it **placed 20/20 and lifted 20/20**. Each episode took a median of 14 s, with 19 Jev calls
(0.1 s each) and 1 Claude Haiku call. With Claude removed entirely, it placed 17/20 with zero slow calls. Jev picking
low-level moves placed 0/20. The scripted expert, which reads the true box pose, places 13/20.

| seeds 0-19, wrist only | placed | lifted | ended on a correct `done` | median s/episode | fast calls/ep (median) | slow calls/ep (median, mean) | $/episode |
|---|---|---|---|---|---|---|---|
| **Jev picks skills; Claude Haiku writes the goal and re-plans after 2 failures** | **20/20** | **20/20** | 20/20 | 13.8 | 19 (Jev, 0.10 s each) | 1, 1.75 (Haiku, 1.5 s each) | 0.0014 |
| Jev picks skills, no Claude at all | 17/20 | 17/20 | 15/20 | 11.9 | 20 | **0** | 0.0013 |
| H2O-Lightning-4B (local) in place of Jev, with Claude | 19/20 | 20/20 | 14/20 | 146 | 0: at 3.8 s each its 34 calls count as slow | 37 (34 local + 3 Haiku) | 0.0006 |
| scripted expert (reads the true pose) | 13/20 | 15/20 | | | | | |
| earlier: Jev picks low-level moves, Claude writes goals (jev-policy) | 0/20 | 10/20 | | 149 | 80 | 10.8 | 0.066 |
| earlier: Claude drives with tools (claude-policy) | 11/20 | 20/20 | | 79 | | 21.4 | 0.29 |

Seconds per episode are simulated arm time plus model time: what the real arm would take, since the arm holds still
while a model answers. Wall clock was 5 s per episode with Jev. Every episode is recorded in `results/episodes.jsonl`;
`python -m valet_sim.scoreboard experiments/jev-skills/results` prints the table. All 95 episodes behind this report
cost $0.12 in API calls. Per-decision logs are in `runs/<config>/seed*/events.jsonl`; frames were not committed.

**Most of the score comes from the skills, not from Jev's picks.** With truth picking the skills (a diagnostic,
not recorded), the same library placed 5/5 on seeds 0-4. What Jev contributes is a sequence that tolerates its own
mistakes:

- A wrong pick is refused by the skill's precondition without moving the arm.
- The refusal goes into the history, and Jev picks again.
- Jev picked an acceptable skill 75% of the time, and that was enough.

**Claude's one useful job is breaking loops.** Without Claude, none of the three misses ever grasped:

- two picked `place` 13 and 14 times out of 14 decisions, refused each time;
- the third cycled between looks and `place`.

With Claude, the re-plan after two failures broke exactly these loops (0.75 re-plans per episode).

## The loop

`jev_skills/runtime.py`. Per decision:

1. **Judge.** If a skill just ran, Jev answers one yes/no question on the new wrist frame: is that skill's done
   criterion true? It also sees the state and the skill's report.
   - Yes (p ≥ 0.7) goes into the history as "Done test: held".
   - No (p ≤ 0.3) goes in as "NOT held".
   - Unsure: the skill's own code test decides, and the history says so.
2. **Choose.** Jev answers one choice question: which skill runs next? The options are the skills plus `done`; a
   parameter is part of the option name (`grasp_near_top`, `grasp_middle`). Jev sees the wrist frame and this state:
   - `goal`: Claude's sentence, or the task sentence when there is no Claude;
   - `arm`: the jaws (open and empty / closed on something, with the reading / closed on nothing), the jaw height,
     and whether the jaws are over the box or the square;
   - `memory`: where the box and the square were found, and whether the box was measured or released on the square;
   - `history`: the last six skills, each with its report and done verdict.
3. **Run.** The skill runs closed-loop until its own test passes or it gives up, then reports what it measured.
4. **Done.** If Jev picks `done`, it judges the final criterion, "The white box is standing on the blue square."
   - Yes ends the episode.
   - Unsure ends it only if the place skill's own check passed: the box was released, its top was seen within 3 cm
     of the square's centre, and the top face looked square.
   - No goes into the history. After two refusals the episode ends.

Claude Haiku 5.5 (effort low) is called at most 5 times per episode. It writes the goal from the first frame. When
two skills in a row fail, it re-plans: it picks the next skill, and its one sentence of advice stays in the state.

The limits are 14 decisions and 150 s of simulated time. The policy reads only joint readings, the gripper's
commanded goal, wrist frames, its own inverse kinematics and its own camera model. `sim.truth()` is read only after
a decision, to label whether the pick and the verdict were right, and to grade the episode.

## The skill set and the done criteria

`jev_skills/skills.py`. Perception is fast:

- Jev draws a box around "the white box" or "the blue square".
- Code refines inside that box: the top face's pixels for the box (gray > 196, low saturation, minus the arm's own
  jaw), the blue pixels for the square.
- Both are back-projected through the calibrated camera.

The mechanics borrow four of the aspire team's validated skills:

- S01: triangulate the top face's centre;
- S02: distrust the square when it is seen around the held box;
- S03: grip and place at vertical pitch;
- S04: re-measure after a blocked descent.

| skill | Jev's "use when" (the option text) | what the code does | done criterion Jev judges |
|---|---|---|---|
| `look_workspace` | `memory` says the white box or the blue square is not located yet | goes back to the start pose; one Jev call boxes both objects; the box top and the blue pixels are back-projected | Both the white box and the blue square are visible in `wrist`. |
| `look_box` | the jaws are empty and the white box's height is not measured | takes two looks down from 6 cm behind the box, 7 cm apart side to side; triangulates the top face's centre across every view for the height; fits the top rectangle at that height for centre, turn and size; tries again if the two views disagree by over 12 mm | `wrist` looks down on the white box from above, with the whole box in view. |
| `look_square` | the blue square's position is only rough | looks down on the square; accepts only a full square that the frame edge does not cut off; while holding the box, accepts it only within 2.5 cm of the clear view | The whole blue square is in view in `wrist`. |
| `grasp_near_top` | the jaws are empty and the box was measured from above: grip it 2.5 cm below its top | rolls the jaws to the box's sides, with the fixed jaw 8 mm clear of the near face; descends at pitch 90 first; if blocked more than 1.5 cm high, backs off and asks for a re-measure; closes and checks the reading is over 3; lifts 6 cm and checks again | The jaws are holding the white box up off the table. |
| `grasp_middle` | the same, gripping halfway up | as above | as above |
| `carry` | the jaws are closed on something and not above the blue square | moves to the drop point, offset for where the box sits in the jaws; re-checks the square (S02) | The jaws are holding the white box above the blue square. |
| `place` | the jaws are closed on something above the blue square | lowers to 1.2 cm above the grip height at the grasp pitch; eases off if stopped; opens, lowering and re-opening up to twice; rises 5 cm; looks: box top within 3 cm of the square's centre, top face square | The white box is standing on the blue square. |
| `done` | `memory` says the white box was released on the blue square | ends the episode (see the loop) | The white box is standing on the blue square. |

`back_off` (open the jaws and raise them) is in the library but out of the final menu; see h3.

## Hillclimb on seeds 0-4

One change at a time. A change is kept if placed goes up. Ties are broken first by episodes ending on a correct
`done`, then by slow calls, then by the right-skill rate. Every row is recorded in `results/episodes.jsonl`.

| # | change | placed | lifted | correct `done` | right skill | Jev verdict right / unsure | fast, slow calls/ep | median s | kept |
|---|---|---|---|---|---|---|---|---|---|
| h0 | baseline: 8 skills; "use when" as prose ("Hover above the box... use when...") | 2/5 | 3/5 | 0 | 33% | 53% / 31% | 34, 2.2 | 16 | |
| h1 | "use when" rewritten as preconditions on the state fields, and the state words to match ("open and empty", "measured from above", "released on the square") | **5/5** | 5/5 | 1 | 55% | 64% / 28% | 30, 2.0 | 25 | yes |
| h2 | `place` ends 6 cm back from the square, looking at the box on it | 3/5 | 4/5 | 0 | 41% | 63% / 26% | 32, 2.0 | 24 | no: from that oblique view the code's tip check read an upright box as lying down, and Jev re-grasped a placed box |
| h3 | drop `back_off` from Jev's menu (picked 68 times in h0-h2, never right) | 5/5 | 5/5 | 0 | 75% | 61% / 29% | 26, 2.0 | 15 | yes |
| h4 | `place` criterion becomes "The white box is standing on the blue square.", the wording jev-claude-8 measured at 9/12. The rise to 24 cm meant to go with it was out of reach on every seed, so only the wording changed | 5/5 | 5/5 | 2 | 59% | 65% / 23% | 27, 2.4 | 24 | yes: more correct `done`s |
| h5 | an unsure final check falls back to `place`'s code check, as each skill's done check already does | 5/5 | 5/5 | **5** | **91%** | 63% / 30% | **19, 1.2** | **14** | yes |

h5 is the final config. Its right-skill rate is higher mainly because episodes now end on time. Before h5, most
wrong picks came after a correct place, when Jev could not confirm `done` and wandered.

## How often Jev was right (seeds 0-19)

A pick counts as right when it is in the set a sensible operator would accept, labelled from the truth after the
pick:

- before the box is measured: `look_box`, or `look_workspace` / `look_square` while something is not located;
- once it is measured: `grasp_near_top`;
- while it is held: `carry`, then `place`;
- once it is placed: `done`.

| | Jev + Claude | Jev alone | local + Claude |
|---|---|---|---|
| right skill | **75%** of 153 | 59% of 184 | 30% of 252 |
| wrong picks | `place` 23, `carry` 7, `grasp_near_top` 7 | `place` 52, `grasp_near_top` 15, `carry` 8 | `place` 72, `grasp_near_top` 52, `grasp_middle` 28, `look_square` 12, `carry` 11 |
| done verdicts right / unsure / wrong | 66% / 25% / 9% of 133 | 78% / 19% / 3% of 164 | 58% / 32% / 10% of 232 |
| "standing on the square" after a real release | yes 2, unsure 19, no 0 | yes 2, unsure 13, no 2 | **yes 19**, unsure 1 |
| says a refused skill is done | 0 of 22 | | 16 of 82 |

Jev's wrong picks:

- Most are `place` or `carry` before any grasp. The skill refuses without moving ("not run: the jaws are not holding
  the box"), and Jev picks again. Each one costs two fast calls.
- 7 (Jev + Claude) and 15 (Jev alone) were `grasp_near_top` after a correct place, instead of `done`. The grasp went
  to the box's old spot, which the benchmark keeps at least 12 cm from the square, and closed on nothing. That is
  luck, not a guard: a re-measured box would have been lifted off the square, as happened once in h0 (seed 4).

Jev's done verdicts, per skill, in the main config:

| skill | verdicts | right | unsure | wrong | the skill's own code test right |
|---|---|---|---|---|---|
| `look_workspace` | 22 | 22 | 0 | 0 | 20 |
| `grasp_near_top` | 25 | 25 | 0 | 0 | 25 |
| `carry` | 25 | 22 | 3 | 0 | 25 |
| `place` | 38 | 19 (17 of them no on a refused `place`) | 19 | 0 | 37 |
| `look_box` | 22 | 0 | 10 | 12 | 22 |
| final `done` check | 20 | 7 yes | 13 unsure | 0 | 20 |

- Jev is reliable when the frame plus the state settle the question. "Holding the box up off the table" (the gripper
  reading is in the state) and "both are visible" were 100% right.
- On "standing on the blue square" after a real release it is almost always unsure. The episode ends correctly only
  because the place skill's code check breaks the tie, and that check was right 37/38.
- It said no to "looks down on the white box from above" on views where the box was in frame at pitch 50-70. That
  criterion is badly worded for an oblique look. The runtime goes by the code check when Jev is unsure, but takes a
  no at face value, so this costs picks.

## Jev against the local model

H2O-Lightning-4B served locally (`run_local.sh`, under the shared heavy slot), in place of Jev for every question:
boxes, picks and verdicts. It placed 19/20 and lifted 20/20, but:

- **Speed.** 3.8 s per call against Jev's 0.10 s: a median of 146 s per episode against 14 s. By the scoreboard's
  rule (any model over about a second is slow), its 34 calls per episode are slow calls, so the config is not
  scalable.
- **Picks.** Right 30% of the time against Jev's 75%. It reaches for `place` and the grasps from almost any state.
- **Verdicts.** Better than Jev on the one Jev fails: it said yes to "standing on the square" after 19 of 20 real
  releases. Worse on refused skills: it said a refused `place` was done 16 times. That broke the failure counter
  Claude's re-plan waits on, and it caused its one miss: on seed 4 it picked `place` 10 times with the box held 18 cm
  from the square.

## Where it fails

Nothing failed in the main config, so these are weak points seen in the logs, not causes of lost episodes:

- **Loops of refused picks.** Without Claude, 3/20 episodes never grasped. In two of them Jev picked `place` on 13
  and 14 of 14 decisions. A refusal costs no motion, but the loop costs the episode.
- **Confirming the finish.** Jev's "standing on the square" verdict is unsure after nearly every real release. The
  place skill's code check carries the finish, and Jev sometimes reaches for a grasp again (7-15 times per 20
  episodes).
- **Bad criterion.** `look_box`'s done criterion is wrong for the view it gets: 0/22 right.
- **Overfitting risk.**
  - The top-face threshold (gray 196) was measured on seeds 0 and 3, and the hillclimb used seeds 0-4. Seeds 5-19
    were not used for any tuning and placed 15/15.
  - The scenes are simpler than a real desk: a white box, a saturated blue square, a plain table. Colour thresholds
    inside Jev's box are enough here. On the real arm the refinement needs a segmenter, which is the SAM 3 team's
    work.
- **Untested on hardware.** The sim's gripper has no backlash, slip or overload latching. Place lands 0.6-1.6 cm
  from centre on all 20 seeds, so the margin to the 3 cm limit looks real, but it has not been tried on the arm.

## What I'd do next

1. **Replace Claude's loop-breaking with code, then drop Claude.** Once a skill refuses on its precondition, take
   that option out of Jev's next choice until the state changes. Skip the done question for a refused skill, since
   it was not done; that also fixes the local model's false yeses. The no-Claude run says this is the only thing
   Claude adds: 17/20 → plausibly 20/20 at 0 slow calls. This needs one 20-seed run.
2. Then take the skill set to the real arm. The skills only use joint readings, the gripper reading and wrist frames,
   which the real SO-101 has. Re-measure the arm's own jaw mask from a real frame, and replace the colour masks
   inside Jev's box with a segmenter.

## Files

- `jev_skills/runtime.py`: the decision loop, the questions, and the grading labels.
- `jev_skills/skills.py`: the skills, their "use when" texts and their done criteria.
- `jev_skills/perception.py`: Jev boxes, refinement in code, triangulation, and rectangle fits.
- `jev_skills/arm.py`: motion, the gripper, the camera model, and the arm's own jaw mask.
- `jev_skills/models.py`: the System One client (Jev or local), the Claude client, and the ledger.
- `jev_skills/run.py`: runs one process per episode, sequentially, and records to the scoreboard. Its
  `--chooser oracle` is a diagnostic that picks from the truth; it is never recorded.
- `run_local.sh`: starts H2O, runs the local-model config, and stops the server.
- `jev_skills/accuracy.py` and `jev_skills/digest.py` summarize the event logs.

```bash
cd experiments/jev-skills && uv sync
uv run pytest                                                  # offline: a fake System One, no API calls
python3 ~/.valet/slot.py --name <you> -- uv run python -m jev_skills.run --config final-jev --seeds 0 1 2 3 4
python3 ~/.valet/slot.py --name <you> -- uv run python -m jev_skills.run --config x --no-claude --seeds 0 1 2 3 4
```
