# Claude runs the arm: report

**Question.** Given tools, can a frontier model pick and place on the valet-sim benchmark, and at what cost and latency?

**Answer.** Mostly. Claude Opus 5.5, with the wrist camera only, lifted the box in **20/20** episodes and placed it in
**11/20**. The scripted expert, which reads the box's true pose, lifts 15/20 and places 13/20. A median episode took
**79 s** and **21 Claude calls** and cost **$0.29**. Six of the nine misses have one cause: Claude grips the box low,
the box swings in the jaws and lies down when released, and the top-down wrist view doesn't show it.

## Headline (final configuration, wrist camera only, seeds 0-19)

| | Claude Opus 5.5 | scripted expert (reads true pose) |
|---|---|---|
| placed | **11/20** | 13/20 |
| lifted | **20/20** | 15/20 |
| median seconds per episode | 79 s wall (75 s of Claude calls); ~101 s with arm motion added | n/a |
| Claude calls per episode | 21.4 mean (14 median when placed, 29 when missed) | 0 |
| Jev calls per episode | 0 (Jev did not help; see below) | 0 |
| tokens per episode | 269k input (248k cache reads, 20.6k cache writes, 45 uncached), 6.9k output | |
| dollars per episode | **$0.29** Claude (placed episodes $0.19); Jev $0.00007 a call when used | |

- Seconds: the sim runs in simulated time and pauses while Claude thinks, so wall time is nearly all model latency
  (median call 3.3 s, p90 7.8 s, max 20 s). The arm itself moved for a median 19 s of simulated time. On a real arm
  those add up: a median of ~101 s per episode.
- 428 Claude calls in the 20-seed run; each turn carries about two tool calls (moves, then a look). Per episode:
  20.8 moves, 13.6 looks, 4.0 gripper calls. No refusals or fallbacks in 1,760 calls across the experiment.
- Cost: output tokens are about half the bill. Images are cheap because the loop is append-only and everything but
  the newest frame is a cache read ($0.20/Mtok).
- Spend: the 20-seed run cost $5.83. The whole experiment cost $27.25 in Claude calls (including one run lost to a
  machine shutdown), plus under a cent of Jev.

## The setup

Claude Opus 5.5 (`claude-opus-5-5`) runs at effort medium with adaptive thinking and server-side refusal fallbacks
on. It gets the goal as one sentence in the first user message, along with the first wrist frame and the joint state:

> Pick up the white box and set it down upright on the blue square, then let go and move the jaws clear of it.

The system prompt is three sentences: you control the arm through tools, you plan and judge everything, calls in a
turn run in order and a failed call skips the rest, and say in one sentence when you are done or can't finish. It has
no stages, no grasp choice and no rules. The tools (`agent.py`):

- `look`: the wrist frame (640x360, upright) plus joint angles, `grasp_point` in metres in the base frame, and servo
  currents.
- `move`: either a grasp-point target `xyz` with `pitch_deg` and `roll_deg`, solved by IK, or raw joint goals. It
  waits for the arm to stop (at most 4 s) and returns the state.
- `gripper`: `close` squeezes until the jaws stall and reports where they stopped. `open` opens to a given percent.

Claude chose every grasp: approach pitch, wrist roll to line the jaws up with the box's yaw, grip height, and when to
retry. It also queued moves in one turn ("open, descend, close, lift, look"). The episode ends when Claude answers
without a tool call, or after 40 calls.

Two tool changes survived iteration:

1. **IK errors say what would work.** "No joints put the grasp point at [...] with the jaws 90 deg below
   horizontal; pitches from 0 to 55 deg reach it." Without this, the failing baseline episodes spent 7-13 calls each on unreachable targets.
2. **A calibrated grid on the frame** (`overlay.py`). The overlay is a red 5 cm grid on the table, labelled x,y in cm
   in the base frame, plus a ring at the grasp point and a + on the table below it. It is computed from joint state,
   the arm's kinematic model and the wrist camera's pose and lens. A calibrated real arm has all of these, and the
   overlay never uses the box's or the square's pose. With it, Claude reads positions off the table ("box at about
   x 22, y 15") instead of reasoning about perspective.

A policy reads only `Sim.state()` and frames. `truth()` and `score()` run only in `run.py`, after the episode.

## Iteration (seeds 0-4, one change at a time)

| run | placed | lifted | median s | Claude calls | Jev calls | input tok | output tok | $ Claude | per seed | kept? |
|---|---|---|---|---|---|---|---|---|---|---|
| v0 baseline, wrist + side frames | 2/5 | 4/5 | 196 | 30.6 | 0 | 649k | 12.1k | 0.56 | P L L P - | |
| v1 v0 + IK errors name reachable pitches | 3/5 | 4/5 | 106 | 24.6 | 0 | 464k | 9.1k | 0.42 | P L P P - | yes |
| v2 v1 + grid and grasp-point overlay | 4/5 | 5/5 | 131 | 23.8 | 0 | 393k | 8.8k | 0.40 | P L P P P | yes |
| **w0 v2, wrist camera only** | 3/5 | 5/5 | 75 | 24.0 | 0 | 336k | 7.8k | 0.33 | P P L P L | new base |
| w1 w0 + 4 mm near clip on the wrist camera | 1/5 | 5/5 | 191 | 28.6 | 0 | 485k | 11.5k | 0.47 | L L L P L | no |
| w2 w0 + Claude may attach Jev checks to calls | 2/5 | 5/5 | 79 | 23.4 | **0** | 344k | 8.0k | 0.34 | L L P L P | no |
| w3 w0 + goal says "centred on" | 3/5 | 4/5 | 84 | 23.2 | 0 | 325k | 9.3k | 0.37 | P P P L - | no |
| w4 w0 + Jev verifies when Claude says done | 4/5 | 5/5 | 166 | 34.2 | 1.2 | 626k | 12.8k | 0.55 | P P L P P | no, see below |
| w5 w0 at effort low | 2/5 | 4/5 | 83 | 24.0 | 0 | 304k | 6.2k | 0.29 | L P L P - | no |

P placed, L lifted only, - neither. The full table with token detail is `results/table.md`, and the per-seed results
are in `results/*.jsonl`.

- **Wrist only.** Partway through, the brief changed to wrist camera only, since the real arm has no side camera. On
  these five seeds, dropping the side view went from 4/5 to 3/5 and made episodes cheaper and faster. Five seeds can't
  separate those two.
- **Noise is about ±1 placement in 5.** w2 and w4 behave like w0 for scoring purposes (Claude never wrote a Jev check
  in w2, and Jev changed no outcome in w4), yet they scored 2/5 and 4/5 against w0's 3/5. The kept changes above made
  big, consistent differences in calls (30.6 to 24) and in cost. Smaller effects need more seeds than this budget
  allowed. The final run reproduced w0 on seeds 0-4 (3/5).
- **w4 was not kept, even though placed went from 3 to 4.** Jev's answers explain none of that difference. It said yes
  on three correct placements and yes on a box lying on its side (seed 2). On seed 4 it said no twice to a correct
  placement, which cost two extra Claude calls before Claude replied that "the automatic check is wrong". The extra
  cost and time in that run came from Claude's own retries, not from Jev.

## Where it fails (the 9 misses in the 20-seed run)

| seeds | what happened | count |
|---|---|---|
| 3, 5, 9, 14, 15, 16 | the box ended **lying on its side** on or beside the square; Claude reported it "upright" each time | 6 |
| 8, 19 | set down **3.1-3.9 cm** from the square's centre (limit 3 cm); Claude judged it "on the square". Seed 15 also landed 4.3 cm off | 2 |
| 0 | three grasp attempts, ran out of calls (40) with the box 7 cm off and the jaws still touching it | 1 |

The tip-overs come from where Claude grips. Tipped boxes were gripped with the grasp point at 36-66% of the box's height
(mean 46%). Placed boxes were gripped at 45-105% (mean 65%). A low grip lets the box pivot in the jaws, so it is set
down on an edge and falls when the jaws open. The point-and-reach team found the same thing with perfect pointing:
gripping 2.5 cm below the top instead of 3 cm above the table took placement from 17/20 to 20/20. From above, the
wrist camera shows a lying box as a slightly longer white rectangle. Claude read every one as upright, and so did Jev
(below).

The off-centre placements are an estimation problem. Claude reads the square's centre off the grid from a distance,
then carries the box over it. During the set-down the box hides the square, so the final correction is made blind.

## Jev

Jev (TypeSafe System One, `jev-latest`) answered in 0.1-0.2 s for about 1,600 input tokens, roughly $0.00007 a
question. It was tried two ways, and neither saved Claude calls or improved placement:

1. **Claude-written checks** (w2). `move` and `gripper` took an optional `check`: a yes/no question for Jev about the
   frame after the call, where "no" skips the rest of the turn. Claude never used it in 5 episodes and 117 calls.
   Claude's own `look` already runs inside the same turn, so a check saves no Claude call unless Claude would otherwise
   stop to look. It didn't.
2. **Verify between turns** (w4). When Claude says it is done, Jev answers "Is the white box standing upright on the
   blue square (not lying on its side), with the robot's jaws clear of it?", and a "no" sends Claude back. It changed
   no outcome and cost two calls on a false "no".

Scored offline on the 20 final wrist frames against the grader (`claude_policy/jev_eval.py`, output in
`results/jev-eval-final.txt`; two identical evaluations differed by one answer near 0.5):

| question | right at p = 0.5 | AUC |
|---|---|---|
| upright on the square with jaws clear (the verify question) | 7/20 | n/a |
| upright (not lying on a long side) | 13-14/20 | 0.71 |
| on the blue square, covering its middle | 17/20, with 3 false yes | 0.60 |

From this top-down view, Jev can't tell a standing box from a lying one well enough to gate on. That is the same
failure Claude has. The point-and-reach team reports Jev "held?" at 24/24 on a side-camera crop, so the view, not
Jev, is the limit here.

## Fidelity caveats

- **The shipped sim lets the wrist camera see through a held box.** The wrist camera's near clip plane is 3.2 cm, so
  the box between the jaws is cut away. The jev-policy team found this. On a real arm, the held box would fill the
  lower frame and hide the square during placement. With a 4 mm near plane (w1), Claude said "the box is blocking my
  view of the blue object" and placed 1/5 against 3/5. The headline uses the sim as shipped, so it compares with the
  expert and the other teams, but it likely flatters the placement step.
- **The sim pauses while Claude thinks.** A held box can't slip during a 20 s call. On the real arm, the servos would
  have to hold it.
- **The overlay assumes calibration.** It needs the real wrist camera's intrinsics and mount pose; valet's
  `hardware/so101.py` lists that calibration. The sim's joint zeros are about 5 degrees off the real arm's, so a real
  overlay must use the real calibration.

## Next step

**Let Claude learn the grasp from its own outcomes, on practice seeds.** Run practice episodes on non-benchmark
seeds (20 and up). After each one, show Claude what happened: the graded result, or at least a final frame from a pose
that shows tilt. Let it keep a short notebook of what to do differently, then carry the notebook into the benchmark.
The dominant failure, a low grip that lets the box swing and tip, is exactly the kind of thing the brief says should
be learned rather than written as a recipe. The fix is one sentence of experience Claude doesn't have today. Run this
together with the near-clip fix in `sim/scene.py`, so placement is measured with the held box occluding the view as
it will on the arm.
