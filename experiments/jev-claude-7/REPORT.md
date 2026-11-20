# jev-claude-7: Jev measures, Claude plans and learns

On the valet-sim benchmark (seeds 0-19, wrist camera only), this policy **placed the box 16/20 and lifted it 18/20**.
The scripted expert, which reads the true box pose, places 13/20 and lifts 15/20. A median episode took 35 s, with
**2.3 Claude calls** ($0.12) and 18 Jev calls ($0.0008). Once its own lessons had settled (seeds 8-19), it placed
11/12 with no replans and two Claude calls per episode: one plan and one lesson. With a local H2O-Lightning-4B doing
all the measuring in Jev's place, it placed 4/5 on seeds 0-4, the same as with Jev; the only paid model left is about
two Claude calls an episode.

| seeds 0-19, wrist camera only | placed | lifted | median s / ep | Claude calls / ep | Claude $ / ep | Jev calls / ep | Jev $ / ep |
|---|---|---|---|---|---|---|---|
| **Opus 5.5 plans, Jev measures** | **16** | **18** | 35 | 2.3 | 0.125 | 18.3 | 0.0008 |
| Haiku 5.5 plans, Jev measures | 0 | 1 | 57 | 4.0 | 0.0095 | 29 | 0.0012 |
| scripted expert (reads the true pose) | 13 | 15 | - | 0 | 0 | 0 | 0 |
| Opus 5.5 plans, local H2O-Lightning-4B measures (seeds 0-4 only) | 4 of 5 | 5 of 5 | 275 | 2.2 | 0.105 | 15.6 | 0 (local) |

The scoreboard rows are in `results/episodes.jsonl` (`python -m valet_sim.scoreboard experiments/jev-claude-7/results`),
with Claude as `slow_calls` and Jev as `fast_calls`.

## Design

Each part does what it is good at. Claude is never asked to drive individual moves.

**Jev measures.** Jev judges where things are in an image well, and distance poorly. So it is only asked "where is
it in this frame": one System One request per frame with three Box questions (the white box, its top face, the blue
square). The Jev runs also asked two visibility yes/no questions the policy never used; they were dropped for the H2O
run. Code turns those 2D boxes into table positions with the arm's own model:

- Forward kinematics of the joint readings gives the camera's pose.
- The fitted lens (pinhole plus barrel distortion) gives each pixel's ray, and the table plane closes the geometry.
- The camera is moved to look straight down at each object, centred; at pitch 66 the wrist camera points straight
  down. In a centred, vertical view the box's position barely depends on its unknown height.
- One more look from higher up and one from the base side give the box's height and size. A square-prism model
  (centre, side, height, yaw) is fitted to all of Jev's boxes of the box at once.
- The square never moves, so every look aimed at it is fused, and looks that disagree with the rest are dropped.

Graded after the fact, the first box estimate was off by a median of 0.3 cm (worst 1.0 cm) and the square by 0.4 cm
(worst 1.8 cm), over the 20 episodes of the Opus run. The height is rougher: median 1.0 cm low, worst 3.2 cm.

**Claude plans.** One call writes the episode's whole plan as JSON. It contains:

- moves of the grasp point G (between the jaws), relative to the box's or the square's measured centre, each with
  pitch, roll and gripper;
- grips with the reading Claude expects: a jaw stalled on the box reads its width, so a miss shows up at once;
- yes/no checks for Jev, written by Claude;
- re-measuring looks.

The prompt gives facts only: the frame, the hand's geometry, where G can reach, and how good the measurements are. On
the hand, it says the fixed jaw is 2 cm from G, how far the moving jaw reaches at each gripper value, and that the
jaws close along the bearing plus the roll. The prompt holds no grasp: no approach, offset, grip height, order of
stages or retreat.

Code runs the plan: IK, straight-line moves, stall detection, and a note when a move stops short because something is
in the way. It stops at the first failed step: a grip reading outside the expected range, a check answered the wrong
way, or an unreachable pose. Claude then gets the log and the current frame and plans again from where the arm is,
at most twice.

**Claude learns.** After each episode the policy measures its own result the same way it measured the scene: back
to the home view, find the box and the square, and compare their centres; a box that comes out much lower than it
stood has tipped. Claude then writes a one- or two-sentence lesson from its log and that self-check, and later
episodes see the last 8 lessons. The score is never shown to the policy. The self-check agreed with the real grade
in 40 of the 40 full-run episodes.

The grasp is learned, not hardcoded, and it is learned in plain sentences that a person can read and correct.
`results/lessons-opus.md` holds the 20 lessons from the Opus run. By about seed 8 they settle on one plan shape:

1. Look at the square, then at the box.
2. Set the roll from the box's edges in the overhead frame, and put G 1.4-1.6 cm against the closing direction.
3. Hover at pitch 60 and z 12, descend at pitch 90 to z 3, and grip to 0, expecting a reading of 15-50.
4. Carry at pitch 60, and lower at pitch 90 to z 4.5, shifted about 1 cm.
5. Open to 60 and rise.
6. Ignore Jev's upright and on-the-square scores.

Claude wrote that last point itself, after its checks of the hold and of the placement had come back "no" on good
episodes.

## Results

**The Opus run learns as it goes.**

| | seeds 0-7 | seeds 8-19 |
|---|---|---|
| placed | 5/8 | 11/12 |
| Claude calls / episode | 3.0 | 1.8 |
| replans / episode | 1.0 | 0 |

The one miss in seeds 8-19 is seed 16, a bug described under "Where it fails". Run in order, seed by seed, the run
cost $2.49 in Claude and $0.016 in Jev for 20 episodes. Wall time per episode is mostly Claude (median 28 s of the
35 s). The simulated arm moved for a median 37 s of sim time.

**Haiku 5.5 cannot plan this.** It placed 0/20 and lifted 1/20, at a twelfth of the cost. Its measurements were as
good, so the failure is the plan:

- it offset G toward the wrong jaw;
- it stacked Jev checks that answered "no" on good grasps, which used up both replans in every episode;
- its lessons doubled down on those checks.

**Local H2O-Lightning-4B in Jev's place.** On seeds 0-4 with Opus planning, swapping Jev for H2O-Lightning-4B served
on this Mac placed 4/5 and lifted 5/5. The Opus + Jev run placed the same 4/5 on those seeds. Its measurements were as
good or better: the first box estimate was off by a median 0.2 cm (worst 0.2), the height by +0.4 cm (worst 0.9), and
the square by 0.4 cm (worst 0.6). The miss was seed 4, set down 4.5 cm off. The self-check measured 4.6 cm and flagged
it.

The cost is time. A frame's request (three Box questions) took a median 16 s against Jev's 0.09 s. Perception
therefore took about 250 s of each 275 s episode, against under 2 s with Jev. Team 11 was running its own 20-seed
benchmark on the same H2O server the whole time, so these times include contention; uncontended, H2O is advertised
at 0.4-0.9 s a question. Either way, nothing in the policy needs Jev specifically: a local model that can draw boxes
does the measuring. The Claude calls per episode stay at about 2.

## Variants on seeds 0-4 (Opus, Jev)

One change at a time. Each change was kept only if placed went up, or held while something else improved.

| variant | change | placed | lifted | median s | Claude calls / $ per ep | replans / ep | kept |
|---|---|---|---|---|---|---|---|
| v1 | Claude plans, Jev checks, replans | 4 | 4 | 60 | 2.4 / 0.15 | 1.4 | yes |
| v2 | + lessons between episodes, measured self-check | 4 | 5 | 80 | 3.2 / 0.20 | 1.2 | yes (lifted 4 to 5) |
| v3 | + fuse every look at the square, drop outliers | 4 | 5 | 58 | 2.8 / 0.14 | 0.8 | yes (faster, cheaper, fixed v2's miss) |
| v4 | + re-measure the box from the home view; size bounds on the fit | 3 | 5 | 64 | 3.0 / 0.18 | 1.0 | no, reverted |
| v5 | + lessons from 5 practice episodes on seeds 100-104 | - | - | - | - | - | cut short by a machine shutdown (seeds 0-3: 1 placed); not rerun under the run cap |

The full runs use v3.

Two fixes were made outside the variant chain. Neither changes how the policy decides:

- `grip` declared a stall before the slewing servo command had reached the jaw, so the jaws never opened.
- Moves now keep the gripper's goal, not its reading, so a held box isn't let go.

Variance is large. Seed 0 was placed in v1 and v2 and tipped in v3 and v4 with the same code. A 5-seed comparison
only shows big effects.

## Where it fails

In the Opus run:

- **Seed 0, tipped.** This is the first episode, so it had no lessons. Jev's "is the box held?" answered no on a
  good grasp (p 0.39, then 0.46), which used up both replans. Claude had gripped an 8.4 cm box 3.5 cm above the table,
  and the box swung and fell over when set down.
- **Seed 6, not lifted.** The descent at pitch 90 pushed the 9 cm box and the jaws closed on nothing. Each retry
  stopped on the now-fallen box.
- **Seed 7, off by 4.6 cm.** The square is at the edge of reach (bearing -40). The self-check measured 3.4 cm and
  flagged it, but nothing acts on the self-check yet.
- **Seed 16, no plan: a bug.** The square's first guess was cast from the latest frame, which was the side view of
  the box, not the overview, and the square was never found. This is fixed: the guess now comes from the overview,
  with a fallback to it. Measuring seed 16 again without Claude finds the square 0.7 cm off. The episode was not
  rerun, because of the run cap.

Across all runs:

- **Jev's yes/no about spatial relations in wrist frames is near chance.** Asked "are the jaws around the box?", "is
  the box held?" or "is it upright on the square?", it said no at p 0.2-0.46 when the answer was yes. Those false
  stops were most of the replans until the lessons dropped the checks. Its 2D boxes, by contrast, were reliable
  throughout.
- **The measured height runs about 1 cm low** (worst 3.2 cm). Claude's lessons compensate: they lower to z 4.5 and
  accept a move that stops short.
- **Tipping is caught only roughly.** The self-check spots a box lying on its side when it looks clearly lower than it
  stood. On expert episodes it caught one of the two boxes that tipped over on the square and missed the other.

## How it differs from the other teams

- **Team 1, Jev picks the moves (0/20, PR #4).** Here Jev never chooses an action. It only answers "where in this
  frame", the question it is good at, and geometry turns that into centimetres.
- **Team 2, Claude drives with look/move/gripper tools (11/20 at $0.29 and 21 calls an episode, PR #5).**
  - Here Claude never reads a position off pixels. It gets centimetres measured by Jev and code, and writes the whole
    plan in one call, relative to those measurements.
  - Claude is called 2-3 times an episode, not about 20.
  - It learns between episodes from a measured self-check, so its grip-height and tipping problems got fixed by its
    own lessons rather than by prompt changes.
- **Team 4, point-and-reach.** Similar perception, a model's image answer plus the camera model, but there the reach
  and grasp are code. Here they are Claude's choice and are learned. Perception here uses Jev's boxes from several
  centred looks rather than one point.
- **Team 5, Physical-Agent end-effector sequences (at most 5/20, PR #3).** Also a plan of end-effector targets, but:
  - targets are relative to measured objects, not to pixels Claude converts itself;
  - every grip and check is an assertion code tests, with a replan on failure;
  - plans improve through lessons.
- **Team 6, ASPIRE skill library.** Both learn across episodes. ASPIRE writes code; here the learned skill is a
  handful of plain sentences, and the code stays fixed.
- **Team 8, another Jev + Claude design.** Here Jev is purely the measuring instrument, and Claude both plans and
  writes its own lessons.

## Next step

Take Claude out of the per-episode loop. From seed 8 on, the plan was the same shape every time, with numbers that
follow from the measurements. So:

1. After practice, have Claude write that shape once as a plan template: a function of the measured box and square.
2. Run episodes from the template with no Claude call.
3. Call Claude only when a step fails or the self-check misses, to revise the template.

That turns a slow model into a coach consulted a few times per hundred episodes. It is the user's target: a cheap
local model, or plain code, executes; Claude teaches. Two smaller changes would close the failures above: let a
self-check miss trigger a re-place (seed 7), and give the descent a higher hover before a straight-down approach to
tall boxes (seed 6).

## Cost

- Benchmark runs: Opus $2.49, Haiku $0.19 and the H2O run's Opus $0.52 in Claude; Jev under $0.05 in total.
- The whole experiment cost about $8 in Claude ($7.88 recorded), including five-seed variants, practice and a run
  lost to a shutdown.
- Under the user's later cap, three runs were made, for $3.21 in Claude: Haiku on seeds 0-19, Opus on seeds 0-19,
  and Opus with local H2O on seeds 0-4 ($0.52).
- No server was started for the H2O run: it used the one team 11 had running, and left it running.

## Reproduce

```bash
cd experiments/jev-claude-7 && uv sync && uv run pytest     # offline, no API calls
python3 ~/.valet/slot.py --name team-7 -- uv run python -m jc7.run --seeds $(seq 0 19) --name full \
    --model claude-opus-5-5 --lessons runs/full/lessons.md --config "v3, opus plans, jev"
uv run python analyze.py runs/full
```

`--fast h2o` swaps Jev for H2O-Lightning-4B served at `127.0.0.1:8741`.
