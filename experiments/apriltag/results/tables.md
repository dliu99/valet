### Localization: 100 trials (0 failed)

cm, median (90th percentile) over seeds. *World*: estimate minus truth. *Reach*: where the grasp point truly goes
when the arm is told, through the same model, to go to the estimate, minus the truth.

**Box, detector pixels** — world / reach

| method | none | offsets | sag | mount | all |
|---|---|---|---|---|---|
| fk | 0.1 (0.3) / 0.1 (0.3) | 2.4 (3.3) / 0.4 (0.8) | 0.6 (0.8) / 0.3 (0.6) | 0.6 (1.4) / 0.6 (1.4) | 2.5 (3.3) / 1.1 (1.4) |
| fk_tri | 0.1 (0.2) / 0.1 (0.2) | 2.5 (3.6) / 0.5 (1.1) | 0.6 (0.9) / 0.4 (0.4) | 0.7 (1.2) / 0.7 (1.2) | 2.7 (3.9) / 0.6 (2.0) |
| tag | 0.3 (0.5) / 0.3 (0.5) | 0.7 (2.1) / 2.3 (3.3) | 0.6 (1.1) / 0.3 (0.9) | 0.4 (0.8) / 0.4 (0.8) | 1.1 (2.3) / 2.1 (3.2) |
| mount_cal | 0.1 (0.3) / 0.1 (0.3) | 1.6 (3.2) / 2.6 (3.3) | 0.9 (1.0) / 0.9 (1.1) | 0.2 (0.4) / 0.2 (0.4) | 1.7 (3.7) / 2.6 (3.3) |
| cal | 0.2 (0.3) / 0.2 (0.4) | 0.2 (0.6) / 0.2 (0.6) | 0.2 (0.5) / 0.2 (0.3) | 0.2 (0.4) / 0.2 (0.4) | 0.4 (1.0) / 0.3 (0.6) |
| cal_tag | 0.3 (0.5) / 0.3 (0.5) | 0.4 (0.7) / 0.4 (0.8) | 0.5 (1.0) / 0.5 (0.8) | 0.4 (0.7) / 0.4 (0.7) | 0.7 (1.1) / 0.5 (1.0) |

**Box, true pixels (geometry alone)** — world / reach

| method | none | offsets | sag | mount | all |
|---|---|---|---|---|---|
| fk | 0.1 (0.2) / 0.1 (0.2) | 2.3 (3.2) / 0.4 (1.2) | 0.5 (0.7) / 0.5 (0.8) | 0.6 (1.2) / 0.6 (1.3) | 2.4 (3.1) / 1.2 (1.8) |
| fk_tri | 0.0 (0.0) / 0.0 (0.0) | 2.4 (3.5) / 0.4 (1.0) | 0.5 (0.7) / 0.4 (0.5) | 0.6 (1.2) / 0.6 (1.2) | 2.7 (4.0) / 0.8 (1.8) |
| tag | 0.2 (0.5) / 0.2 (0.5) | 0.6 (1.8) / 2.5 (3.8) | 0.4 (1.0) / 0.6 (1.1) | 0.4 (0.7) / 0.4 (0.7) | 0.7 (2.4) / 2.3 (3.6) |
| mount_cal | 0.1 (0.2) / 0.1 (0.2) | 1.6 (3.2) / 2.7 (3.4) | 0.7 (0.9) / 1.0 (1.1) | 0.1 (0.3) / 0.1 (0.3) | 1.6 (3.0) / 2.6 (3.4) |
| cal | 0.1 (0.2) / 0.1 (0.3) | 0.1 (0.4) / 0.1 (0.4) | 0.2 (0.3) / 0.2 (0.4) | 0.1 (0.3) / 0.1 (0.3) | 0.2 (0.7) / 0.2 (0.6) |
| cal_tag | 0.2 (0.5) / 0.2 (0.5) | 0.3 (0.5) / 0.3 (0.5) | 0.4 (0.9) / 0.4 (0.7) | 0.4 (0.6) / 0.4 (0.6) | 0.5 (0.8) / 0.4 (0.7) |

**Blue square, detector pixels** — world / reach

| method | none | offsets | sag | mount | all |
|---|---|---|---|---|---|
| fk | 0.1 (0.1) / 0.1 (0.1) | 2.8 (4.1) / 1.0 (1.5) | 1.1 (1.2) / 0.3 (0.4) | 0.8 (1.6) / 0.8 (1.6) | 3.0 (5.0) / 1.2 (1.8) |
| tag | 0.3 (0.5) / 0.3 (0.5) | 0.3 (2.1) / 2.3 (3.4) | 0.4 (0.9) / 0.6 (0.9) | 0.5 (0.8) / 0.5 (0.8) | 0.6 (3.3) / 2.1 (3.3) |
| mount_cal | 0.1 (0.1) / 0.1 (0.1) | 0.6 (1.4) / 2.0 (2.9) | 0.4 (0.4) / 0.6 (0.7) | 0.1 (0.1) / 0.1 (0.1) | 0.8 (2.0) / 2.0 (2.9) |
| cal | 0.1 (0.1) / 0.1 (0.1) | 0.1 (0.2) / 0.1 (0.2) | 0.1 (0.2) / 0.1 (0.1) | 0.1 (0.1) / 0.1 (0.1) | 0.2 (0.5) / 0.1 (0.2) |
| cal_tag | 0.3 (0.5) / 0.3 (0.5) | 0.3 (0.4) / 0.3 (0.5) | 0.3 (0.5) / 0.2 (0.5) | 0.3 (0.5) / 0.3 (0.5) | 0.4 (0.6) / 0.3 (0.4) |

**Blue square, true pixels (geometry alone)** — world / reach

| method | none | offsets | sag | mount | all |
|---|---|---|---|---|---|
| fk | 0.0 (0.0) / 0.0 (0.0) | 2.8 (4.1) / 0.9 (1.4) | 0.9 (1.1) / 0.2 (0.3) | 0.8 (1.5) / 0.8 (1.5) | 2.9 (4.8) / 1.0 (1.8) |
| tag | 0.3 (0.4) / 0.3 (0.4) | 0.3 (2.0) / 2.2 (3.5) | 0.3 (0.9) / 0.6 (0.9) | 0.4 (0.7) / 0.4 (0.7) | 0.6 (3.1) / 2.2 (3.3) |
| mount_cal | 0.0 (0.0) / 0.0 (0.0) | 0.6 (1.4) / 2.0 (2.9) | 0.3 (0.4) / 0.7 (0.8) | 0.0 (0.0) / 0.0 (0.0) | 0.8 (2.0) / 2.0 (3.1) |
| cal | 0.0 (0.0) / 0.0 (0.0) | 0.0 (0.0) / 0.1 (0.1) | 0.1 (0.1) / 0.2 (0.2) | 0.0 (0.0) / 0.0 (0.0) | 0.2 (0.5) / 0.1 (0.3) |
| cal_tag | 0.2 (0.4) / 0.3 (0.4) | 0.2 (0.3) / 0.2 (0.4) | 0.3 (0.4) / 0.2 (0.4) | 0.3 (0.4) / 0.3 (0.5) | 0.4 (0.5) / 0.2 (0.4) |

**Tag in view** (trials whose frame showed the tag, of 20 none, 20 offsets, 20 sag, 20 mount, 20 all)

| view | none | offsets | sag | mount | all |
|---|---|---|---|---|---|
| start | 19 | 16 | 19 | 17 | 13 |
| box | 10 | 9 | 11 | 9 | 10 |
| box2 | 4 | 5 | 6 | 5 | 6 |
| target | 12 | 11 | 12 | 11 | 11 |

Camera pose from the tag in the start view (tag 59 px across): 9.4 mm, 1.25 deg median; 10.5 mm, 1.31 deg 90th pct.

### Calibration from a sweep of tag views

Mount error after the fit, mm / deg, median (90th pct). `mount`: hand-eye only. `full`: mount, joint zeros and sag.

| errors on | nominal mount | hand-eye (mount only) | full fit | joint zeros left, deg | sag fitted (true) |
|---|---|---|---|---|---|
| none | 0.0 / 0.00 (0.0 / 0.00) | 0.3 / 0.07 (0.3 / 0.07) | 1.5 / 0.23 (1.5 / 0.23) | 0.21 (0.21) | -0.25 (0.0) |
| offsets | 0.0 / 0.00 (0.0 / 0.00) | 35.3 / 10.95 (74.0 / 20.73) | 2.1 / 1.71 (2.6 / 2.46) | 0.36 (0.55) | 0.08 (0.0) |
| sag | 0.0 / 0.00 (0.0 / 0.00) | 22.1 / 5.09 (22.1 / 5.09) | 1.3 / 0.44 (1.3 / 0.44) | 0.26 (0.26) | 3.65 (4.0) |
| mount | 5.4 / 2.64 (8.8 / 4.54) | 0.3 / 0.11 (0.5 / 0.16) | 1.3 / 0.52 (1.7 / 1.51) | 0.33 (0.44) | -0.28 (0.0) |
| all | 5.4 / 2.64 (8.8 / 4.55) | 43.0 / 12.58 (74.1 / 22.17) | 2.0 / 1.62 (3.8 / 2.56) | 0.67 (1.25) | 4.03 (4.0) |

### Closed-loop probe: servo over the box on the image, then descend

Reach error at the grasp, cm, median (90th pct); failures in brackets.

| reference | none | offsets | sag | mount | all |
|---|---|---|---|---|---|
| nominal | 0.1 (0.3) | 0.5 (0.8) [1] | 0.5 (0.7) | 0.6 (1.4) | 1.0 (1.5) [1] |
| taught | 0.1 (0.3) | 0.4 (1.0) [1] | 0.6 (0.9) | 0.1 (0.4) | 0.5 (1.0) [1] |

### A table grid drawn on the frame

Median over seeds of each frame's median grid point: px off in the frame / cm off on the table; and cm off when
the arm is told to go to the grid's centre. Tag-drawn grids use the tag where the frame shows it, else the last
tag view carried by FK.

| errors | view | tag in view | fk | tag | cal |
|---|---|---|---|---|---|
| offsets | start | 16/20 | 28 px / 3.2 cm | 3 px / 0.4 cm | 0 px / 0.0 cm |
| offsets | box | 9/20 | 32 px / 3.2 cm; reach 2.4 cm | 7 px / 0.7 cm; reach 2.4 cm | 0 px / 0.0 cm; reach 0.0 cm |
| offsets | target | 11/20 | 30 px / 3.2 cm; reach 2.4 cm | 4 px / 0.4 cm; reach 2.4 cm | 0 px / 0.0 cm; reach 0.0 cm |
| all | start | 13/20 | 27 px / 3.5 cm | 5 px / 0.6 cm | 9 px / 1.1 cm |
| all | box | 9/20 | 32 px / 3.2 cm; reach 2.2 cm | 12 px / 1.2 cm; reach 2.2 cm | 2 px / 0.2 cm; reach 0.2 cm |
| all | target | 11/20 | 32 px / 3.1 cm; reach 2.3 cm | 6 px / 0.6 cm; reach 2.3 cm | 2 px / 0.2 cm; reach 0.2 cm |

### End to end

| errors | method | episodes | placed | lifted | median off target when lifted, cm | grasps retried | held check right |
|---|---|---|---|---|---|---|---|
| none | fk | 20 | 18 | 20 | 1.1 | 1 | 21/21 |
| all | fk | 20 | 9 | 12 | 1.2 | 6 | 15/17 |
| all | tag | 20 | 6 | 8 | 1.8 | 11 | 11/11 |
| all | cal | 20 | 18 | 19 | 1.0 | 6 | 24/24 |

### Team 6's final program (PR #8), H2O `localize`, all errors on

| variant | episodes | placed | lifted | median off target when lifted, cm | localize calls | crashed |
|---|---|---|---|---|---|---|
| no-tag | 20 | 5 | 11 | 3.2 | 89 | 0 |
| tag-runtime | 20 | 2 | 8 | 3.7 | 93 | 0 |
| tag-calibrated | 20 | 14 | 14 | 1.7 | 89 | 0 |

