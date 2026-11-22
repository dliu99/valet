Box centre error on the table, cm (median / p90 / max), 30 poses; a miss is a view where the method gave no position.

| method | views | positions | error cm |
|---|---|---|---|
| main `samAxis`, start view | 30 | 23 | 1.34 / 10.59 / 13.79 |
| main `samAxis`, `lookAt` view | 30 | 25 | 1.12 / 6.75 / 7.87 |
| main `find`: both views, agreeing within 2 cm | 30 | 12 | 0.94 / 1.18 / 1.33 |
| mask fit, start view | 30 | 30 | 0.28 / 1.63 / 3.34 |
| mask fit, `lookAt` view (size from the start view) | 30 | 30 | 0.23 / 1.81 / 2.64 |
| mask fit, both views together | 30 | 30 | 0.03 / 0.09 / 0.32 |

main's reading per view: foot 27, centre 21, not found 11, out of range 1

**this policy (SAM 3 mask fit, scripted)**: placed 7/10, lifted 9/10, set down upright 10/10; off the spot when placed 0.80 / 1.85 / 1.85 cm; per episode (median): wall 3 s, SAM 3 frames 3, LLM calls 0

| ep | outcome | lifted | upright | off cm |
|---|---|---|---|---|
| 0 | done | True | True | 0.8 |
| 1 | tipped | True | True | 0.56 |
| 2 | done | True | True | 1.85 |
| 3 | tipped | False | True | 9.16 |
| 4 | done | True | True | 0.61 |
| 5 | done | True | True | 1.07 |
| 6 | done | True | True | 0.91 |
| 7 | tipped | True | True | 7.5 |
| 8 | done | True | True | 0.71 |
| 9 | off | True | True | 2.74 |

**main's brain as it ships (`valet do --no-claude`)**: placed 0/10, lifted 1/10, set down upright 9/10; off the spot when placed - cm; per episode (median): wall 36 s, Jev calls 14, skills run 10, LLM calls 0

| ep | outcome | lifted | upright | off cm |
|---|---|---|---|---|
| 0 | run #2: done: Jev chose done (p 0.53) | False | True | 9.75 |
| 1 | run #49: unsure: Jev was unsure 2 steps in a row (last: approach at p 0.36) | False | False | 9.94 |
| 2 | run #89: unsure: Jev was unsure 2 steps in a row (last: descend at p 0.29) | False | True | 9.34 |
| 3 | run #163: unsure: Jev was unsure 2 steps in a row (last: lift at p 0.24) | False | True | 9.04 |
| 4 | run #235: unsure: Jev was unsure 2 steps in a row (last: line_up at p 0.25) | False | True | 9.68 |
| 5 | run #274: unsure: Jev was unsure 2 steps in a row (last: approach at p 0.22) | False | True | 9.06 |
| 6 | run #319: unsure: Jev was unsure 2 steps in a row (last: close at p 0.28) | False | True | 9.67 |
| 7 | run #355: budget: 30 steps taken | True | True | 9.07 |
| 8 | run #486: done: Jev chose done (p 0.43) | False | True | 9.75 |
| 9 | run #527: unsure: Jev was unsure 2 steps in a row (last: lift at p 0.20) | False | True | 9.28 |

**main's brain, armd's gripper stop raised from 0.9 to 1.5 A**: placed 0/10, lifted 5/10, set down upright 9/10; off the spot when placed - cm; per episode (median): wall 49 s, Jev calls 16, skills run 12, LLM calls 0

| ep | outcome | lifted | upright | off cm |
|---|---|---|---|---|
| 0 | run #570: unsure: Jev was unsure 2 steps in a row (last: approach at p 0.35) | False | True | 8.69 |
| 1 | run #606: done: Jev chose done (p 0.49) | True | True | 9.45 |
| 2 | run #674: done: Jev chose done (p 0.47) | True | True | 7.68 |
| 3 | run #716: done: Jev chose done (p 0.44) | True | True | 6.88 |
| 4 | run #779: unsure: Jev was unsure 2 steps in a row (last: nudge_out at p 0.22) | False | True | 7.4 |
| 5 | run #829: done: Jev chose done (p 0.37) | True | False | 1.93 |
| 6 | run #882: unsure: Jev was unsure 2 steps in a row (last: descend at p 0.31) | False | True | 8.99 |
| 7 | run #918: unsure: Jev was unsure 2 steps in a row (last: approach at p 0.24) | False | True | 8.83 |
| 8 | run #971: unsure: Jev was unsure 2 steps in a row (last: line_up at p 0.29) | True | True | 9.91 |
| 9 | run #1051: done: Jev chose done (p 0.49) | False | True | 8.53 |

