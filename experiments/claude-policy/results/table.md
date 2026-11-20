| run | placed | lifted | median s | Claude calls | Jev calls | input tok | output tok | $ Claude | $ Jev | per seed |
|---|---|---|---|---|---|---|---|---|---|---|
| v0 baseline, wrist+side | 2/5 | 4/5 | 196 | 30.6 | 0.0 | 649k | 12.1k | 0.56 | 0.00000 | 0:P 1:L 2:L 3:P 4:- |
| v1 v0 + IK hint | 3/5 | 4/5 | 106 | 24.6 | 0.0 | 464k | 9.1k | 0.42 | 0.00000 | 0:P 1:L 2:P 3:P 4:- |
| v2 v1 + overlay | 4/5 | 5/5 | 131 | 23.8 | 0.0 | 393k | 8.8k | 0.40 | 0.00000 | 0:P 1:L 2:P 3:P 4:P |
| w0 v2 with wrist only | 3/5 | 5/5 | 75 | 24.0 | 0.0 | 336k | 7.8k | 0.33 | 0.00000 | 0:P 1:P 2:L 3:P 4:L |
| w1 w0 + 4 mm near clip | 1/5 | 5/5 | 191 | 28.6 | 0.0 | 485k | 11.5k | 0.47 | 0.00000 | 0:L 1:L 2:L 3:P 4:L |
| w2 w0 + Claude-written Jev checks | 2/5 | 5/5 | 79 | 23.4 | 0.0 | 344k | 8.0k | 0.34 | 0.00000 | 0:L 1:L 2:P 3:L 4:P |
| w3 w0 + centred in goal | 3/5 | 4/5 | 84 | 23.2 | 0.0 | 325k | 9.3k | 0.37 | 0.00000 | 0:P 1:P 2:P 3:L 4:- |
| w4 w0 + Jev verify at end | 4/5 | 5/5 | 166 | 34.2 | 1.2 | 626k | 12.8k | 0.55 | 0.00005 | 0:P 1:P 2:L 3:P 4:P |
| w5 w0 at effort low | 2/5 | 4/5 | 83 | 24.0 | 0.0 | 304k | 6.2k | 0.29 | 0.00000 | 0:L 1:P 2:L 3:P 4:- |
| final w0 on seeds 0-19 | 11/20 | 20/20 | 79 | 21.4 | 0.0 | 269k | 6.9k | 0.29 | 0.00000 | 0:L 1:P 2:P 3:L 4:P 5:L 6:P 7:P 8:L 9:L 10:P 11:P 12:P 13:P 14:L 15:L 16:L 17:P 18:P 19:L |
