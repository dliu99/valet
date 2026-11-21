# Box fit: one view or two (seeds 20-39, SAM 3 at 504 px)

`python -m sam3p.variants`: the start-pose frame and a look from 22 cm above the box (aimed 1.5 cm off at random),
per prompt wording, with and without merging touching box instances. Errors in cm on the table; 99 marks no box found.

| box prompt | start-view mask IoU < 0.8 | start view alone: centre med / p90 / max | above view, size from start view: med / p90 / max | two views fitted together: med / p90 / max | two views: height error max | start view alone: height error p90 / max |
|---|---|---|---|---|---|---|
| white box | 3/20 | 0.17 / 2.53 / 5.34 | 0.11 / 0.94 / miss (1) | 0.03 / 0.06 / 0.09 | 0.22 | 2.26 / 5.55 |
| white box + merge | 1/20 | 0.10 / 0.42 / 2.42 | 0.07 / 0.46 / miss (1) | 0.03 / 0.06 / 0.08 | 0.23 | 1.56 / 3.64 |
| white cube | 2/20 | 0.14 / 1.63 / 4.94 | 0.08 / 0.82 / 1.02 | 0.03 / 0.07 / 0.11 | 0.51 | 2.89 / 5.41 |
| white cube + merge | 0/20 | 0.12 / 0.45 / 1.42 | 0.07 / 0.28 / 0.83 | 0.03 / 0.06 / 0.09 | 0.16 | 1.70 / 4.76 |

The bad start views are boxes far to the left whose shaded side face reads dark grey: SAM 3's "white box" mask
covers only the lit faces, and a single-view fit then makes the box short (down to the 2.5 cm bound) and shifts it
2-5 cm. Fitting one box to both views at once fixes size and height (max errors 0.09 cm and 0.22 cm).
