"""Print which pitches reach each (reach, height) with the arm's real joint limits."""
import json, sys
from pathlib import Path
import numpy as np
from valet.armd.kinematics import Kinematics, ARM
cal = json.loads(next(Path.home().glob(".cache/huggingface/lerobot/calibration/robots/*/valet_follower.json")).read_text())
lim = {m: (cal[m]["range_max"] - cal[m]["range_min"]) / 2 * 360 / 4095 - 3.0 for m in ARM}
print({m: round(v) for m, v in lim.items()})
k = Kinematics()
seed = np.array([0, -44, 38, 38, -97.8])
for reach in [x / 100 for x in range(12, 34, 2)]:
    print(f"reach {reach*100:.0f}:", "  ".join(f"z{z*100:.0f}:{(lambda p: f'{min(p):.0f}..{max(p):.0f}' if p else '-')(k.pitches(0, reach, z, -97.8, seed, lim))}" for z in (0.02, 0.06, 0.10, 0.14, 0.18, 0.22, 0.26)))
