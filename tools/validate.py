"""Where each camera model puts the cap seen in earlier views, against where the grasp found the bottle."""
import numpy as np
from valet.armd.kinematics import Kinematics
from valet.armd.locate import REAL_CAMERA, Camera, Locator

views = [  # joints (pan, lift, elbow, flex, roll), cap-centre pixel read off the frame
    ((13.8, 1.23, -34.15, 85.23, -97.8), (0.42, 0.585)),
    ((7.82, -1.58, -12.53, 67.91, -97.8), (0.50, 0.61)),
    ((4.31, 13.98, -26.42, 66.07, -97.8), (0.56, 0.795)),
    ((4.31, 13.19, -8.4, 49.71, -97.8), (0.57, 0.83)),
    ((7.74, -50.11, -1.27, 93.49, -97.8), (0.515, 0.33)),
]
kin = Kinematics()
models = {"sim": Locator(kin, 0.0), "fit": Locator(kin, 0.0, REAL_CAMERA)}
for H in (0.07, 0.08, 0.09, 0.10):
    for name, loc in models.items():
        out = []
        for q, (u, v) in views:
            r = loc.locate(np.array([*q, 0.0]), u, v, 16 / 9, H)
            out.append((round(r["pan_deg"], 1), round(r["reach_m"] * 100, 1)) if r["hit"] else None)
        pts = [p for p in out if p]
        spread = np.std(np.array(pts), axis=0).round(1) if pts else None
        print(f"H {H*100:.0f} {name}: {out} spread {spread}")
