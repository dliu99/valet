"""Bottle size and axis from one box: python size.py left top right bottom"""
import json, sys, urllib.request
import numpy as np
L, T, R, B = map(float, sys.argv[1:5])
def loc(u, v, h=0.0):
    req = urllib.request.Request("http://127.0.0.1:8750/locate", json.dumps({"u": u, "v": v, "aspect": 16/9, "height_m": h}).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))
bl, br, bc = loc(L, B), loc(R, B), loc((L+R)/2, B)
pl, pr, pc = (np.array(x["xyz_m"]) for x in (bl, br, bc))
diam = float(np.linalg.norm(pl - pr))
print("base near edge", {k: bc[k] for k in ("pan_deg", "reach_m")}, "diameter_cm", round(diam*100, 1))
r = diam / 2
cam = np.array(bc["ray"]["origin_m"]); d0 = np.array(bc["ray"]["dir"])
horiz = (pc - cam)[:2]; horiz /= np.linalg.norm(horiz)
axis = pc[:2] + horiz * r
top = loc((L+R)/2, T, 0.0)
o, d = np.array(top["ray"]["origin_m"]), np.array(top["ray"]["dir"])
# far top edge: where the top ray is r beyond the axis (horizontally, along the view direction)
s = ((axis + horiz * r - o[:2]) @ horiz) / (d[:2] @ horiz)
p = o + d * s
print("axis xy", axis.round(4), "height_cm", round(float(p[2])*100, 1))
