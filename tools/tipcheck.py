"""Where does the camera model say the grasp point is, vs the pixel where the fixed jaw tip shows?"""
import json, sys, urllib.request
import numpy as np
def post(path, body):
    return json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8750" + path, json.dumps(body).encode(), {"Content-Type": "application/json"})))
st = json.load(urllib.request.urlopen("http://127.0.0.1:8750/state"))
tip = np.array([st["tip"]["x"], st["tip"]["y"], st["tip"]["z"]])
print("tip", tip, "pan", st["pan_deg"], "reach", st["reach_m"], "z", st["height_m"])
for u, v in [(0.43, 0.62), (0.43, 0.68), (0.44, 0.72)]:
    r = post("/locate", {"u": u, "v": v, "aspect": 16 / 9, "height_m": float(tip[2])})
    o, d = np.array(r["ray"]["origin_m"]), np.array(r["ray"]["dir"])
    s = (tip - o) @ d
    miss = tip - (o + s * d)
    print((u, v), "cam", o.round(3), "dist along ray", round(s, 3), "miss_cm", (miss * 100).round(1), "|miss|", round(float(np.linalg.norm(miss)) * 100, 1))
