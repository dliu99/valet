"""Table-plane points for pixels of the upright wrist image: python pts.py height_cm u,v u,v ..."""
import json, sys, urllib.request
h = float(sys.argv[1]) / 100
for uv in sys.argv[2:]:
    u, v = map(float, uv.split(","))
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8750/locate", json.dumps({"u": u, "v": v, "aspect": 16/9, "height_m": h}).encode(), {"Content-Type": "application/json"})))
    print(uv, {k: r.get(k) for k in ("pan_deg", "reach_m", "xyz_m")})
