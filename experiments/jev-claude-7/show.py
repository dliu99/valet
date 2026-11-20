"""Print an episode's plans, steps and result: uv run python show.py runs/<name>/seed-NN"""
import json, sys
for l in open(sys.argv[1] + "/events.jsonl"):
    e = json.loads(l)
    k = e["kind"]
    if k in ("plan", "replan"):
        print("==", k, e.get("why", ""), "\n  approach:", e["approach"])
        for s in e["steps"]:
            keep = {"do", "why"} | ({"rel", "G", "pitch", "roll", "gripper"} if s["do"] == "move" else {"gripper", "expect"} if s["do"] == "grip" else {"question", "want"} if s["do"] == "check" else {"rel"})
            print("   ", {kk: v for kk, v in s.items() if kk in keep})
    elif k == "step":
        print("  ->", {kk: v for kk, v in e.items() if kk not in ("spec", "t", "kind", "plan")})
    elif k == "estimate":
        print(k, {kk: v for kk, v in e.items() if kk not in ("t", "kind")})
    elif k == "result":
        print(k, {kk: v for kk, v in e.items() if kk not in ("t", "kind", "jev", "claude")})
    elif k == "error":
        print(e["trace"])
