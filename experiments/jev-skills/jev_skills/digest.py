"""Print one line per decision of an episode log: the choice, whether it was right, and the done judgement."""
import json, sys
from pathlib import Path

for path in sys.argv[1:]:
    print("==", path)
    for line in Path(path).read_text().splitlines():
        e = json.loads(line)
        k = e["kind"]
        if k == "chose":
            ok = "?" if e["right"] is None else ("ok" if e["choice"] in e["right"] else "WRONG")
            top = sorted(e["probs"].items(), key=lambda kv: -kv[1])[:3]
            print(f"  {e['step']:2} chose {e['choice']:15} {ok:5} right={e['right']} top={top}")
        elif k == "ran":
            print(f"     ran {e['skill']}: ok={e['ok']} {e['report'][:150]}")
        elif k == "judged":
            print(f"     judged {e['skill']} p={e['p']} {e['verdict']} code_ok={e['code_ok']} truth={e['truth']}")
        elif k in ("final_check", "replan", "goal"):
            print(f"     {k}: {json.dumps({x: y for x, y in e.items() if x not in ('t', 'kind')})[:300]}")
