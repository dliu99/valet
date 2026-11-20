"""Per run: how often Jev chose an acceptable skill, and how often its done verdicts matched the truth."""
import json, sys
from collections import Counter
from pathlib import Path


def stats(run_dir: Path) -> dict:
    chose = right = 0
    judged = Counter()
    per_skill = {}
    final = Counter()
    for f in sorted(run_dir.glob("seed*/events.jsonl")):
        for line in f.read_text().splitlines():
            e = json.loads(line)
            if e["kind"] == "chose" and e["right"] is not None:
                chose += 1
                right += e["choice"] in e["right"]
            elif e["kind"] == "judged":
                judged[(e["verdict"], e["truth"])] += 1
                c = per_skill.setdefault(e["skill"].replace("grasp_middle", "grasp_near_top"), Counter())
                c["n"] += 1
                c["right"] += (e["verdict"] == "yes" and e["truth"]) or (e["verdict"] == "no" and not e["truth"])
                c["unsure"] += e["verdict"] == "unsure"
                c["code_right"] += e["code_ok"] == e["truth"]
            elif e["kind"] == "final_check":
                final[("yes" if e["p"] >= 0.7 else "no" if e["p"] <= 0.3 else "unsure", e["truth"])] += 1
    n = sum(judged.values())
    agree = judged[("yes", True)] + judged[("no", False)]
    return {"run": run_dir.name, "choices": chose, "right_choice": round(right / max(chose, 1), 3),
            "judgements": n, "judge_right": round(agree / max(n, 1), 3),
            "judge_unsure": round((judged[("unsure", True)] + judged[("unsure", False)]) / max(n, 1), 3),
            "judge_wrong": round((judged[("yes", False)] + judged[("no", True)]) / max(n, 1), 3),
            "final_checks": {f"{v}/{t}": c for (v, t), c in final.items()},
            "per_skill": {k: dict(v) for k, v in sorted(per_skill.items())}}


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(json.dumps(stats(Path(d))))
