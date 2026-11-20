"""Gather the numbers REPORT.md quotes into results/summary.json: runs, candidates, skills, evaluations and spend.

    uv run python -m aspire.collect
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import ledger
from .evolve import ROOT, SKILLS, STATE

RUNS = ("main", "variant-skills", "variant-skills-b", "variant-noskills", "variant-noskills-b")
EVALS = {
    "benchmark_final_heldout": "heldout/benchmark_final",
    "benchmark_final_heldout_after_merge": "heldout/benchmark_final_after_merge",
    "benchmark_final_heldout_jev": "heldout/benchmark_final_jev",
    "benchmark_final_heldout_jev_found_fix": "heldout/benchmark_final_jev_v2",
    "benchmark_final_heldout_h2o": "heldout/benchmark_final_h2o",
    "zeroshot_variant_debug": "heldout/zeroshot_variant_debug",
    "zeroshot_variant_heldout": "heldout/zeroshot_variant_heldout",
    "variant_skills_final_heldout": "heldout/variant_skills_final",
    "variant_noskills_final_heldout": "heldout/variant_noskills_final",
    **{f"diagnostic_{c}_heldout": f"heldout/diag_{c}" for c in ("r0a0", "r0a2", "r1a1", "r1v0", "r1v1")},
}


def main() -> int:
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    summary: dict = {"runs": {}, "evals": {}, "skills": [], "spend": {}}
    for name in RUNS:
        path = STATE / name / "archive.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        sessions = []
        for s in sorted((STATE / name / "sessions").glob("*/session.json")):
            j = json.loads(s.read_text())
            sessions.append({k: j[k] for k in ("name", "role", "usd", "turns", "tool_calls", "submitted")})
        summary["runs"][name] = {
            "candidates": [{k: c[k] for k in ("id", "round", "parent", "origin", "placed", "lifted", "fitness", "usd", "skill")}
                           for c in d["archive"]],
            "sessions": sessions,
            "events": [e for e in d["events"] if e["kind"] not in ("candidate",)],
        }
        shutil.copy(path, out / f"archive-{name}.json")
    for key, rel in EVALS.items():
        p = STATE / rel / "eval.json"
        if p.exists():
            e = json.loads(p.read_text())
            summary["evals"][key] = {k: e.get(k) for k in ("program", "task", "placed", "lifted", "n", "median_sim_s",
                                                         "localize_calls", "localize_paid", "usd")}
            summary["evals"][key]["unplaced"] = [r["seed"] for r in e["rows"] if not r["placed"]]
    for f in sorted(SKILLS.glob("*.md")):
        front = f.read_text().split("\n---\n", 1)[0].removeprefix("---\n")
        summary["skills"].append({"file": f.name, **{k: json.loads(v) for k, v in
                                                     (line.split(": ", 1) for line in front.splitlines())}})
    summary["spend"] = {"total_usd": round(ledger.spent(), 3), "by_role": ledger.summary()}
    tokens = {k: sum(r[k] for r in summary["spend"]["by_role"].values()) for k in ("input", "output", "cache_read", "cache_write")}
    summary["spend"]["tokens"] = tokens
    expert = [json.loads(l) for l in (out / "expert_reference.jsonl").read_text().splitlines()]
    summary["expert"] = [{k: e[k] for k in ("task", "seeds", "placed", "lifted", "n")} for e in expert]
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary["evals"][k] for k in summary["evals"]}, indent=1))
    print(json.dumps(summary["spend"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
