"""aspire evolve | eval | spend"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import ledger, tasks


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="aspire", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("evolve", help="run the debugging loop on the debug seeds")
    e.add_argument("--name", required=True)
    e.add_argument("--task", default="benchmark", choices=sorted(tasks.EPISODES))
    e.add_argument("--k", type=int, default=3)
    e.add_argument("--rounds", type=int, default=5)
    e.add_argument("--no-skills", action="store_true", help="give actors no skill library")
    e.add_argument("--freeze-skills", action="store_true", help="read the library but admit nothing new")
    e.add_argument("--actor-tool-calls", type=int, default=10)

    v = sub.add_parser("eval", help="run one program once on a seed set")
    v.add_argument("program", type=Path)
    v.add_argument("--task", default="benchmark", choices=sorted(tasks.EPISODES))
    v.add_argument("--seeds", default="heldout", help="heldout, debug, or a comma list")
    v.add_argument("--out", type=Path, required=True)
    v.add_argument("--backend", default="claude", choices=["claude", "jev", "h2o"], help="what answers localize")

    sub.add_parser("spend", help="API spend so far, by role")

    h = sub.add_parser("history", help="candidates and skills of an evolve run, round by round")
    h.add_argument("--name", required=True)

    a = p.parse_args(argv)
    ledger.load_env()
    if a.cmd == "evolve":
        from .evolve import Evolution

        evo = Evolution(a.name, a.task, a.k, a.rounds, use_skills=not a.no_skills,
                        write_skills=not (a.no_skills or a.freeze_skills), actor_tool_calls=a.actor_tool_calls)
        best = evo.run()
        print(json.dumps({"best": best.id, "placed": best.placed, "lifted": best.lifted, "fitness": best.fitness,
                          "spent_usd": round(ledger.spent(), 2)}))
        return 0
    if a.cmd == "eval":
        from . import engine

        seeds = (tasks.HELDOUT_SEEDS[a.task] if a.seeds == "heldout" else tasks.DEBUG_SEEDS[a.task] if a.seeds == "debug"
                 else [int(s) for s in a.seeds.split(",")])
        before = ledger.spent()
        t0 = time.time()
        ev = engine.evaluate(a.program.read_text(), seeds, a.task, a.out, backend=a.backend)
        usd = ledger.spent() - before
        rows = [{"seed": r["seed"], **r["score"], "error": r["error"], "calls": len(r["calls"]),
                 "localize_calls": r["localize_calls"], "sim_s": r["sim_s"]} for r in ev["results"]]
        report = {"program": str(a.program), "task": a.task, "backend": a.backend, "seeds": list(seeds), "placed": ev["placed"],
                  "lifted": ev["lifted"], "n": ev["n"], "median_sim_s": ev["median_sim_s"],
                  "localize_calls": ev["localize_calls"], "localize_paid": ev["localize_paid"], "usd": round(usd, 4), "wall_s": round(time.time() - t0),
                  "rows": rows}
        (a.out / "eval.json").write_text(json.dumps(report, indent=1))
        for r in rows:
            print(f"seed {r['seed']:3}: {'placed' if r['placed'] else 'lifted' if r['lifted'] else 'failed':7} "
                  f"off {r['off_target_m']:.3f} tilt {r['tilt_deg']:5.1f} released {r['released']} {r['error'] or ''}")
        print(f"placed {ev['placed']}/{ev['n']}, lifted {ev['lifted']}/{ev['n']}, median {ev['median_sim_s']} s sim, "
              f"{ev['localize_calls']} localize calls ({ev['localize_paid']} not cached), ${usd:.3f} ledger delta")
        return 0
    if a.cmd == "spend":
        s = ledger.summary()
        for role, r in sorted(s.items()):
            print(f"{role:16} {r['calls']:5} calls  ${r['usd']:8.3f}  in {r['input']:>9} out {r['output']:>8} "
                  f"cache_read {r['cache_read']:>9} cache_write {r['cache_write']:>8}")
        print(f"total ${ledger.spent():.3f} of ${ledger.budget_usd():.0f}")
        return 0
    if a.cmd == "history":
        from .evolve import STATE

        d = json.loads((STATE / a.name / "archive.json").read_text())
        for c in sorted(d["archive"], key=lambda c: (c["round"], c["id"])):
            fails = [s for s, v in c["per_seed"].items() if not v["placed"]]
            print(f"round {c['round']} {c['id']:12} {c['origin']:8} parent {str(c['parent']):12} placed {c['placed']:2} "
                  f"lifted {c['lifted']:2} fitness {c['fitness']:6.3f} ${c['usd']:.2f} unplaced {','.join(fails)}")
        for e in d["events"]:
            if e["kind"].startswith("skill") or e["kind"] in ("round_end", "budget"):
                print(json.dumps(e))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
