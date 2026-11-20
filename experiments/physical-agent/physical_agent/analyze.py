"""Read runs/<name>/ folders and report per condition: placed and lifted, turns, seconds, tokens, dollars, and what
happened after failed grasps. Uses the logged truth, which the model never saw.

    uv run python -m physical_agent.analyze opus-zero-0-4 opus-demo-0-4 ...
    uv run python -m physical_agent.analyze --sheet opus-zero-0-19/seed-03 2 5 8   # frames for the report
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .tool import to_base

RUNS = Path(__file__).resolve().parent.parent / "runs"
RESULTS = RUNS.parent / "results"
CLOSED = 20.0
OPEN = 30.0


def load(run: str) -> tuple[dict, list[dict]]:
    """A run's settings and summary, rebuilt from its seeds' results (a run may have been extended or merged)."""
    from .run import summarize

    rows = [json.loads(f.read_text()) for f in sorted((RUNS / run).glob("seed-*/result.json"))]
    if not rows:
        raise SystemExit(f"{run}: no results")
    r0 = rows[0]
    summary = {"model": r0["model"], "condition": r0["condition"], "views": r0.get("views", "both"),
               "frame": r0.get("frame", "base"), "demo_seed": r0.get("demo_seed"), **summarize(rows)}
    episodes = []
    for row in rows:
        path = RUNS / run / f"seed-{row['seed']:02d}" / "events.jsonl"
        episodes.append({**row, "events": [json.loads(line) for line in path.read_text().splitlines()]})
    return summary, episodes


def grasps(events: list[dict], seed: int) -> list[dict]:
    """Each closing of the gripper starts an attempt that lasts until it is opened again; it held if the box was
    lifted or held at the end of any call in between. `miss_m` is how far the closing point was, in xy, from the
    box's centre at the start of that call."""
    from valet_sim.task import episode

    b = episode(seed).box
    box_xy = (b.x, b.y)
    attempts: list[dict] = []
    grip = 60.0
    current = None
    lifted_before = 0.0
    for e in events:
        acts = e.get("actions") or []
        ran = acts[: e.get("ran", 0)]
        truth = e.get("truth") or {}
        for a in ran:
            if "forward" in a:
                a = to_base(a, np.array(e["start"]))
            g = float(a["gripper"])
            if current is None and g <= CLOSED and grip > CLOSED:
                miss = ((a["x"] - box_xy[0]) ** 2 + (a["y"] - box_xy[1]) ** 2) ** 0.5
                current = {"turn": e["turn"], "held": False, "miss_m": round(miss, 4), "pitch": float(a["pitch_deg"]),
                           "roll": float(a["roll_deg"]), "z": float(a["z"])}
                attempts.append(current)
            elif current is not None and g >= OPEN:
                current["released_turn"] = e["turn"]
                current = None
            grip = g
        if truth:
            lifted = truth.get("lifted_m", 0.0)
            for att in attempts:
                if att.get("released_turn", e["turn"]) >= e["turn"] >= att["turn"] and (truth.get("held") or lifted > lifted_before + 0.01):
                    att["held"] = True
            lifted_before = max(lifted_before, lifted)
            box_xy = (truth["box"]["x"], truth["box"]["y"])
    return attempts


def recovery(ep: dict) -> dict:
    atts = grasps(ep["events"], ep["seed"])
    failed = [a for a in atts if not a["held"]]
    first_fail = failed[0]["turn"] if failed else None
    retried = bool(first_fail) and any(a["turn"] > first_fail for a in atts)
    recovered = bool(first_fail) and any(a["held"] and a["turn"] > first_fail for a in atts)
    ik_fail = sum("failed" in (e.get("result") or "") for e in ep["events"])
    tipped = any((e.get("truth") or {}).get("box", {}).get("tilt_deg", 0) > 15 for e in ep["events"])
    return {"attempts": len(atts), "failed": len(failed), "retried": retried, "recovered": recovered,
            "first_miss_m": atts[0]["miss_m"] if atts else None, "misses_m": [a["miss_m"] for a in atts],
            "demo_pose": sum(abs(a["pitch"] - 70) < 1 and abs(a["z"] - 0.035) < 0.004 and abs(a["roll"] + 90) < 1 for a in atts),
            "rolled": sum(abs(a["roll"] + 90) >= 5 for a in atts),
            "ik_failures": ik_fail, "tipped": tipped, "false_done": ep["ended"] == "done" and not ep["placed"]}


def table(runs: list[str]) -> str:
    head = ("| run | model | views | frame | condition | placed | lifted | median turns | median s (wall / sim) | "
            "median tokens | $ / episode | grasps (held) | episodes with a failed grasp: retried / recovered / placed | "
            "ended without a call, not placed | turn budget hit | median miss at first grasp (cm) | "
            "lifted, not placed: > 3 cm off / tilted / in jaws | grasps at the demo's pose | grasps with roll turned |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for run in runs:
        s, eps = load(run)
        recs = [recovery(e) for e in eps]
        with_fail = [(e, r) for e, r in zip(eps, recs) if r["failed"]]
        n = s["n"]
        lines.append(
            f"| {run} | {s['model'].removeprefix('claude-')} | {s.get('views', 'both')} | {s.get('frame', 'base')} | "
            f"{s['condition']}{'' if s.get('demo_seed') in (None, 104) else ' (seed ' + str(s['demo_seed']) + ')'} | "
            f"{s['placed']}/{n} | {s['lifted']}/{n} | {s['median_turns']} | "
            f"{s['median_wall_s']:.0f} / {s['median_sim_s']:.0f} | {s['median_tokens']:.0f} | {s['mean_dollars']:.2f} | "
            f"{sum(r['attempts'] for r in recs)} ({sum(r['attempts'] - r['failed'] for r in recs)}) | "
            f"{len(with_fail)}: {sum(r['retried'] for _, r in with_fail)} / {sum(r['recovered'] for _, r in with_fail)} / "
            f"{sum(e['placed'] for e, _ in with_fail)} | {sum(r['false_done'] for r in recs)} | "
            f"{s['ended'].get('turn budget', 0)} | {_median_cm([r['first_miss_m'] for r in recs])} | {_why_not(eps)} | "
            f"{sum(r['demo_pose'] for r in recs)} | {sum(r['rolled'] for r in recs)} |")
    return "\n".join(lines)


def _why_not(eps: list[dict]) -> str:
    near = [e for e in eps if e["lifted"] and not e["placed"]]
    off = sum(e["off_target_m"] > 0.03 for e in near)
    tilted = sum(e["tilt_deg"] > 15 for e in near)
    held = sum(not e["released"] for e in near)
    return f"{off} / {tilted} / {held}"


def _median_cm(xs: list) -> str:
    xs = [x for x in xs if x is not None]
    return f"{100 * statistics.median(xs):.1f}" if xs else "-"


def episodes(run: str) -> str:
    _, eps = load(run)
    out = []
    for e in eps:
        r = recovery(e)
        verdict = "placed" if e["placed"] else "lifted" if e["lifted"] else "failed"
        out.append(f"{run} seed {e['seed']:2}: {verdict:6} off {e['off_target_m']:.3f} m, tilt {e['tilt_deg']}, "
                   f"released {e['released']}, turns {e['turns']} ({e['ended']}), grasps {r['attempts']} "
                   f"(failed {r['failed']}, recovered {r['recovered']}, misses cm {[round(100 * m, 1) for m in r['misses_m']]}), "
                   f"ik failures {r['ik_failures']}, "
                   f"tipped {r['tipped']}, ${e['dollars']:.2f}")
    return "\n".join(out)


def show(episode_dir: str) -> str:
    """One episode turn by turn: what the model said, what it called, and the truth after (grading only)."""
    out = []
    for line in (RUNS / episode_dir / "events.jsonl").read_text().splitlines():
        e = json.loads(line)
        said = (e.get("reasoning") or e.get("thinking_summary") or "").replace("\n", " ")
        out.append(f"T{e['turn']}: {said[:400]}")
        for a in (e.get("actions") or [])[: max(e.get("ran", 0), 1)]:
            out.append("    " + ", ".join(f"{k} {v}" for k, v in a.items() if k != "control_steps"))
        if e.get("result"):
            t = e["truth"]
            out.append(f"    -> {e['result']} | truth: held {t['held']}, box ({t['box']['x']:.3f}, {t['box']['y']:.3f}) "
                       f"tilt {t['box']['tilt_deg']}")
    return "\n".join(out)


def sheet(episode_dir: str, turns: list[int], out: Path, width: int = 480) -> Path:
    """Scene and wrist frames side by side, one row per turn, labelled."""
    d = RUNS / episode_dir
    rows = []
    for t in turns:
        name = "final" if t < 0 else f"t{t:02d}"
        pair = [Image.open(d / f"{name}-{cam}.jpg").convert("RGB") for cam in ("scene", "wrist")]
        h = int(pair[0].height * width / pair[0].width)
        row = Image.new("RGB", (2 * width, h), "white")
        for k, im in enumerate(pair):
            row.paste(im.resize((width, h)), (k * width, 0))
        ImageDraw.Draw(row).text((8, 6), "final" if t < 0 else "start" if t == 0 else f"after turn {t}", fill=(255, 255, 0))
        rows.append(row)
    sheet_im = Image.new("RGB", (2 * width, sum(r.height for r in rows)), "white")
    y = 0
    for r in rows:
        sheet_im.paste(r, (0, y))
        y += r.height
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet_im.save(out, quality=82)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m physical_agent.analyze", description=__doc__)
    p.add_argument("runs", nargs="*")
    p.add_argument("--sheet", help="runs-relative episode folder, then turns (-1 = final)")
    p.add_argument("--show", help="runs-relative episode folder: print it turn by turn")
    p.add_argument("--out", type=Path)
    p.add_argument("--export", action="store_true", help="also write results/<run>.json: settings, summary, per-seed rows")
    a = p.parse_args(argv)
    if a.show:
        print(show(a.show))
        return 0
    if a.sheet:
        turns = [int(t) for t in a.runs]
        print(sheet(a.sheet, turns, a.out or RUNS / "sheet.jpg"))
        return 0
    if a.export:
        RESULTS.mkdir(exist_ok=True)
        for run in a.runs:
            summary, eps = load(run)
            rows = [{k: v for k, v in e.items() if k != "events"} | {"grasps": recovery(e)} for e in eps]
            (RESULTS / f"{run}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(table(a.runs))
    print()
    for run in a.runs:
        print(episodes(run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
