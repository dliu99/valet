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

from PIL import Image, ImageDraw

RUNS = Path(__file__).resolve().parent.parent / "runs"
CLOSED = 20.0
OPEN = 30.0


def load(run: str) -> tuple[dict, list[dict]]:
    d = json.loads((RUNS / run / "summary.json").read_text())
    episodes = []
    for row in d["rows"]:
        path = RUNS / run / f"seed-{row['seed']:02d}" / "events.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        episodes.append({**row, "events": events})
    return d["summary"], episodes


def grasps(events: list[dict]) -> list[dict]:
    """Each closing of the gripper starts an attempt that lasts until it is opened again; it held if the box was
    lifted or held at the end of any call in between."""
    attempts: list[dict] = []
    grip = 60.0
    current = None
    lifted_before = 0.0
    for e in events:
        acts = e.get("actions") or []
        ran = acts[: e.get("ran", 0)]
        truth = e.get("truth") or {}
        for a in ran:
            g = float(a["gripper"])
            if current is None and g <= CLOSED and grip > CLOSED:
                current = {"turn": e["turn"], "held": False}
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
    return attempts


def recovery(ep: dict) -> dict:
    atts = grasps(ep["events"])
    failed = [a for a in atts if not a["held"]]
    first_fail = failed[0]["turn"] if failed else None
    retried = bool(first_fail) and any(a["turn"] > first_fail for a in atts)
    recovered = bool(first_fail) and any(a["held"] and a["turn"] > first_fail for a in atts)
    ik_fail = sum("failed" in (e.get("result") or "") for e in ep["events"])
    tipped = any((e.get("truth") or {}).get("box", {}).get("tilt_deg", 0) > 15 for e in ep["events"])
    return {"attempts": len(atts), "failed": len(failed), "retried": retried, "recovered": recovered,
            "ik_failures": ik_fail, "tipped": tipped, "false_done": ep["ended"] == "done" and not ep["placed"]}


def table(runs: list[str]) -> str:
    head = ("| run | model | condition | placed | lifted | median turns | median s (wall / sim) | median tokens | "
            "$ / episode | grasp attempts | failed grasps | episodes with a failed grasp: retried / recovered / placed | "
            "said done but not placed | turn budget hit |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for run in runs:
        s, eps = load(run)
        recs = [recovery(e) for e in eps]
        with_fail = [(e, r) for e, r in zip(eps, recs) if r["failed"]]
        n = s["n"]
        lines.append(
            f"| {run} | {s['model']} | {s['condition']} | {s['placed']}/{n} | {s['lifted']}/{n} | {s['median_turns']} | "
            f"{s['median_wall_s']:.0f} / {s['median_sim_s']:.0f} | {s['median_tokens']:.0f} | {s['mean_dollars']:.2f} | "
            f"{sum(r['attempts'] for r in recs)} | {sum(r['failed'] for r in recs)} | "
            f"{len(with_fail)}: {sum(r['retried'] for _, r in with_fail)} / {sum(r['recovered'] for _, r in with_fail)} / "
            f"{sum(e['placed'] for e, _ in with_fail)} | {sum(r['false_done'] for r in recs)} | "
            f"{s['ended'].get('turn budget', 0)} |")
    return "\n".join(lines)


def episodes(run: str) -> str:
    _, eps = load(run)
    out = []
    for e in eps:
        r = recovery(e)
        verdict = "placed" if e["placed"] else "lifted" if e["lifted"] else "failed"
        out.append(f"{run} seed {e['seed']:2}: {verdict:6} off {e['off_target_m']:.3f} m, tilt {e['tilt_deg']}, "
                   f"released {e['released']}, turns {e['turns']} ({e['ended']}), grasps {r['attempts']} "
                   f"(failed {r['failed']}, recovered {r['recovered']}), ik failures {r['ik_failures']}, "
                   f"tipped {r['tipped']}, ${e['dollars']:.2f}")
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
        ImageDraw.Draw(row).text((8, 6), "final" if t < 0 else f"after turn {t}", fill=(255, 255, 0))
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
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    if a.sheet:
        turns = [int(t) for t in a.runs]
        print(sheet(a.sheet, turns, a.out or RUNS / "sheet.jpg"))
        return 0
    print(table(a.runs))
    print()
    for run in a.runs:
        print(episodes(run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
