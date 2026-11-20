"""Print a run's per-seed table for the report, and save a strip of wrist frames from one episode.

    uv run python summarize.py runs/<run>
    uv run python summarize.py runs/<run> --strip 3 10 20 30 --seed 4 --out frames/seed04.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def table(run: Path) -> None:
    data = json.loads((run / "summary.json").read_text())
    print("| seed | placed | lifted | touched | closest cm | off target cm | end | ticks | episode s | Jev calls | Claude calls | $ |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in data["results"]:
        print(f"| {r['seed']} | {'yes' if r['placed'] else ''} | {'yes' if r['lifted'] else ''} | {'yes' if r['touched'] else ''} | "
              f"{r['closest_cm']} | {r['off_target_m'] * 100:.1f} | {r['end']} | {r['ticks']} | {r['episode_s']} | "
              f"{r['jev_calls']} | {r['claude_calls']} | {r['usd']['total']:.3f} |")
    print()
    print(json.dumps(data["summary"], indent=1))


def strip(run: Path, seed: int, ticks: list[int], out: Path, rep: str = "") -> None:
    ep = run / (f"seed{seed:02d}" + (f"-{rep}" if rep else ""))
    rows = {json.loads(l)["tick"]: json.loads(l) for l in (ep / "ticks.jsonl").read_text().splitlines() if '"move"' in l and '"rewrite"' not in l}
    w, h = 320, 180
    sheet = Image.new("RGB", (w * len(ticks), h + 34), "white")
    for i, t in enumerate(ticks):
        sheet.paste(Image.open(ep / f"t{t:03d}.jpg").resize((w, h)), (i * w, 0))
        r = rows[t]
        d = ImageDraw.Draw(sheet)
        d.text((i * w + 4, h + 2), f"t{t} {r['arm']['jaws']}, jaws {r['arm']['jaws_height_cm']} cm up", fill="black")
        d.text((i * w + 4, h + 17), f"Jev: {r['move']} {r['amount'] or ''}", fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=85)
    print(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--seed", type=int)
    p.add_argument("--strip", type=int, nargs="*")
    p.add_argument("--out", type=Path)
    p.add_argument("--rep", default="", help="r0, r1 for runs with --repeats")
    a = p.parse_args()
    if a.strip:
        strip(a.run, a.seed, a.strip, a.out, a.rep)
    else:
        table(a.run)


if __name__ == "__main__":
    main()
