"""Tables for REPORT.md from results/*.jsonl: `uv run python report.py > results/tables.md`."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from apriltag_exp.errors import KINDS
from apriltag_exp.localize import METHODS

R = Path(__file__).parent / "results"


def load(name: str) -> list[dict]:
    p = R / name
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def cell(values: list[float]) -> str:
    v = np.array([x for x in values if x is not None and np.isfinite(x)])
    if not len(v):
        return "-"
    return f"{np.median(v) / 10:.1f} ({np.percentile(v, 90) / 10:.1f})"


def localization() -> None:
    rows = [r for r in load("localize.jsonl") if "results" in r]
    failed = [r for r in load("localize.jsonl") if "results" not in r]
    print(f"### Localization: {len(rows)} trials ({len(failed)} failed)\n")
    print("cm, median (90th percentile) over seeds. *World*: estimate minus truth. *Reach*: where the grasp point truly goes")
    print("when the arm is told, through the same model, to go to the estimate, minus the truth.\n")
    for what, label in (("box", "Box"), ("target", "Blue square")):
        for src, src_label in (("det", "detector pixels"), ("true", "true pixels (geometry alone)")):
            print(f"**{label}, {src_label}** — world / reach\n")
            print("| method | " + " | ".join(KINDS) + " |")
            print("|---|" + "---|" * len(KINDS))
            for m in METHODS:
                if what == "target" and m == "fk_tri":
                    continue
                cells = []
                for k in KINDS:
                    rs = [r["results"].get(f"{m}/{src}", {}) for r in rows if r["kind"] == k]
                    cells.append(f"{cell([x.get(f'{what}_world_mm') for x in rs])} / {cell([x.get(f'{what}_reach_mm') for x in rs])}")
                print(f"| {m} | " + " | ".join(cells) + " |")
            print()
    seen = defaultdict(lambda: defaultdict(int))
    for r in rows:
        for view, s in r["tag_seen"].items():
            seen[view][r["kind"]] += s
    n = defaultdict(int)
    for r in rows:
        n[r["kind"]] += 1
    print("**Tag in view** (trials whose frame showed the tag, of " + ", ".join(f"{n[k]} {k}" for k in KINDS) + ")\n")
    print("| view | " + " | ".join(KINDS) + " |")
    print("|---|" + "---|" * len(KINDS))
    for view in ("start", "box", "box2", "target"):
        print(f"| {view} | " + " | ".join(str(seen[view][k]) for k in KINDS) + " |")
    pe = [r["tag_pose_error_start"] for r in rows if r.get("tag_pose_error_start")]
    if pe:
        print(f"\nCamera pose from the tag in the start view (tag {np.median([p['side_px'] for p in pe]):.0f} px across): "
              f"{np.median([p['mm'] for p in pe]):.1f} mm, {np.median([p['deg'] for p in pe]):.2f} deg median; "
              f"{np.percentile([p['mm'] for p in pe], 90):.1f} mm, {np.percentile([p['deg'] for p in pe], 90):.2f} deg 90th pct.\n")


def calibration() -> None:
    rows = [r for r in load("localize.jsonl") if "results" in r]
    print("### Calibration from a sweep of tag views\n")
    print("Mount error after the fit, mm / deg, median (90th pct). `mount`: hand-eye only. `full`: mount, joint zeros and sag.\n")
    print("| errors on | nominal mount | hand-eye (mount only) | full fit | joint zeros left, deg | sag fitted (true) |")
    print("|---|---|---|---|---|---|")
    for k in KINDS:
        rs = [r for r in rows if r["kind"] == k]
        if not rs:
            continue

        def mm_deg(key):
            mm = [r["true_mount_vs_fit"][key][0] for r in rs]
            deg = [r["true_mount_vs_fit"][key][1] for r in rs]
            return f"{np.median(mm):.1f} / {np.median(deg):.2f} ({np.percentile(mm, 90):.1f} / {np.percentile(deg, 90):.2f})"

        left = [np.abs(np.array(r["fit"]["full"]["offsets_deg"][:4]) - np.array(r["errors"]["offsets_deg"][:4])).max() for r in rs]
        comp = [r["fit"]["full"]["compliance"] for r in rs]
        print(f"| {k} | {mm_deg('nominal')} | {mm_deg('mount')} | {mm_deg('full')} | {np.median(left):.2f} ({np.percentile(left, 90):.2f}) "
              f"| {np.median(comp):.2f} ({rs[0]['errors']['compliance']:.1f}) |")
    print()


def servo() -> None:
    rows = load("servo.jsonl")
    if not rows:
        return
    print("### Closed-loop probe: servo over the box on the image, then descend\n")
    print("Reach error at the grasp, cm, median (90th pct); failures in brackets.\n")
    print("| reference | " + " | ".join(KINDS) + " |")
    print("|---|" + "---|" * len(KINDS))
    for ref in ("nominal", "taught"):
        cells = []
        for k in KINDS:
            rs = [r for r in rows if r["kind"] == k and r["reference"] == ref]
            bad = sum("failed" in r for r in rs)
            cells.append(cell([r.get("reach_mm") for r in rs]) + (f" [{bad}]" if bad else ""))
        print(f"| {ref} | " + " | ".join(cells) + " |")
    print()


def policy() -> None:
    rows = load("policy.jsonl")
    if not rows:
        return
    print("### End to end\n")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["kind"], r["method"])].append(r)
    print("| errors | method | episodes | placed | lifted | median off target when lifted, cm | grasps retried | held check right |")
    print("|---|---|---|---|---|---|---|---|")
    for (k, m), rs in sorted(groups.items(), key=lambda kv: (kv[0][0] != "none", ["fk", "tag", "cal"].index(kv[0][1]))):
        checks = [c for r in rs for c in r.get("log", []) if c.get("check") == "held:code"]
        right = sum(c["held"] == c["truth_held"] for c in checks)
        retried = sum(any("attempt" in c for c in r.get("log", [])) for r in rs)
        off = [r["off_target_m"] * 100 for r in rs if r.get("lifted")]
        print(f"| {k} | {m} | {len(rs)} | {sum(r['placed'] for r in rs)} | {sum(r['lifted'] for r in rs)} | "
              f"{np.median(off) if off else float('nan'):.1f} | {retried} | {right}/{len(checks)} |")
    print()


def grid() -> None:
    rows = [r for r in load("grid.jsonl") if "views" in r]
    if not rows:
        return
    print("### A table grid drawn on the frame\n")
    print("Median over seeds of each frame's median grid point: px off in the frame / cm off on the table; and cm off when")
    print("the arm is told to go to the grid's centre. Tag-drawn grids use the tag where the frame shows it, else the last")
    print("tag view carried by FK.\n")
    print("| errors | view | tag in view | fk | tag | cal |")
    print("|---|---|---|---|---|---|")
    for k in ("offsets", "all"):
        for view in ("start", "box", "target"):
            vs = [r["views"][view] for r in rows if r["kind"] == k and view in r["views"]]
            if not vs:
                continue
            cells = []
            for m in ("fk", "tag", "cal"):
                px = np.median([v[m]["px"] for v in vs])
                cm = np.median([v[m]["cm"] for v in vs])
                rc = [v[m].get("reach_cm") for v in vs if v[m].get("reach_cm") is not None]
                cells.append(f"{px:.0f} px / {cm:.1f} cm" + (f"; reach {np.median(rc):.1f} cm" if rc else ""))
            print(f"| {k} | {view} | {sum(v['tag_seen'] for v in vs)}/{len(vs)} | " + " | ".join(cells) + " |")
    print()


def aspire() -> None:
    rows = [r for r in load("aspire.jsonl") if r.get("backend") != "oracle"]
    if not rows:
        return
    print("### Team 6's final program (PR #8), H2O `localize`, all errors on\n")
    print("| variant | episodes | placed | lifted | median off target when lifted, cm | localize calls | crashed |")
    print("|---|---|---|---|---|---|---|")
    groups = defaultdict(list)
    for r in rows:
        groups[r["variant"]].append(r)
    for v in ("no-tag", "tag-runtime", "tag-calibrated"):
        rs = groups.get(v, [])
        if not rs:
            continue
        off = [r["score"]["off_target_m"] * 100 for r in rs if r["score"].get("lifted")]
        print(f"| {v} | {len(rs)} | {sum(r['score']['placed'] for r in rs)} | {sum(r['score']['lifted'] for r in rs)} | "
              f"{np.median(off) if off else float('nan'):.1f} | {sum(r.get('localize_calls', 0) for r in rs)} | "
              f"{sum('crashed' in (r.get('error') or '') for r in rs)} |")
    print()


if __name__ == "__main__":
    localization()
    calibration()
    servo()
    grid()
    policy()
    aspire()
