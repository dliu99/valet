"""Compact tables for the report from results/perception.jsonl and the benchmark's per-episode logs.

    uv run python -m sam3p.tables
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _med(v):
    v = [x for x in v if x is not None]
    return "-" if not v else f"{statistics.median(v):.2f}"


def _p90(v):
    v = sorted(x for x in v if x is not None)
    return "-" if not v else f"{v[min(len(v) - 1, int(0.9 * len(v)))]:.2f}"


def perception(rows: list[dict]) -> str:
    """All views pooled; only frames where the object is wholly inside the frame (not cut by its edge)."""
    head = ("| method | s / frame | box found | box mask IoU | box bbox IoU | box centre cm, fit (med / p90) | box centre cm, bbox centre "
            "| box yaw err deg | box height err cm (start view) | square found | square mask IoU | square cm, fit (med / p90) | square cm, bbox centre |")
    out = [head, "|" + "---|" * 13]
    for m in dict.fromkeys(r["method"] for r in rows):
        rs = [r for r in rows if r["method"] == m]
        b = [r for r in rs if r["box_visible"] > 200]
        s = [r for r in rs if r["square_visible"] > 200]
        h = [abs(r["h_cm"]) for r in b if r["view"] == "start" and r.get("h_cm") is not None]
        # box centre: the fit with the start view's size where there is one (what the policy does), else the free fit
        bc = [r.get("hw_box_cm", r.get("box_cm")) if r["view"] != "start" else r.get("box_cm") for r in b]
        out.append(" | ".join([f"| {m}", _med(r.get("s") for r in rs), f"{sum(r['box_found'] for r in b)}/{len(b)}",
                               _med(r.get("box_mask_iou") for r in b), _med(r["box_bbox_iou"] for r in b),
                               f"{_med(bc)} / {_p90(bc)}", f"{_med(r['box_bbcentre_cm'] for r in b)} / {_p90(r['box_bbcentre_cm'] for r in b)}",
                               _med(r.get("yaw_deg") for r in b), _med(h), f"{sum(r['square_found'] for r in s)}/{len(s)}",
                               _med(r.get("square_mask_iou") for r in s), f"{_med(r.get('square_cm') for r in s)} / {_p90(r.get('square_cm') for r in s)}",
                               f"{_med(r['square_bbcentre_cm'] for r in s)} / {_p90(r['square_bbcentre_cm'] for r in s)}"]) + " |")
    return "\n".join(out)


def by_view(rows: list[dict], methods: tuple[str, ...]) -> str:
    views = list(dict.fromkeys(r["view"] for r in rows))
    out = ["| view | " + " | ".join(f"{m} box cm" for m in methods) + " | " + " | ".join(f"{m} square cm" for m in methods) + " |",
           "|" + "---|" * (1 + 2 * len(methods))]
    for v in views:
        cells = []
        for key in ("box", "square"):
            for m in methods:
                rs = [r for r in rows if r["method"] == m and r["view"] == v and r[f"{key}_visible"] > 200]
                if m == "jev":
                    vals = [r[f"{key}_bbcentre_cm"] for r in rs]
                elif key == "box":
                    vals = [r.get("hw_box_cm", r.get("box_cm")) if v != "start" else r.get("box_cm") for r in rs]
                else:
                    vals = [r.get("square_cm") for r in rs]
                cells.append(f"{_med(vals)} / {_p90(vals)}")
        out.append(f"| {v} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> int:
    rows = [json.loads(line) for line in (RESULTS / "perception.jsonl").read_text().splitlines() if line.strip()]
    print(perception(rows))
    print()
    print(by_view(rows, tuple(m for m in ("truth-mask", "sam3-336", "sam3-504", "sam3-1008", "jev") if any(r["method"] == m for r in rows))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
