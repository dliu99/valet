"""Score Jev's end-of-episode judgement against the grader: each episode's clean final wrist frame, three yes/no
questions in one request, compared with `score(truth())` from result.json.

    uv run python -m claude_policy.jev_eval runs/<run>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .agent import VERIFY_QUESTION

QUESTIONS = {
    "verify": (VERIFY_QUESTION, lambda r: r["placed"]),
    "upright": ("Is the white box standing upright (on one of its small square ends, not lying on a long side)?",
                lambda r: r["tilt_deg"] <= 15),
    "on_square": ("Is the white box sitting on the blue square, covering its middle?", lambda r: r["off_target_m"] <= 0.03),
}


def main(argv: list[str] | None = None) -> int:
    from typesafe_sdk import Image, Noul, TypeSafeClient

    from .jev import CAMERAS, IMAGES_HEADER
    from .run import load_env

    argv = sys.argv[1:] if argv is None else argv
    load_env()
    client = TypeSafeClient()
    rows = []
    for d in sorted(Path(argv[0]).glob("seed*")):
        frame, result = d / "final-wrist.jpg", d / "result.json"
        if not (frame.exists() and result.exists()):
            continue
        r = json.loads(result.read_text())
        a = client.system_one(state={"wrist": Image(frame.read_bytes())},
                              questions={k: Noul(instructions=f"{q} ({CAMERAS['wrist']}.)") for k, (q, _) in QUESTIONS.items()},
                              extra_headers=IMAGES_HEADER)
        rows.append({"seed": r["seed"], **{k: (round(float(a.answers[k].noul), 2), bool(truth(r))) for k, (_, truth) in QUESTIONS.items()}})
    for k in QUESTIONS:
        right = sum((p >= 0.5) == t for p, t in (row[k] for row in rows))
        fp = sum(p >= 0.5 and not t for p, t in (row[k] for row in rows))
        fn = sum(p < 0.5 and t for p, t in (row[k] for row in rows))
        print(f"{k:10} {right}/{len(rows)} right, {fp} false yes, {fn} false no")
    for row in rows:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
