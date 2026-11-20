"""Everything a run did, on disk, so thresholds can be tuned and failures replayed.

runs/<recipe>-<time>/
  events.jsonl   one line per motion, skill, check, escalation and decision
  checks.jsonl   one line per Jev question: id, text, answer, how the runtime read it, frame file.
                 Add an "actual" field by hand (true/false) to measure Jev on your own frames.
  frames/        every frame a question was asked about, NNNN-<tag>.jpg
  result.json    outcome, layers placed, counts, time
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

from .judge import Answer, Locate, OneOf, Question, YesNo, plain


class Records:
    def __init__(self, root: str | Path, name: str):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = Path(root) / f"{name}-{stamp}"
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        self._n = 0
        self._t0 = time.monotonic()

    def frame(self, frame: np.ndarray, tag: str) -> str:
        self._n += 1
        rel = f"frames/{self._n:04d}-{tag.replace(':', '_').replace(' ', '_')}.jpg"
        PILImage.fromarray(frame).save(self.dir / rel, quality=90)
        return rel

    def event(self, kind: str, **data: Any) -> None:
        self._append("events.jsonl", {"t": round(time.monotonic() - self._t0, 3), "kind": kind, **data})

    def checks(self, step: str, frame_file: str, questions: dict[str, Question], answers: dict[str, Answer], readings: dict[str, str]) -> None:
        for qid, q in questions.items():
            self._append(
                "checks.jsonl",
                {
                    "step": step,
                    "id": qid,
                    "question": _text(q),
                    "answer": plain(answers[qid]),
                    "read_as": readings.get(qid),
                    "frame": frame_file,
                },
            )

    def result(self, **data: Any) -> None:
        data["seconds"] = round(time.monotonic() - self._t0, 1)
        (self.dir / "result.json").write_text(json.dumps(data, indent=2, default=_json))

    def _append(self, name: str, row: dict[str, Any]) -> None:
        with open(self.dir / name, "a") as f:
            f.write(json.dumps(row, default=_json) + "\n")


def _text(q: Question) -> str:
    if isinstance(q, YesNo):
        return q.question
    if isinstance(q, Locate):
        return q.target
    if isinstance(q, OneOf):
        return f"{q.question} {list(q.options)}"
    return str(q)


def _json(o: Any) -> Any:
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)
