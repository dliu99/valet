"""Narrow, typed questions about one wrist-camera frame, answered by Jev.

The runtime never asks "what should I do next?". It asks yes/no questions about the
scene ("is the cheese on the sandwich?"), where something is, or which of a few named
cases holds, and decides in code. Question ids are for code only and are not sent to
the model, so the question text must carry its whole meaning; it refers to the frame
as `wrist`.
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Protocol

import numpy as np
from PIL import Image as PILImage


@dataclass(frozen=True)
class YesNo:
    question: str
    yes: str | None = None  # what counts as yes
    no: str | None = None  # what counts as no


@dataclass(frozen=True)
class Locate:
    target: str  # e.g. "the slice of cheese in `wrist`"


@dataclass(frozen=True)
class OneOf:
    question: str
    options: dict[str, str]  # option name -> what it covers


Question = YesNo | Locate | OneOf


@dataclass(frozen=True)
class Box:
    top: float
    bottom: float
    left: float
    right: float
    confidence: float | None = None


@dataclass(frozen=True)
class Picked:
    choice: str
    confidence: float
    probabilities: dict[str, float]


Answer = float | Box | Picked  # YesNo -> probability of yes


def plain(a: Answer) -> Any:
    """An answer as JSON-ready data, for records and escalations."""
    return asdict(a) if is_dataclass(a) else round(float(a), 4)


class Judge(Protocol):
    def ask(self, frame: np.ndarray, questions: dict[str, Question], facts: dict[str, Any] | None = None) -> dict[str, Answer]:
        """Answer every question about `frame` in one request. `facts` is extra state the
        questions may refer to by name."""
        ...


@dataclass(frozen=True)
class Thresholds:
    yes: float = 0.7  # a YesNo at or above this is yes
    no: float = 0.3  # at or below this is no; between is unsure
    box_min_confidence: float = 0.5

    def read(self, p: float) -> str:
        return "yes" if p >= self.yes else "no" if p <= self.no else "unsure"


def jpeg_bytes(frame: np.ndarray, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    PILImage.fromarray(frame).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class JevJudge:
    """Judge backed by TypeSafe's System One API. Needs TYPESAFE_API_KEY.

    Images and Box questions are labelled "Dreamer-ready" by TypeSafe: fine for a
    prototype, with accuracy that may change between model releases. Box needs Jev 1.14.
    """

    IMAGES_HEADER = {"TypeSafe-Danger-Zone-Images": "true"}

    def __init__(self, model: str | None = None, timeout: float = 10.0, client: Any = None):
        from typesafe_sdk import TypeSafeClient

        self._client = client or TypeSafeClient()
        self._model = model
        self._timeout = timeout

    def ask(self, frame: np.ndarray, questions: dict[str, Question], facts: dict[str, Any] | None = None) -> dict[str, Answer]:
        from typesafe_sdk import Image

        state: dict[str, Any] = {"wrist": Image(jpeg_bytes(frame)), **(facts or {})}
        response = self._client.system_one(
            state=state,
            questions={qid: _to_sdk(q) for qid, q in questions.items()},
            model=self._model,
            timeout=self._timeout,
            extra_headers=self.IMAGES_HEADER,
        )
        return {qid: _from_sdk(response.answers[qid]) for qid in questions}

    def close(self) -> None:
        self._client.close()


def _to_sdk(q: Question):
    from typesafe_sdk import Box as SdkBox, Choice, Noul, NoulCriteria

    if isinstance(q, YesNo):
        criteria = NoulCriteria(true=q.yes, false=q.no) if (q.yes or q.no) else None
        return Noul(instructions=q.question, criteria=criteria)
    if isinstance(q, Locate):
        return SdkBox(instructions=q.target)
    if isinstance(q, OneOf):
        return Choice(instructions=q.question, criteria=dict(q.options))
    raise TypeError(f"unknown question type {type(q).__name__}")


def _from_sdk(a) -> Answer:
    if a.type == "noul":
        return float(a.noul)
    if a.type == "box":
        b = a.box
        return Box(b.top, b.bottom, b.left, b.right, a.confidence)
    if a.type == "choice":
        return Picked(a.choice, float(a.confidence), dict(a.probabilities))
    raise TypeError(f"unexpected answer type {a.type}")
