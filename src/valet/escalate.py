"""Who decides when Jev is unsure: a person at the terminal, or Claude.

An escalation carries the frames, the step, every answer Jev gave and the decisions the
runtime can carry out. The decider picks one of those; nothing else is executed.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .judge import jpeg_bytes

DECISIONS = {
    "continue": "The step succeeded; go on to the next one.",
    "retry": "Do the same step again.",
    "rescan": "Look over the whole table again, then retry the step.",
    "skip": "Leave this layer out and go on.",
    "abort": "Stop and send the arm home.",
}


@dataclass
class Escalation:
    step: str  # e.g. "place cheese"
    reason: str  # why the runtime could not decide
    answers: dict[str, Any]  # Jev's answers for this step, by question id
    frames: list[np.ndarray] = field(default_factory=list)  # most recent last
    allowed: list[str] = field(default_factory=lambda: list(DECISIONS))
    facts: dict[str, Any] = field(default_factory=dict)  # stack height, attempts, ...


@dataclass(frozen=True)
class Decision:
    choice: str
    reason: str
    by: str


class Escalator(Protocol):
    def decide(self, esc: Escalation) -> Decision: ...


class Abort:
    """For unattended runs: anything unsure stops the run."""

    def decide(self, esc: Escalation) -> Decision:
        return Decision("abort", f"no decider configured ({esc.reason})", "abort")


class AskPerson:
    def __init__(self, show_frame=None):
        self._show = show_frame  # callable(frame) -> path, so the person can open the picture

    def decide(self, esc: Escalation) -> Decision:
        print(f"\n[valet] unsure at '{esc.step}': {esc.reason}")
        for qid, a in esc.answers.items():
            print(f"  {qid}: {a}")
        if self._show and esc.frames:
            print(f"  frame: {self._show(esc.frames[-1])}")
        options = "/".join(esc.allowed)
        while True:
            choice = input(f"  decide ({options}): ").strip().lower()
            if choice in esc.allowed:
                return Decision(choice, "person", "person")


SYSTEM = """You supervise a small robot arm (SO-101, one wrist camera looking down past the gripper) \
that assembles a sandwich on a cutting board, one layer at a time. Code runs the task and checks each \
step by asking a fast vision model narrow yes/no questions. You are called only when those answers \
were unsure or contradicted each other. Look at the pictures, weigh the answers, and choose exactly \
one of the allowed decisions. Prefer the cheapest decision that leaves the sandwich correct; choose \
"abort" only when continuing could damage something or the task cannot be finished."""


class AskClaude:
    """Escalate to Claude Opus 5.5 with the frames and Jev's answers; returns one allowed decision.

    Server-side refusal fallback is on (`fallbacks="default"`): if the model declines, the API
    re-runs the request on a fallback model. A decline from the whole chain becomes "abort".
    """

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "high", client: Any = None):
        import anthropic

        self._client = client or anthropic.Anthropic()
        self._model = model
        self._effort = effort

    def decide(self, esc: Escalation) -> Decision:
        content: list[dict[str, Any]] = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.standard_b64encode(jpeg_bytes(f)).decode()},
            }
            for f in esc.frames[-2:]
        ]
        content.append({"type": "text", "text": _context(esc)})
        schema = {
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": esc.allowed},
                "reason": {"type": "string"},
            },
            "required": ["decision", "reason"],
            "additionalProperties": False,
        }
        response = self._client.beta.messages.create(
            model=self._model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SYSTEM,
            messages=[{"role": "user", "content": content}],
            output_config={"effort": self._effort, "format": {"type": "json_schema", "schema": schema}},
        )
        if response.stop_reason == "refusal":
            return Decision("abort", "the model declined to decide", self._model)
        if response.stop_reason == "max_tokens":
            return Decision("abort", "the model ran out of tokens before deciding", self._model)
        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
        if data["decision"] not in esc.allowed:  # the schema's enum should prevent this
            return Decision("abort", f"model chose {data['decision']!r}, which is not allowed", response.model)
        return Decision(data["decision"], data["reason"], response.model)


def _context(esc: Escalation) -> str:
    return json.dumps(
        {
            "step": esc.step,
            "why_you_were_called": esc.reason,
            "vision_answers": esc.answers,
            "facts": esc.facts,
            "allowed_decisions": {k: DECISIONS[k] for k in esc.allowed},
            "pictures": "The last picture is the most recent, taken after the step." if len(esc.frames) > 1 else "One picture, taken after the step.",
        },
        indent=2,
        default=str,
    )
