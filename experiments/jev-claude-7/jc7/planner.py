"""Claude writes the plan: the moves, the grips, and the checks Jev must pass before the plan goes on.

Claude is called once to plan, again only when a step fails (to plan from where the arm is), and once after the
episode to write a lesson for the next one. The prompt states facts about the arm and the measuring; how to grasp
is Claude's to decide."""

from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass, field

from PIL import Image

from .jev import load_env

MODEL = "claude-opus-5-5"
# $ per million tokens: input, output, cache read, cache write (writes at 1.25x input)
PRICES = {
    "claude-opus-5-5": (4.0, 20.0, 0.20, 5.0),
    "claude-sonnet-5-5": (2.0, 10.0, 0.10, 2.5),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
}

SYSTEM = """You plan how a small robot arm (SO-101, two-jaw gripper) picks up a white box and sets it down upright on a blue square on the table, in a simulator. At the end the jaws must be open and off the box. Code runs your plan step by step and stops it when a step fails; you then get the log and plan again from where the arm is.

Units are cm. x is forward from the base, y to the arm's left, z up from the table. The base turns about (3.9, 0); a point's bearing is its direction from there, in degrees from +x toward +y.

The hand
- G is the grasp point between the jaws. pitch is the jaws' angle below horizontal (90 points straight down). roll is the wrist roll, -157 to 163.
- The jaws close along a horizontal direction at angle bearing + roll: at roll -90 across the line from the base, at roll 0 along it. This is exact when pitch is steep; at low pitch, a roll other than -90 or 90 tilts it.
- The fixed jaw's inner face is 2.0 cm from G, against the closing direction, and never moves. The jaw tips reach 1.5 cm past G.
- The moving jaw's tip is 0.9, 2.2, 3.5, 4.7, 5.8, 6.7, 7.4 cm from G along the closing direction at gripper 10, 20, 30, 40, 50, 60, 70. Opening also swings it back toward the wrist. Gripper 0 is shut.
- Closing stops where the jaws meet something. A held box is held by friction only.
- Where G can reach, as distance from (3.9, 0) in cm, by pitch and height of G:
  pitch 90: z3 8-27, z6 8-25, z10 12-20, z15 none
  pitch 60: z3 8-34, z6 10-33, z10 10-32, z15 12-29, z20 none
  pitch 45: z3 8-37, z10 12-36, z15 10-33, z20 12-29
  pitch 30: z3 18-38, z10 16-38, z20 10-34
- A move takes G in a straight line at servo speed. If the asked pitch cannot reach, the nearest pitch within 15 degrees that can is used (the log says which); otherwise the step fails.

The scene is measured with the wrist camera only: Jev, a fast vision model, boxes things in frames and code turns the boxes into table positions. Typical errors: box centre 0.5 cm (up to about 1.1), box side 1 cm, box height 1.5 cm (sometimes 3); square centre 0.5 cm (up to about 1.2). The box's yaw is not measured; judge it from the overhead image if it matters to you.

Steps
- move: G = [x, y, z] relative to `rel`: "table" is absolute, "box" is from the box's estimated footprint centre (on the table), "square" is from the square's centre (on the table). Also pitch, roll, and gripper (null keeps it).
- grip: drive the gripper toward `gripper`; it stops early on contact. `expect` = [lo, hi] is the reading you expect afterwards; outside it the plan stops.
- check: a yes/no `question` Jev answers about the current wrist frame (it judges what is where in the image well and distances poorly; refer to the image as `wrist`). `want` is the answer that lets the plan go on.
- look: measure the box or the square again (`rel`); the camera moves to look at it, and later steps relative to it use the new estimate.
Fill fields a step does not use with null, "" or "yes". Give each step a few words of `why`. Use few checks, where they decide something."""

STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "do": {"type": "string", "enum": ["move", "grip", "check", "look"]},
        "rel": {"type": "string", "enum": ["table", "box", "square"]},
        "G": {"type": ["array", "null"], "items": {"type": "number"}},
        "pitch": {"type": ["number", "null"]},
        "roll": {"type": ["number", "null"]},
        "gripper": {"type": ["number", "null"]},
        "expect": {"type": ["array", "null"], "items": {"type": "number"}},
        "question": {"type": "string"},
        "want": {"type": "string", "enum": ["yes", "no"]},
        "why": {"type": "string"},
    },
    "required": ["do", "rel", "G", "pitch", "roll", "gripper", "expect", "question", "want", "why"],
    "additionalProperties": False,
}
PLAN_SCHEMA = {
    "type": "object",
    "properties": {"approach": {"type": "string"}, "steps": {"type": "array", "items": STEP_SCHEMA}},
    "required": ["approach", "steps"],
    "additionalProperties": False,
}
LESSON_SCHEMA = {
    "type": "object",
    "properties": {"lesson": {"type": "string"}},
    "required": ["lesson"],
    "additionalProperties": False,
}


def image_block(im: Image.Image, max_side: int = 640) -> dict:
    im = im.copy()
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=85)
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.standard_b64encode(buf.getvalue()).decode()}}


@dataclass
class ClaudeMeter:
    model: str = MODEL
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    seconds: float = 0.0
    log: list[dict] = field(default_factory=list)

    @property
    def usd(self) -> float:
        i, o, r, w = PRICES[self.model]
        return (self.input_tokens * i + self.output_tokens * o + self.cache_read * r + self.cache_write * w) / 1e6


class Planner:
    def __init__(self, effort: str = "medium", model: str = MODEL):
        import anthropic

        load_env()
        self.client = anthropic.Anthropic()
        self.effort = effort
        self.model = model
        self.meter = ClaudeMeter(model)
        self.messages: list[dict] = []

    def _call(self, content: list[dict], schema: dict, kind: str, max_tokens: int = 32000) -> dict:
        self.messages.append({"role": "user", "content": content})
        t0 = time.monotonic()
        with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=self.messages,
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
        ) as stream:
            r = stream.get_final_message()
        dt = time.monotonic() - t0
        u = r.usage
        self.meter.calls += 1
        self.meter.input_tokens += u.input_tokens
        self.meter.output_tokens += u.output_tokens
        self.meter.cache_read += getattr(u, "cache_read_input_tokens", 0) or 0
        self.meter.cache_write += getattr(u, "cache_creation_input_tokens", 0) or 0
        self.meter.seconds += dt
        if r.stop_reason == "refusal":
            raise RuntimeError("Claude declined to plan")
        text = next((b.text for b in r.content if b.type == "text"), "")
        self.messages.append({"role": "assistant", "content": r.content})
        out = json.loads(text)
        self.meter.log.append({"kind": kind, "s": round(dt, 1), "in": u.input_tokens, "out": u.output_tokens, "reply": out})
        return out

    def plan(self, brief: str, images: list[tuple[str, Image.Image]]) -> dict:
        content: list[dict] = []
        for caption, im in images:
            content.append({"type": "text", "text": caption})
            content.append(image_block(im))
        content.append({"type": "text", "text": brief})
        return self._call(content, PLAN_SCHEMA, "plan")

    def replan(self, report: str, images: list[tuple[str, Image.Image]]) -> dict:
        content: list[dict] = []
        for caption, im in images:
            content.append({"type": "text", "text": caption})
            content.append(image_block(im))
        content.append({"type": "text", "text": report})
        return self._call(content, PLAN_SCHEMA, "replan")

    def lesson(self, report: str, images: list[tuple[str, Image.Image]]) -> str:
        content: list[dict] = []
        for caption, im in images:
            content.append({"type": "text", "text": caption})
            content.append(image_block(im))
        content.append({"type": "text", "text": report})
        return self._call(content, LESSON_SCHEMA, "lesson", max_tokens=16000)["lesson"]
