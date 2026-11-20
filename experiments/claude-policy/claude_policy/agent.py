"""Claude runs the arm: one goal sentence, three tools, and Claude plans and judges the rest.

The loop is append-only (every response goes back unchanged), so prompt caching and preserved thinking hold."""

from __future__ import annotations

import base64
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol

from .arm import Arm, Unreachable

GOAL = "Pick up the white box and set it down upright on the blue square, then let go and move the jaws clear of it."

SYSTEM = (
    "You control a robot arm through tools: an SO-101 with five joints, a two-jaw gripper and a camera on the wrist. "
    "You plan and judge everything yourself. Tool calls in one turn run in order; a failed call skips the rest. "
    "When the goal is met, or you are sure it can't be, answer in one sentence without calling tools."
)

# $ per million tokens: input, 5-minute cache write, cache read, output.
PRICES = {
    "claude-opus-5-5": (4.0, 5.0, 0.20, 20.0),
    "claude-opus-5": (5.0, 6.25, 0.50, 25.0),
    "claude-opus-4-8": (5.0, 6.25, 0.50, 25.0),
    "claude-fable-5-2": (8.0, 10.0, 0.20, 40.0),
    "claude-sonnet-5-5": (2.0, 2.5, 0.10, 10.0),
}

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
CHECK_PROPERTY = {
    "type": "string",
    "description": "Optional yes/no question about the wrist frame right after this call, answered in about a second "
    "by a fast vision model. If the answer is no, the rest of this turn's calls are skipped.",
}


@dataclass(frozen=True)
class Config:
    model: str = "claude-opus-5-5"
    effort: str = "medium"
    max_side: int = 640
    cams: tuple[str, ...] = ("wrist", "side")
    jev_checks: bool = False
    overlay: bool = False
    max_turns: int = 40
    max_cost_usd: float = 2.0


class Checker(Protocol):
    def ask(self, jpeg: bytes, question: str) -> tuple[float, dict]:
        """Probability of yes, and usage/latency for the record."""
        ...


def tools(cfg: Config) -> list[dict]:
    look = {
        "name": "look",
        "description": (
            "Frames from the wrist camera and a fixed side camera, plus the joint state. Wrist frames are upright: at "
            "wrist roll -90, image up is away from the base and image right is the arm's right. The side camera is off "
            "the arm's right, looking across the table; image right is away from the base. State: joint angles in "
            "degrees; `grasp_point`, the spot between the jaw tips, in metres in the base frame (x forward, y left, "
            "z up from the table top); servo currents in amps."
            + (" Both frames carry a red grid on the table top every 5 cm (10 cm in the side frame), labelled x,y in "
               "cm, a red ring at the grasp point and a red + on the table right below it." if cfg.overlay else "")
        ),
        "input_schema": {
            "type": "object",
            "properties": {"cams": {"type": "array", "items": {"type": "string", "enum": ["wrist", "side"]},
                                    "description": "Default: both."}},
            "additionalProperties": False,
        },
    }
    move = {
        "name": "move",
        "description": (
            "Move the arm and wait until it stops (at most 4 s; the servos slew at up to 60 deg/s). Give either `xyz` "
            "with `pitch_deg`, solved by IK for the grasp point, or `joints`. The gripper keeps its setting. Returns "
            "the state reached."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "xyz": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
                        "description": "Grasp point target, metres, base frame."},
                "pitch_deg": {"type": "number", "description": "How far the jaws point below horizontal; 90 is straight down."},
                "roll_deg": {"type": "number", "description": "Wrist roll, default -90 (wrist camera level). Turning it "
                             "turns the line the jaws close along."},
                "joints": {"type": "object", "description": "Goals in degrees for any of these joints.",
                           "properties": {j: {"type": "number"} for j in JOINT_NAMES}, "additionalProperties": False},
            },
            "additionalProperties": False,
        },
    }
    gripper = {
        "name": "gripper",
        "description": (
            "`close` squeezes until the jaws stop on something, then holds, and reports where they stopped (0 is fully "
            "shut). `open` opens to `percent` (default 70; 100 is widest) and waits."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"action": {"type": "string", "enum": ["open", "close"]},
                           "percent": {"type": "number", "minimum": 0, "maximum": 100}},
            "required": ["action"],
            "additionalProperties": False,
        },
    }
    if cfg.jev_checks:
        for t in (move, gripper):
            t["input_schema"]["properties"]["check"] = CHECK_PROPERTY
    return [look, move, gripper]


@dataclass
class Tally:
    claude_calls: int = 0
    input_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    claude_usd: float = 0.0
    claude_s: float = 0.0
    jev_calls: int = 0
    jev_input_tokens: int = 0
    jev_s: float = 0.0
    tool_calls: dict[str, int] = field(default_factory=dict)

    def add_claude(self, model: str, usage: Any, seconds: float) -> float:
        p_in, p_write, p_read, p_out = PRICES.get(model, PRICES["claude-opus-5-5"])
        i, w = usage.input_tokens or 0, usage.cache_creation_input_tokens or 0
        r, o = usage.cache_read_input_tokens or 0, usage.output_tokens or 0
        usd = (i * p_in + w * p_write + r * p_read + o * p_out) / 1e6
        self.claude_calls += 1
        self.input_tokens += i
        self.cache_write_tokens += w
        self.cache_read_tokens += r
        self.output_tokens += o
        self.claude_usd += usd
        self.claude_s += seconds
        return usd


def image_block(jpeg: bytes) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(jpeg).decode()}}


class Tools:
    """Runs Claude's tool calls on the arm and keeps the frames for the record."""

    def __init__(self, arm: Arm, cfg: Config, checker: Checker | None, record: Callable[[dict], None],
                 save_frame: Callable[[str, bytes], None], tally: Tally):
        self.arm, self.cfg, self.checker = arm, cfg, checker
        self.record, self.save_frame, self.tally = record, save_frame, tally

    def look(self, cams: tuple[str, ...] | None = None) -> list[dict]:
        frames = self.arm.frames(tuple(cams or self.cfg.cams))
        blocks: list[dict] = []
        for cam, jpeg in frames.items():
            self.save_frame(cam, jpeg)
            blocks += [{"type": "text", "text": cam}, image_block(jpeg)]
        blocks.append({"type": "text", "text": json.dumps(self.arm.state())})
        return blocks

    def run(self, name: str, args: dict) -> tuple[list[dict] | str, bool, bool]:
        """Returns (content, is_error, stop_the_turn)."""
        self.tally.tool_calls[name] = self.tally.tool_calls.get(name, 0) + 1
        try:
            if name == "look":
                return self.look(args.get("cams")), False, False
            if name == "move":
                if args.get("joints"):
                    out = self.arm.move_joints(args["joints"])
                elif args.get("xyz") is not None and args.get("pitch_deg") is not None:
                    out = self.arm.move_to(args["xyz"], args["pitch_deg"], args.get("roll_deg", -90.0))
                else:
                    return "give either xyz with pitch_deg, or joints", True, True
            elif name == "gripper":
                if args.get("action") == "close":
                    out = self.arm.close()
                elif args.get("action") == "open":
                    out = self.arm.open(float(args.get("percent", 70.0)))
                else:
                    return "action must be open or close", True, True
            else:
                return f"no tool named {name}", True, True
        except (Unreachable, ValueError) as e:
            return str(e), True, True
        text = json.dumps(out)
        question = args.get("check") if self.cfg.jev_checks else None
        if question and self.checker:
            p, info = self._check(question)
            verdict = "yes" if p >= 0.5 else "no"
            text += f"\ncheck {json.dumps(question)}: {verdict} (p_yes={p:.2f})"
            if verdict == "no":
                return text + "\nThe rest of this turn's calls were skipped.", False, True
        return text, False, False

    def _check(self, question: str) -> tuple[float, dict]:
        jpeg = self.arm.frames(("wrist",), max_side=640, overlay=False)["wrist"]  # Jev takes 280-1568 px a side
        self.save_frame("check", jpeg)
        t0 = time.monotonic()
        p, info = self.checker.ask(jpeg, question)
        dt = time.monotonic() - t0
        self.tally.jev_calls += 1
        self.tally.jev_input_tokens += int(info.get("input_tokens") or 0)
        self.tally.jev_s += dt
        self.record({"type": "jev", "question": question, "p_yes": round(p, 4), "seconds": round(dt, 2), **info})
        return p, info


def request(cfg: Config, messages: list) -> dict:
    return dict(
        model=cfg.model,
        max_tokens=16000,
        system=SYSTEM,
        tools=tools(cfg),
        messages=messages,
        thinking={"type": "adaptive", "display": "summarized"},
        output_config={"effort": cfg.effort},
        cache_control={"type": "ephemeral"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )


def run_episode(arm: Arm, client: Any, cfg: Config, record: Callable[[dict], None],
                save_frame: Callable[[str, bytes], None], checker: Checker | None = None) -> dict:
    tally = Tally()
    t = Tools(arm, cfg, checker, record, save_frame, tally)
    messages: list[dict] = [{"role": "user", "content": [{"type": "text", "text": GOAL}, *t.look()]}]
    end = "turn limit"
    final_text = ""
    for turn in range(cfg.max_turns):
        t0 = time.monotonic()
        resp = client.beta.messages.create(**request(cfg, messages))
        dt = time.monotonic() - t0
        usd = tally.add_claude(resp.model, resp.usage, dt)
        calls = [b for b in resp.content if b.type == "tool_use"]
        text = " ".join(b.text for b in resp.content if b.type == "text").strip()
        thinking = " ".join(getattr(b, "thinking", "") or "" for b in resp.content if b.type == "thinking").strip()
        record({"type": "claude", "turn": turn, "model": resp.model, "stop_reason": resp.stop_reason, "seconds": round(dt, 2),
                "usd": round(usd, 5), "usage": _usage(resp.usage), "text": text, "thinking": thinking,
                "calls": [{"name": c.name, "input": c.input} for c in calls]})
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use" or not calls:
            end = resp.stop_reason
            final_text = text
            break
        results, stopped = [], None
        for c in calls:
            if stopped:
                results.append({"type": "tool_result", "tool_use_id": c.id, "is_error": True,
                                "content": f"skipped: {stopped} stopped this turn"})
                continue
            content, is_error, stop = t.run(c.name, dict(c.input))
            results.append({"type": "tool_result", "tool_use_id": c.id, "content": content, "is_error": is_error})
            record({"type": "tool", "turn": turn, "name": c.name, "input": c.input, "is_error": is_error,
                    "result": content if isinstance(content, str) else "[frames]", "motion_s": round(arm.motion_s, 2)})
            if stop:
                stopped = c.name
        messages.append({"role": "user", "content": results})
        if tally.claude_usd > cfg.max_cost_usd:
            end = "cost cap"
            break
    return {"end": end, "final_text": final_text, "motion_s": round(arm.motion_s, 2), **asdict(tally)}


def _usage(u: Any) -> dict:
    return {k: getattr(u, k, None) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")}
