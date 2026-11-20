"""The Physical Agent loop: the instruction once, then each turn the scene and wrist views and the arm state; the
model writes brief reasoning and makes one `execute_absolute_eef_sequence` call. History stays in context. The
episode ends when the model replies without a call, or at the turn budget.

Code only executes and records. The truth is logged after each call for grading and analysis; it never reaches
the model."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .demo import INSTRUCTION, Demo
from .scene import grade, observation_blocks, observe, open_episode
from .tool import TOOL

SYSTEM = (
    "You control a simulated SO-101 robot arm on a table. Each turn you see the scene camera, the wrist camera on "
    "the gripper, and the arm's state. Write a short paragraph of reasoning, then make one "
    "execute_absolute_eef_sequence call. Its result only says which targets ran; judge what happened from the "
    "next views. When the task is done, reply without a call."
)

# $ per million tokens: input, output, cache read, 5-minute cache write.
PRICES = {
    "claude-opus-5-5": (4.0, 20.0, 0.20, 5.0),
    "claude-sonnet-5-5": (2.0, 10.0, 0.10, 2.5),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
    "claude-fable-5-2": (8.0, 40.0, 0.20, 10.0),
}


@dataclass
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0

    def add(self, u) -> None:
        self.input += u.input_tokens or 0
        self.output += u.output_tokens or 0
        self.cache_read += getattr(u, "cache_read_input_tokens", 0) or 0
        self.cache_write += getattr(u, "cache_creation_input_tokens", 0) or 0

    def dollars(self, model: str) -> float:
        i, o, r, w = PRICES[model]
        return (self.input * i + self.output * o + self.cache_read * r + self.cache_write * w) / 1e6

    def total_tokens(self) -> int:
        return self.input + self.output + self.cache_read + self.cache_write


@dataclass
class Config:
    model: str = "claude-opus-5-5"
    effort: str = "medium"
    turns: int = 15
    max_tokens: int = 16000


@dataclass
class Episode:
    seed: int
    turns: int = 0
    ended: str = ""
    usage: Usage = field(default_factory=Usage)
    wall_s: float = 0.0
    sim_s: float = 0.0
    result: dict = field(default_factory=dict)


def first_message(obs, demo: Demo | None) -> list[dict]:
    blocks: list[dict] = list(demo.blocks()) if demo else []
    blocks.append({"type": "text", "text": f"Instruction: {INSTRUCTION}"})
    blocks += observation_blocks(obs, "Current observation:")
    return blocks


def run_episode(client, seed: int, cfg: Config, demo: Demo | None = None, out: Path | None = None) -> Episode:
    sim, arm, _ = open_episode(seed)
    ep = Episode(seed)
    log = _Log(out)
    t0 = time.time()
    obs = observe(arm)
    log.frames(0, obs)
    messages: list[dict] = [{"role": "user", "content": first_message(obs, demo)}]
    tool = {**TOOL, "strict": True}
    while True:
        if ep.turns >= cfg.turns:
            ep.ended = "turn budget"
            break
        t_call = time.time()
        resp = client.messages.create(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            system=SYSTEM,
            tools=[tool],
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": cfg.effort},
            cache_control={"type": "ephemeral"},
            messages=messages,
        )
        ep.turns += 1
        ep.usage.add(resp.usage)
        messages.append({"role": "assistant", "content": resp.content})
        text = " ".join(b.text for b in resp.content if b.type == "text").strip()
        thought = " ".join(b.thinking for b in resp.content if b.type == "thinking" and b.thinking).strip()
        calls = [b for b in resp.content if b.type == "tool_use"]
        event = {"turn": ep.turns, "stop_reason": resp.stop_reason, "reasoning": text, "thinking_summary": thought,
                 "api_s": round(time.time() - t_call, 1),
                 "usage": resp.usage.model_dump(exclude_none=True)}
        if resp.stop_reason == "refusal":
            log.event({**event, "ended": "refusal"})
            ep.ended = "refusal"
            break
        if not calls:
            log.event({**event, "ended": "done" if resp.stop_reason == "end_turn" else resp.stop_reason})
            ep.ended = "done" if resp.stop_reason == "end_turn" else str(resp.stop_reason)
            break
        call = calls[0]
        actions = call.input.get("actions", []) if isinstance(call.input, dict) else []
        done = arm.execute(actions)
        ep.sim_s += done.sim_s
        obs = observe(arm)
        log.frames(ep.turns, obs)
        log.event({**event, "actions": actions, "result": done.text, "ran": done.ran, "state": obs.state,
                   "truth": sim.truth()})
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": call.id, "content": done.text},
            *observation_blocks(obs, "Current observation:"),
        ]})
    sim.step(0.5)
    ep.wall_s = time.time() - t0
    ep.result = grade(sim)
    log.frames(ep.turns + 1, observe(arm), final=True)
    return ep


class _Log:
    def __init__(self, out: Path | None):
        self.out = out
        if out:
            out.mkdir(parents=True, exist_ok=True)
            (out / "events.jsonl").write_text("")

    def event(self, e: dict) -> None:
        if self.out:
            with open(self.out / "events.jsonl", "a") as f:
                f.write(json.dumps(e, default=str) + "\n")

    def frames(self, turn: int, obs, final: bool = False) -> None:
        if self.out:
            name = "final" if final else f"t{turn:02d}"
            obs.scene.save(self.out / f"{name}-scene.jpg", quality=85)
            obs.wrist.save(self.out / f"{name}-wrist.jpg", quality=85)
