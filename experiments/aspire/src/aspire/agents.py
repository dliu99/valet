"""Coding agents on the Claude API: an actor that writes and debugs a program from traces, a skill writer that turns a
fix into a library entry, and a verifier (a fresh actor) that must reproduce the fix from the entry alone."""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import engine, ledger, tasks

MODEL = os.environ.get("ASPIRE_ACTOR_MODEL", "claude-opus-5-5")
ACTOR_EFFORT = os.environ.get("ASPIRE_ACTOR_EFFORT", "medium")
API_MD = (Path(__file__).resolve().parents[2] / "api.md").read_text()
MAX_IMAGES_PER_VIEW = 6

SYSTEM = f"""You are a robotics programmer. You write a Python program that makes a simulated SO-101 arm pick up a \
white box and set it down on a blue square, using only the primitive API below. You improve it by running it on \
debug scenes and reading the traces it leaves: every primitive call's inputs, outputs and status, and keyframes from \
the wrist and side cameras.

<api>
{API_MD}
</api>

Your tools:
- run_program runs a program on some of the debug scenes and returns, for each scene, the outcome and the trace of \
primitive calls. It also returns a run id.
- view_keyframes shows the camera frames of a run (or of a program you were given) after chosen calls, with \
overlays: the wrist camera (what the program sees) or the observer side camera (for your debugging only; the \
program cannot see it).
- submit hands in your final program and ends your session.

The program will be graded later on scenes you never see: other box poses, sizes and target positions. So never \
branch on the seed or on anything that identifies a debug scene, and never hardcode a position read from a debug \
trace. Prefer measurements (localize, camera geometry, joint readings, currents) over constants fitted to the debug \
scenes. When a scene fails, find the call where it went wrong (from the trace and the keyframes) before you change \
code, and say what you found."""

TOOLS = [
    {
        "name": "run_program",
        "description": "Run a program (Python source defining run(robot)) on debug scenes. Returns the run id, the "
                       "score per scene, and the primitive-call trace of each scene (failures in full; placed scenes "
                       "shortened). Running all scenes takes about a minute.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The whole program."},
                "seeds": {"type": "array", "items": {"type": "integer"},
                          "description": "Which debug scenes to run; omit or leave empty for all of them."},
            },
            "required": ["code"],
        },
    },
    {
        "name": "view_keyframes",
        "description": "Show camera frames recorded during a run. Call 0 is the start; for move_to_pose, set_gripper "
                       "and observe calls you get the frame right after the call; for localize, the frame it looked "
                       "at with its bounding box. Set final to also get the side view after grading. At most 6 images.",
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "seed": {"type": "integer"},
                "calls": {"type": "array", "items": {"type": "integer"}},
                "camera": {"type": "string", "enum": ["wrist", "side", "both"]},
                "final": {"type": "boolean"},
            },
            "required": ["run_id", "seed", "calls", "camera"],
        },
    },
    {
        "name": "submit",
        "description": "Hand in the final program. Ends the session.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The whole program."},
                "notes": {"type": "string", "description": "What you changed and why, citing the failures it fixes."},
            },
            "required": ["code", "notes"],
        },
    },
]


@dataclass
class Runs:
    """Run ids an agent can look at: its own runs and the evaluations it was shown."""

    root: Path
    paths: dict[str, Path] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class Session:
    name: str
    role: str
    user: list
    workdir: Path
    runs: Runs
    task: str = "benchmark"
    max_tool_calls: int = 14
    effort: str = ACTOR_EFFORT
    debug_seeds: tuple = ()
    log: list = field(default_factory=list)

    def run(self) -> dict:
        import anthropic

        client = anthropic.Anthropic(max_retries=4)
        seeds = self.debug_seeds or tasks.DEBUG_SEEDS[self.task]
        budget = {"type": "text", "text": f"You have {self.max_tool_calls} tool calls in this session, submit included."}
        messages = [{"role": "user", "content": [*self.user, budget]}]
        submitted, last_code, notes = None, None, ""
        usd, tool_calls, turns = 0.0, 0, 0
        self.workdir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        while turns < self.max_tool_calls + 4:
            turns += 1
            ledger.check(1.0)
            with client.messages.stream(
                model=MODEL, max_tokens=48000, system=SYSTEM, tools=TOOLS, messages=messages,
                cache_control={"type": "ephemeral"}, output_config={"effort": self.effort},
            ) as stream:
                r = stream.get_final_message()
            usd += ledger.record_claude(self.role, MODEL, r.usage)
            messages.append({"role": "assistant", "content": r.content})
            texts = [b.text for b in r.content if b.type == "text"]
            self.log.append({"turn": turns, "stop": r.stop_reason, "text": texts,
                             "tools": [{"name": b.name, "input": {k: v for k, v in b.input.items() if k != "code"}}
                                       for b in r.content if b.type == "tool_use"],
                             "usage": r.usage.model_dump() if hasattr(r.usage, "model_dump") else str(r.usage)})
            if r.stop_reason == "refusal":
                break
            uses = [b for b in r.content if b.type == "tool_use"]
            if not uses:
                messages.append({"role": "user", "content": "Call submit with your final program when you are done."})
                if r.stop_reason == "end_turn" and turns > self.max_tool_calls:
                    break
                continue
            results = []
            for b in uses:
                tool_calls += 1
                if b.name == "submit":
                    submitted, notes = b.input.get("code", ""), b.input.get("notes", "")
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": "Submitted."})
                    continue
                try:
                    if b.name == "run_program":
                        last_code = b.input["code"]
                        content = self._run_program(b.input["code"], b.input.get("seeds") or list(seeds), seeds)
                    elif b.name == "view_keyframes":
                        content = self._view(b.input)
                    else:
                        content = f"unknown tool {b.name}"
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": content})
                except ledger.BudgetExceeded:
                    raise
                except Exception as e:
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": f"{type(e).__name__}: {e}", "is_error": True})
            left = self.max_tool_calls - tool_calls
            if submitted is not None:
                messages.append({"role": "user", "content": results})
                break
            if left <= 2:
                results.append({"type": "text", "text": f"You have {max(left, 0)} tool calls left. Submit your best program now."})
            messages.append({"role": "user", "content": results})
            if left < 0:
                break
        code = submitted if submitted is not None else last_code
        out = {"name": self.name, "role": self.role, "code": code, "notes": notes, "submitted": submitted is not None,
               "usd": round(usd, 4), "turns": turns, "tool_calls": tool_calls, "wall_s": round(time.time() - t0)}
        (self.workdir / "session.json").write_text(json.dumps({**out, "log": self.log}, indent=1, default=str))
        if code:
            (self.workdir / "submitted.py").write_text(code)
        return out

    def _run_program(self, code: str, seeds: list[int], allowed) -> str:
        bad = [s for s in seeds if s not in allowed]
        if bad:
            return f"seeds {bad} are not debug scenes; the debug scenes are {list(allowed)}"
        with self.runs.lock:
            run_id = f"{self.name}-run{sum(1 for k in self.runs.paths if k.startswith(self.name + '-run')) + 1}"
            path = self.workdir / run_id
            self.runs.paths[run_id] = path
        try:
            engine.check_program(code)
        except engine.ProgramRejected as e:
            return f"run {run_id}: program rejected before running: {e}"
        ev = engine.evaluate(code, seeds, self.task, path)
        return f"run id {run_id}\n" + render_eval(ev)

    def _view(self, a: dict) -> list:
        path = self.runs.paths.get(a["run_id"])
        if path is None:
            return [{"type": "text", "text": f"unknown run id {a['run_id']}; known: {sorted(self.runs.paths)}"}]
        d = path / f"seed_{a['seed']}"
        if not d.exists():
            return [{"type": "text", "text": f"run {a['run_id']} has no scene {a['seed']}"}]
        cams = ["wrist", "side"] if a["camera"] == "both" else [a["camera"]]
        files = []
        for i in a["calls"]:
            if i == 0:
                files += [d / f"c00_before_{c}.jpg" for c in cams]
            elif (d / f"c{i:02d}_seen.jpg").exists():
                files.append(d / f"c{i:02d}_seen.jpg")
            else:
                files += [d / f"c{i:02d}_after_{c}.jpg" for c in cams]
        if a.get("final"):
            files.append(d / "final_side.jpg")
        out: list = []
        for f in files[:MAX_IMAGES_PER_VIEW]:
            if not f.exists():
                out.append({"type": "text", "text": f"no frame {f.name}"})
                continue
            out.append({"type": "text", "text": f.name})
            out.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": base64.b64encode(f.read_bytes()).decode()}})
        if len(files) > MAX_IMAGES_PER_VIEW:
            out.append({"type": "text", "text": f"{len(files) - MAX_IMAGES_PER_VIEW} more frames not shown"})
        return out


def render_eval(ev: dict, max_calls: int = 40) -> str:
    """Scores, then full traces of the failures and one line per placed scene."""
    lines = [f"placed {ev['placed']}/{ev['n']}, lifted {ev['lifted']}/{ev['n']}"]
    for r in ev["results"]:
        if r["score"]["placed"]:
            lines.append(f"seed {r['seed']}: PLACED (off_target {r['score']['off_target_m']} m, {len(r['calls'])} calls)")
    for r in ev["results"]:
        if not r["score"]["placed"]:
            lines.append("")
            lines.append(engine.summarize(r, max_calls))
    return "\n".join(lines)


def call_text(model: str, role: str, system: str, user: str, effort: str = "high", max_tokens: int = 32000) -> str:
    import anthropic

    ledger.check(0.5)
    client = anthropic.Anthropic(max_retries=4)
    with client.messages.stream(model=model, max_tokens=max_tokens, system=system,
                                messages=[{"role": "user", "content": user}], output_config={"effort": effort}) as s:
        r = s.get_final_message()
    ledger.record_claude(role, model, r.usage)
    return "".join(b.text for b in r.content if b.type == "text")
