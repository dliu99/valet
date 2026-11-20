"""The robot execution engine: run a program on seeded scenes, trace every primitive call, grade with task.score.

A program is Python source defining `run(robot)`. It may import only numpy, math and a few standard modules, and
may not touch names that start with an underscore, so it cannot reach the simulator behind `robot`. Grading
(`valet_sim.task.score` on the true box pose) happens after the program returns and is never visible to it."""

from __future__ import annotations

import ast
import builtins
import json
import multiprocessing as mp
import os
import signal
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from . import ledger, tasks
from .robot import EpisodeLimit, Limits, Recorder, Robot

ALLOWED_IMPORTS = {"numpy", "math", "itertools", "functools", "collections", "statistics", "dataclasses", "typing"}
BANNED_NAMES = {"eval", "exec", "compile", "open", "input", "globals", "locals", "vars", "getattr", "setattr",
                "delattr", "__import__", "breakpoint", "memoryview", "help", "exit", "quit"}
BANNED_SUBSTRINGS = ("truth", "valet_sim", "mujoco", "lifted_m", ".load(", ".save(", "tofile", "fromfile")
SETTLE_AFTER_S = 1.0
WALL_LIMIT_S = int(os.environ.get("ASPIRE_WALL_LIMIT_S", "300"))


class ProgramRejected(ValueError):
    pass


def check_program(code: str) -> None:
    """Static rules: parses, defines run(robot), imports only allowed modules, no private or banned names."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ProgramRejected(f"syntax error: {e}") from e
    if not any(isinstance(n, ast.FunctionDef) and n.name == "run" for n in tree.body):
        raise ProgramRejected("the program must define a top-level function run(robot)")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise ProgramRejected(f"import of {a.name} is not allowed (allowed: {sorted(ALLOWED_IMPORTS)})")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                raise ProgramRejected(f"import from {node.module} is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ProgramRejected(f"attribute {node.attr!r}: names starting with an underscore are off limits")
        elif isinstance(node, ast.Name) and (node.id in BANNED_NAMES or node.id.startswith("__")):
            raise ProgramRejected(f"name {node.id!r} is not allowed")
    for s in BANNED_SUBSTRINGS:
        if s in code:
            raise ProgramRejected(f"{s!r} is not allowed in a program")


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".")[0] not in ALLOWED_IMPORTS:
        raise ImportError(f"import of {name} is not allowed")
    return builtins.__import__(name, globals, locals, fromlist, level)


SAFE_BUILTINS = {k: getattr(builtins, k) for k in (
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "int", "isinstance", "len", "list",
    "map", "max", "min", "pow", "range", "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip", "print",
    "Exception", "ValueError", "RuntimeError", "KeyError", "IndexError", "ZeroDivisionError", "TypeError",
    "ArithmeticError", "StopIteration", "AssertionError", "NotImplementedError", "object", "hasattr", "frozenset",
    "iter", "next", "repr", "type", "callable", "slice", "format", "hash", "id", "chr", "ord")}


@dataclass(frozen=True)
class EpisodeSpec:
    code: str
    seed: int
    task: str = "benchmark"
    out: str | None = None
    backend: str = "claude"
    keyframes: bool = True


def run_episode(spec: EpisodeSpec) -> dict:
    """One program on one scene. Never raises: errors end up in the result."""
    from valet_sim.sim import Sim, SimSpec, with_box
    from valet_sim.task import score

    t0 = time.time()
    out = Path(spec.out) if spec.out else None
    rec = Recorder(out, keyframes=spec.keyframes)
    ep = tasks.make(spec.task, spec.seed)
    sim = Sim(with_box(SimSpec(), ep.box, ep.target_xy))
    error = None
    try:
        check_program(spec.code)
        robot = Robot(sim, rec, backend=spec.backend, limits=Limits())
        env = {"__builtins__": {**SAFE_BUILTINS, "__import__": _safe_import, "__build_class__": builtins.__build_class__},
               "__name__": "program"}
        printed: list[str] = []
        env["__builtins__"]["print"] = lambda *a, **k: robot.log(*a)
        exec(compile(spec.code, "<program>", "exec"), env)
        _alarm(WALL_LIMIT_S)
        try:
            env["run"](robot)
        finally:
            _alarm(0)
    except ProgramRejected as e:
        error = f"rejected: {e}"
    except EpisodeLimit as e:
        error = f"limit: {e}"
    except _WallClock:
        error = f"limit: more than {WALL_LIMIT_S} s of wall time"
    except ledger.BudgetExceeded as e:
        error = f"budget: {e}"
    except Exception as e:
        error = "".join(traceback.format_exception_only(type(e), e)).strip()
        tb = [f for f in traceback.extract_tb(e.__traceback__) if f.filename == "<program>"]
        if tb:
            error += " (program line " + ", ".join(str(f.lineno) for f in tb) + ")"
    sim.step(SETTLE_AFTER_S)
    grade = score(sim.truth())
    result = {"seed": spec.seed, "task": spec.task, "score": grade, "error": error, "calls": rec.calls, "log": rec.log,
              "sim_s": round(float(sim.data.time), 1), "wall_s": round(time.time() - t0, 1),
              "localize_calls": sum(1 for c in rec.calls if c["prim"] == "localize"),
              "localize_paid": sum(1 for c in rec.calls if c["prim"] == "localize" and not c.get("cached"))}
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        rec.save(sim.frame("side"), "final_side.jpg")
        (out / "trace.json").write_text(json.dumps(result, indent=1))
    return result


class _WallClock(BaseException):
    pass


def _on_alarm(signum, frame):
    raise _WallClock()


def _alarm(seconds: int) -> None:
    try:
        signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(seconds)
    except ValueError:  # not the main thread
        pass


_POOL: ProcessPoolExecutor | None = None


def pool(workers: int = int(os.environ.get("ASPIRE_WORKERS", "10"))) -> ProcessPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=ledger.load_env)
    return _POOL


def evaluate(code: str, seeds, task: str = "benchmark", out: Path | None = None, backend: str = "claude",
             keyframes: bool = True) -> dict:
    """The program on every seed, in parallel. Returns per-seed results and totals."""
    specs = [EpisodeSpec(code, s, task, str(out / f"seed_{s}") if out else None, backend, keyframes) for s in seeds]
    results = list(pool().map(run_episode, specs))
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "program.py").write_text(code)
    return totals(results)


def totals(results: list[dict]) -> dict:
    placed = sum(r["score"]["placed"] for r in results)
    lifted = sum(r["score"]["lifted"] for r in results)
    return {"placed": placed, "lifted": lifted, "n": len(results), "fitness": fitness(results), "results": results,
            "median_sim_s": sorted(r["sim_s"] for r in results)[len(results) // 2] if results else 0.0,
            "localize_calls": sum(r["localize_calls"] for r in results),
            "localize_paid": sum(r.get("localize_paid", 0) for r in results)}


def fitness(results: list[dict]) -> float:
    """Placed counts 1, lifted 0.3, and closeness to the target breaks ties among the unplaced."""
    f = 0.0
    for r in results:
        s = r["score"]
        f += 1.0 if s["placed"] else (0.3 if s["lifted"] else 0.0) + 0.1 * max(0.0, 1 - s["off_target_m"] / 0.15) * s["lifted"]
    return round(f, 3)


def outcome(r: dict) -> str:
    s = r["score"]
    return "PLACED" if s["placed"] else ("lifted, not placed" if s["lifted"] else "not lifted")


def summarize(r: dict, max_calls: int = 40) -> str:
    """One episode as text for an agent: outcome, then each primitive call with its status and result."""
    s = r["score"]
    lines = [f"seed {r['seed']}: {outcome(r)} | off_target {s['off_target_m']} m, tilt {s['tilt_deg']} deg, "
             f"resting {s['resting']}, released {s['released']} | {len(r['calls'])} calls, {r['sim_s']} s"]
    calls = r["calls"]
    shown = calls if len(calls) <= max_calls else calls[:max_calls // 2] + [None] + calls[-max_calls // 2:]
    for c in shown:
        if c is None:
            lines.append(f"  ... {len(calls) - max_calls} calls omitted ...")
            continue
        lines.append("  " + _call_line(c))
    for l in r["log"][-12:]:
        lines.append(f"  log {l}")
    if r["error"]:
        lines.append(f"  ERROR {r['error']}")
    return "\n".join(lines)


def _call_line(c: dict) -> str:
    a, res = c["args"], c.get("result", {})
    if c["prim"] == "move_to_pose":
        head = f"#{c['i']} move_to_pose({a['xyz']}, pitch={a['pitch_deg']:g}, roll={a['roll_deg']:g})"
        if not res:
            return f"{head} -> {c['status']} {c.get('error', '')}"
        tail = f"-> {res['status']}"
        if res["status"] != "unreachable":
            tail += f" err {res['error_m'] * 1000:.0f} mm, at {res['grasp_point']} pitch {res['pitch_deg']}"
        else:
            tail += f" ({res.get('message', '')})"
        return f"{head} {tail} gripper {c.get('state_after', c['state_before'])['joints']['gripper']:.1f}"
    if c["prim"] == "set_gripper":
        head = f"#{c['i']} set_gripper({a['pct']:g})"
        return f"{head} -> {res.get('status', c['status'])} reading {res.get('gripper')} amps {res.get('amps')}" if res else f"{head} -> {c['status']} {c.get('error', '')}"
    if c["prim"] == "localize":
        head = f"#{c['i']} localize({a['text']!r}, {a['camera']})"
        if not res:
            return f"{head} -> {c['status']} {c.get('error', '')}"
        if not res["found"]:
            return f"{head} -> NOT found"
        return f"{head} -> bbox {res['bbox']} centre {res['center_px']} table_xy {res['table_xy']} (camera at grasp point {c['state_before']['grasp_point']})"
    if c["prim"] == "observe":
        sb = c["state_before"]
        return f"#{c['i']} observe() -> grasp_point {sb['grasp_point']} pitch {sb['pitch_deg']} gripper {sb['joints']['gripper']:.1f}"
    return f"#{c['i']} {c['prim']}({a}) -> {c['status']}"


def summarize_all(ev: dict, only_failures: bool = False, max_calls: int = 40) -> str:
    head = f"placed {ev['placed']}/{ev['n']}, lifted {ev['lifted']}/{ev['n']}, fitness {ev['fitness']}"
    rs = [r for r in ev["results"] if not (only_failures and r["score"]["placed"])]
    return head + "\n\n" + "\n\n".join(summarize(r, max_calls) for r in rs)
