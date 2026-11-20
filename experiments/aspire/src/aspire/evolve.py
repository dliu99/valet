"""The ASPIRE loop: K actors per round write or repair programs on the debug scenes, conditioned on the best programs
so far and their failure traces; fixes become skill-library entries only after a fresh verifier reproduces them from
the entry alone."""

from __future__ import annotations

import difflib
import json
import math
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import agents, engine, ledger, tasks

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "state"
SKILLS = ROOT / "skills"


@dataclass
class Candidate:
    id: str
    round: int
    parent: str | None
    origin: str
    placed: int
    lifted: int
    fitness: float
    per_seed: dict
    notes: str = ""
    usd: float = 0.0
    skill: str | None = None


@dataclass
class Evolution:
    name: str
    task: str = "benchmark"
    k: int = 3
    rounds: int = 5
    n_parents: int = 2
    use_skills: bool = True
    write_skills: bool = True
    skills_per_round: int = 3
    actor_tool_calls: int = 10
    verifier_tool_calls: int = 6
    archive: list[Candidate] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.dir = STATE / self.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.runs = agents.Runs(self.dir)
        self.lock = threading.Lock()
        self.seeds = tasks.DEBUG_SEEDS[self.task]
        if (self.dir / "archive.json").exists():
            d = json.loads((self.dir / "archive.json").read_text())
            self.archive = [Candidate(**c) for c in d["archive"]]
            self.events = d.get("events", [])
            for c in self.archive:
                self.runs.paths[c.id] = self.dir / "cands" / c.id

    # ---- bookkeeping -------------------------------------------------------------------------------------------

    def save(self) -> None:
        (self.dir / "archive.json").write_text(json.dumps(
            {"archive": [asdict(c) for c in self.archive], "events": self.events}, indent=1))

    def event(self, kind: str, **kw) -> None:
        e = {"t": round(time.time()), "kind": kind, "spent_usd": round(ledger.spent(), 2), **kw}
        print(json.dumps(e), flush=True)
        with self.lock:
            self.events.append(e)
            self.save()

    def code(self, cid: str) -> str:
        return (self.dir / "cands" / cid / "program.py").read_text()

    def ranked(self) -> list[Candidate]:
        return sorted(self.archive, key=lambda c: (-c.fitness, c.round, c.id))

    def eval_text(self, cid: str) -> str:
        results = [json.loads((self.dir / "cands" / cid / f"seed_{s}" / "trace.json").read_text()) for s in self.seeds]
        return agents.render_eval(engine.totals(results))

    def admit(self, cid: str, rnd: int, parent: str | None, origin: str, code: str, notes: str, usd: float,
              skill: str | None = None) -> Candidate:
        ev = engine.evaluate(code, self.seeds, self.task, self.dir / "cands" / cid)
        per_seed = {str(r["seed"]): {"placed": r["score"]["placed"], "lifted": r["score"]["lifted"],
                                     "off_target_m": r["score"]["off_target_m"], "error": r["error"],
                                     "localize_calls": r["localize_calls"]} for r in ev["results"]}
        c = Candidate(cid, rnd, parent, origin, ev["placed"], ev["lifted"], ev["fitness"], per_seed, notes, usd, skill)
        with self.lock:
            self.runs.paths[cid] = self.dir / "cands" / cid
            self.archive.append(c)
        self.event("candidate", id=cid, parent=parent, origin=origin, placed=c.placed, lifted=c.lifted,
                   fitness=c.fitness, usd=usd)
        return c

    # ---- skills ------------------------------------------------------------------------------------------------

    def library(self) -> list[Path]:
        return sorted(SKILLS.glob("*.md"))

    def library_text(self) -> str:
        files = self.library()
        if not self.use_skills or not files:
            return "<skill_library>\nThe skill library is empty.\n</skill_library>"
        body = "\n\n".join(_strip_front(f.read_text()) for f in files)
        return ("<skill_library>\nRepairs that earlier programs needed, each verified by reproducing it from the entry "
                f"alone. Use the ones whose failure signature you see.\n\n{body}\n</skill_library>")

    # ---- one round ---------------------------------------------------------------------------------------------

    def run(self) -> Candidate:
        start = max((c.round for c in self.archive), default=-1) + 1
        ended = {e["round"] for e in self.events if e["kind"] == "round_end"}
        if start > 0 and start - 1 not in ended:  # a round cut off part way: run its missing actors
            rnd = start - 1
            have = {c.id for c in self.archive}
            jobs = [j for j in self.plan(rnd, from_scratch=rnd == 0) if f"r{rnd}a{j['k']}" not in have]
            self.event("round_resume", round=rnd, missing=[f"r{rnd}a{j['k']}" for j in jobs])
            for j in jobs:
                shutil.rmtree(self.dir / "sessions" / f"r{rnd}a{j['k']}", ignore_errors=True)
            with ThreadPoolExecutor(max(1, len(jobs))) as pool:
                list(pool.map(self.actor_job, jobs))
            best = self.ranked()[0]
            self.event("round_end", round=rnd, best=best.id, placed=best.placed, lifted=best.lifted, fitness=best.fitness)
        done = {e["round"] for e in self.events if e["kind"] == "skills_done"}
        for rnd in range(start):
            if self.write_skills and rnd not in done:
                self.skills_from(rnd, [c for c in self.archive if c.round == rnd and c.origin == "actor"])
                self.event("skills_done", round=rnd)
        for rnd in range(start, self.rounds):
            best = self.ranked()[0] if self.archive else None
            if best and best.placed == len(self.seeds):
                break
            self.event("round_start", round=rnd)
            jobs = self.plan(rnd)
            with ThreadPoolExecutor(len(jobs)) as pool:
                outs = list(pool.map(self.actor_job, jobs))
            children = [c for c in outs if c is not None]
            if self.write_skills:
                self.skills_from(rnd, children)
                self.event("skills_done", round=rnd)
            best = self.ranked()[0]
            self.event("round_end", round=rnd, best=best.id, placed=best.placed, lifted=best.lifted, fitness=best.fitness)
        return self.ranked()[0]

    def plan(self, rnd: int, from_scratch: bool = False) -> list[dict]:
        if from_scratch or not self.archive:
            return [{"round": rnd, "k": k, "parent": None, "other": None} for k in range(self.k)]
        top = self.ranked()[: self.n_parents]
        jobs = []
        for k in range(self.k):
            parent = top[k % len(top)]
            other = top[(k + 1) % len(top)] if len(top) > 1 else None
            fails = [int(s) for s, v in parent.per_seed.items() if not v["placed"]]
            focus = fails[k % len(fails):] + fails[: k % len(fails)] if fails else []
            jobs.append({"round": rnd, "k": k, "parent": parent.id, "other": other.id if other else None, "focus": focus[:3]})
        return jobs

    def actor_job(self, job: dict) -> Candidate | None:
        cid = f"r{job['round']}a{job['k']}"
        parts = [self.library_text()]
        if job["parent"] is None:
            parts.append("Write the program from scratch. Start simple, run it, and fix what the traces show. Run it on "
                         "all debug scenes before you submit.")
        else:
            p = job["parent"]
            parts.append(f"<program id=\"{p}\">\n{self.code(p)}\n</program>\n\nIts result on the debug scenes (run id "
                         f"\"{p}\"; view_keyframes works on it):\n{self.eval_text(p)}")
            if job.get("other"):
                o = next(c for c in self.archive if c.id == job["other"])
                parts.append(f"Another program from the archive (placed {o.placed}/{len(self.seeds)}), for ideas; run id "
                             f"\"{o.id}\":\n<program id=\"{o.id}\">\n{self.code(o.id)}\n</program>")
            focus = ", ".join(str(s) for s in job.get("focus", []))
            parts.append("Improve the first program so it places the box on more scenes without losing the ones it "
                         f"already places. Start with seed{'s' if ',' in focus else ''} {focus or 'of your choice'}. "
                         "Run it on all debug scenes before you submit.")
        session = agents.Session(cid, "actor", [{"type": "text", "text": "\n\n".join(parts)}], self.dir / "sessions" / cid,
                                 self.runs, self.task, self.actor_tool_calls)
        try:
            out = session.run()
        except ledger.BudgetExceeded as e:
            self.event("budget", id=cid, message=str(e))
            return None
        if not out["code"]:
            self.event("no_program", id=cid)
            return None
        return self.admit(cid, job["round"], job["parent"], "actor", out["code"], out["notes"], out["usd"])

    def skills_from(self, rnd: int, children: list[Candidate]) -> None:
        """Repairs this round produced: a child that beats its parent across rounds, and, inside each actor's session,
        a run that fixes scenes the previous run (on the same scenes) failed. The best few go to the skill writer."""
        fixes = []
        by_id = {c.id: c for c in self.archive}
        for c in children:
            if c.parent in by_id:
                fixes.append(self.fix(c.parent, self.dir / "cands" / c.parent, c.id, self.dir / "cands" / c.id, c.id))
        for sdir in sorted((self.dir / "sessions").glob(f"r{rnd}a*")):
            runs = sorted(sdir.glob(f"{sdir.name}-run*"), key=lambda p: int(p.name.rsplit("run", 1)[1]))
            for a, b in zip(runs, runs[1:]):
                fixes.append(self.fix(a.name, a, b.name, b, sdir.name))
        fixes = sorted((f for f in fixes if f), key=lambda f: -(len(f["fixed"]) - len(f["broke"])))
        chosen, per_source = [], {}
        for f in fixes:
            if per_source.get(f["source"], 0) < 2 and len(chosen) < self.skills_per_round:
                chosen.append(f)
                per_source[f["source"]] = per_source.get(f["source"], 0) + 1
        self.event("fixes_found", round=rnd, n=len(fixes),
                   chosen=[{k: f[k] for k in ("before", "after", "fixed", "broke")} for f in chosen])
        for i, f in enumerate(chosen):
            try:
                self.extract_and_verify(rnd, i, f)
            except ledger.BudgetExceeded as e:
                self.event("budget", message=str(e))
                return

    def fix(self, before: str, before_dir: Path, after: str, after_dir: Path, source: str) -> dict | None:
        rb, ra = _results(before_dir), _results(after_dir)
        common = sorted(set(rb) & set(ra))
        if not common:
            return None
        fixed, broke = {}, []
        for s in common:
            b, a = rb[s]["score"], ra[s]["score"]
            if a["placed"] and not b["placed"]:
                fixed[str(s)] = "placed"
            elif a["lifted"] and not b["lifted"] and not b["placed"]:
                fixed[str(s)] = "lifted"
            elif (b["placed"] and not a["placed"]) or (b["lifted"] and not a["lifted"]):
                broke.append(s)
        if not fixed or len(fixed) <= len(broke):
            return None
        if engine.fitness([ra[s] for s in common]) <= engine.fitness([rb[s] for s in common]):
            return None
        return {"before": before, "before_dir": before_dir, "after": after, "after_dir": after_dir, "fixed": fixed,
                "broke": broke, "source": source, "seeds": common}

    def extract_and_verify(self, rnd: int, i: int, f: dict) -> None:
        before, after = (f["before_dir"] / "program.py").read_text(), (f["after_dir"] / "program.py").read_text()
        fixed = f["fixed"]
        seeds = sorted(fixed, key=int)
        tr = lambda d: "\n\n".join(engine.summarize(json.loads((d / f"seed_{s}" / "trace.json").read_text())) for s in seeds)
        diff = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), "before.py", "after.py"))
        existing = "\n\n".join(_strip_front(p.read_text()) for p in self.library()) or "(empty)"
        user = f"""A program failed on some debug scenes; a later version of it fixed them. Write a skill-library entry \
that would let another programmer, who sees only the failing program, its traces and your entry, make the same repair.

<existing_library>
{existing}
</existing_library>

<before>
{before}
</before>

<before_traces>
{tr(f["before_dir"])}
</before_traces>

<after>
{after}
</after>

<after_traces>
{tr(f["after_dir"])}
</after_traces>

<diff>
{diff}
</diff>

Pick the one change in the diff that did the most to fix these scenes. If it is already covered by an entry in the \
library, or it only retunes a constant to fit the debug scenes, reply with exactly NONE. Otherwise reply with one \
entry in this format, under 350 words, with no seed numbers or scene-specific positions in it:

# <short title>
## Failure signature
<what the trace shows when this failure happens: which primitive, which status or reading, what the keyframes show>
## When to apply
## Repair
## Code sketch
```python
<a short sketch against the primitive API>
```"""
        system = ("You maintain a library of validated repairs for robot programs written against this primitive API."
                  f"\n\n<api>\n{agents.API_MD}\n</api>")
        entry = agents.call_text(agents.MODEL, "skill_writer", system, user).strip()
        if entry.upper().startswith("NONE") or not entry.startswith("#"):
            self.event("skill_none", before=f["before"], after=f["after"], fixed=fixed)
            return
        title = entry.splitlines()[0].lstrip("# ").strip()
        self.event("skill_candidate", before=f["before"], after=f["after"], fixed=fixed, title=title)

        before_results = _results(f["before_dir"])
        vid = f"r{rnd}v{i}"
        user_v = [{"type": "text", "text": f"""<skill>
{entry}
</skill>

<program id="{f["before"]}">
{before}
</program>

Its result on the debug scenes (run id "{f["before"]}"; view_keyframes works on it):
{agents.render_eval(engine.totals(list(before_results.values())))}

Repair this program by applying the skill above where its failure signature shows up in the traces. Run it on the \
debug scenes, then submit."""}]
        session = agents.Session(vid, "verifier", user_v, self.dir / "sessions" / vid,
                                 agents.Runs(self.dir, {f["before"]: f["before_dir"]}), self.task, self.verifier_tool_calls)
        out = session.run()
        if not out["code"]:
            self.event("skill_rejected", title=title, reason="verifier produced no program")
            return
        v = self.admit(vid, rnd, f["before"], "verifier", out["code"], out["notes"], out["usd"], skill=title)
        vres = _results(self.dir / "cands" / vid)
        reproduced = [s for s, kind in fixed.items() if v.per_seed[s][kind]]
        need = max(1, math.ceil(len(fixed) / 2))
        v_fit = engine.fitness([vres[s] for s in before_results])
        b_fit = engine.fitness(list(before_results.values()))
        ok = len(reproduced) >= need and v_fit >= b_fit
        record = {"title": title, "before": f["before"], "after": f["after"], "fixed": fixed, "verifier": vid,
                  "reproduced": reproduced, "verifier_placed": v.placed, "before_seeds": sorted(before_results),
                  "verifier_fitness_on_before_seeds": v_fit, "before_fitness": b_fit}
        if not ok:
            (self.dir / "rejected_skills").mkdir(exist_ok=True)
            (self.dir / "rejected_skills" / f"{vid}.md").write_text(_front(record) + entry + "\n")
            self.event("skill_rejected", **record)
            return
        n = len(self.library()) + 1
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:48]
        path = SKILLS / f"{n:02d}-{slug}.md"
        path.write_text(_front({"id": f"S{n:02d}", "task": self.task, "run": self.name, "round": rnd, **record}) + entry + "\n")
        self.event("skill_admitted", path=str(path.relative_to(ROOT)), **record)


def _results(d: Path) -> dict[int, dict]:
    return {int(p.parent.name.split("_")[1]): json.loads(p.read_text()) for p in sorted(d.glob("seed_*/trace.json"))}


def _front(d: dict) -> str:
    return "---\n" + "\n".join(f"{k}: {json.dumps(v)}" for k, v in d.items()) + "\n---\n\n"


def _strip_front(text: str) -> str:
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end > 0:
            return text[end + 5:].strip()
    return text.strip()
