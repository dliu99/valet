"""Benchmark episodes onto the shared scoreboard (`valet_sim.scoreboard`): one line per held-out episode.

Runtime perception is the only model use while a program runs: `localize` calls answered by Claude count as slow
calls, those answered by a System One model (Jev hosted, or H2O-Lightning-4B locally) as fast calls. Offline spend
(writing, debugging and verifying) is not runtime cost and is reported in REPORT.md only."""

from __future__ import annotations

import json
from pathlib import Path

from valet_sim.scoreboard import Episode, record

from . import ledger

EPISODES = Path(__file__).resolve().parents[2] / "results" / "episodes.jsonl"
CLAUDE_USD_PER_CALL = 0.0032  # measured: Opus 5.5, effort low, ~650 input and ~30 output tokens per frame
USD_PER_CALL = {"claude": CLAUDE_USD_PER_CALL, "jev": ledger.JEV_USD_PER_CALL, "h2o": 0.0}


def record_results(results: list[dict], backend: str, config: str, path: Path = EPISODES) -> None:
    for r in results:
        n = r["localize_calls"]
        record(path, Episode(
            experiment="aspire", config=config, seed=int(r["seed"]), placed=bool(r["score"]["placed"]),
            lifted=bool(r["score"]["lifted"]), seconds=float(r["sim_s"]),
            slow_calls=n if backend == "claude" else 0, fast_calls=0 if backend == "claude" else n,
            usd=round(n * USD_PER_CALL[backend], 4), calls={f"localize:{backend}": n}))


def backfill(run_dir: Path, backend: str, config: str) -> int:
    """Record an evaluation that already ran, from its saved traces."""
    results = [json.loads(p.read_text()) for p in sorted(run_dir.glob("seed_*/trace.json"))]
    record_results(results, backend, config)
    return len(results)
