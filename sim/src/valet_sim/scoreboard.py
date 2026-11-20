"""One score for every approach: placed and lifted on the benchmark seeds, and what it cost to get there.

Each experiment appends one JSON line per episode to `experiments/<name>/results/episodes.jsonl` with `record()`.
`python -m valet_sim.scoreboard <worktree or results dir> ...` prints one row per (experiment, config) and flags
configs that lean on a slow model: more than `SLOW_CALLS_PER_EPISODE` calls to Claude (or any model slower than
about a second) per episode."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .task import BENCHMARK_SEEDS

SLOW_CALLS_PER_EPISODE = 5


@dataclass(frozen=True)
class Episode:
    experiment: str
    config: str
    seed: int
    placed: bool
    lifted: bool
    seconds: float
    slow_calls: int
    fast_calls: int
    usd: float
    calls: dict[str, int] = field(default_factory=dict)


def record(path: Path, episode: Episode) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(episode)) + "\n")


def load(roots: list[Path]) -> list[Episode]:
    files = [f for r in roots for f in ([r] if r.is_file() else r.rglob("episodes.jsonl")) if ".venv" not in f.parts]
    rows = [json.loads(line) for f in files for line in f.read_text().splitlines() if line.strip()]
    return [Episode(**{k: v for k, v in r.items() if k in Episode.__dataclass_fields__}) for r in rows]


def table(episodes: list[Episode]) -> list[dict]:
    groups: dict[tuple[str, str], dict[int, Episode]] = {}
    for e in episodes:
        if e.seed in BENCHMARK_SEEDS:
            groups.setdefault((e.experiment, e.config), {})[e.seed] = e
    out = []
    for (experiment, config), by_seed in groups.items():
        eps = list(by_seed.values())
        slow = statistics.median(e.slow_calls for e in eps)
        out.append({"experiment": experiment, "config": config, "seeds": len(eps),
                    "placed": sum(e.placed for e in eps), "lifted": sum(e.lifted for e in eps),
                    "median_s": round(statistics.median(e.seconds for e in eps)), "slow_calls": slow,
                    "fast_calls": statistics.median(e.fast_calls for e in eps),
                    "usd": round(statistics.mean(e.usd for e in eps), 3), "scalable": slow <= SLOW_CALLS_PER_EPISODE})
    return sorted(out, key=lambda r: (-r["scalable"], -r["placed"] / max(r["seeds"], 1), -r["lifted"]))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m valet_sim.scoreboard", description=__doc__)
    p.add_argument("roots", nargs="+", type=Path)
    rows = table(load(p.parse_args(argv).roots))
    head = ("experiment", "config", "seeds", "placed", "lifted", "median_s", "slow_calls", "fast_calls", "usd", "scalable")
    widths = [max(len(h), *(len(str(r[h])) for r in rows)) if rows else len(h) for h in head]
    for line in [head, *([r[h] for h in head] for r in rows)]:
        print("  ".join(str(v).ljust(w) for v, w in zip(line, widths)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
