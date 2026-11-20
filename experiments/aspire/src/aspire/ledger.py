"""Every paid call, appended to one JSONL file shared by all processes, and the budget that stops the run.

Prices are per million tokens (Anthropic first-party, cached 2026-11-03 in the claude-api skill). Jev calls are
counted, with a per-call price set by JEV_USD_PER_CALL."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

PRICES = {  # input, output, cache read, cache write (5 min)
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10, 2.50),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
}
JEV_USD_PER_CALL = float(os.environ.get("JEV_USD_PER_CALL", "0.002"))
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "state" / "ledger.jsonl"


class BudgetExceeded(RuntimeError):
    pass


def ledger_path() -> Path:
    return Path(os.environ.get("ASPIRE_LEDGER", DEFAULT_PATH))


def budget_usd() -> float:
    return float(os.environ.get("ASPIRE_BUDGET_USD", "40"))


def usd_for(model: str, usage) -> float:
    p_in, p_out, p_read, p_write = PRICES.get(model, PRICES["claude-opus-5-5"])
    read = getattr(usage, "cache_read_input_tokens", 0) or 0
    write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    return (usage.input_tokens * p_in + usage.output_tokens * p_out + read * p_read + write * p_write) / 1e6


def record(role: str, model: str, usd: float, **tokens) -> None:
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"t": round(time.time(), 1), "role": role, "model": model, "usd": round(usd, 6), **tokens})
    with open(path, "a") as f:
        f.write(line + "\n")


def record_claude(role: str, model: str, usage) -> float:
    usd = usd_for(model, usage)
    record(role, model, usd, input=usage.input_tokens, output=usage.output_tokens,
           cache_read=getattr(usage, "cache_read_input_tokens", 0) or 0,
           cache_write=getattr(usage, "cache_creation_input_tokens", 0) or 0)
    return usd


def entries() -> list[dict]:
    path = ledger_path()
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def spent() -> float:
    return sum(e["usd"] for e in entries())


def check(extra_usd: float = 0.0) -> None:
    """Raise before a call that would take total spend past the budget."""
    total = spent()
    if total + extra_usd > budget_usd():
        raise BudgetExceeded(f"spent ${total:.2f} of ${budget_usd():.2f}")


def summary() -> dict:
    out: dict[str, dict] = {}
    for e in entries():
        r = out.setdefault(e["role"], {"calls": 0, "usd": 0.0, "input": 0, "output": 0, "cache_read": 0, "cache_write": 0})
        r["calls"] += 1
        r["usd"] += e["usd"]
        for k in ("input", "output", "cache_read", "cache_write"):
            r[k] += e.get(k, 0)
    return out


def load_env(path: Path | None = None) -> None:
    """Read KEY=VALUE lines from valet's .env into the environment, without printing them."""
    path = path or Path(os.environ.get("VALET_ENV", "/Users/devin/projects/valet/.env"))
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().removeprefix("export ").strip()
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))
