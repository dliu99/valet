"""The fast deciders (Jev, or H2O-Lightning-4B behind the same System One API) and the slow one (Claude), with a
ledger of calls, seconds and dollars per episode."""

from __future__ import annotations

import base64
import io
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
DECIDERS = {
    "jev": {"url": "https://api.typesafe.ai/v1/systemone", "model": "jev-latest", "key": "TYPESAFE_API_KEY", "usd_per_token": 0.042 / 1e6},
    "local": {"url": "http://127.0.0.1:8741/v1/systemone", "model": "h2oai/h2o-lightning-4b", "key": None, "usd_per_token": 0.0},
}
CLAUDE_MODEL = os.environ.get("JEV_SKILLS_CLAUDE_MODEL", "claude-haiku-5-5")
CLAUDE_USD_PER_TOKEN = {"claude-haiku-5-5": (0.10 / 1e6, 0.50 / 1e6), "claude-opus-5-5": (4.0 / 1e6, 20.0 / 1e6),
                        "claude-sonnet-5-5": (2.0 / 1e6, 10.0 / 1e6)}


def load_env() -> None:
    for d in [HERE, *HERE.parents]:
        f = d / ".env"
        if f.is_file():
            for line in f.read_text().splitlines():
                k, sep, v = line.partition("=")
                if sep and k.strip() and not k.lstrip().startswith("#"):
                    os.environ.setdefault(k.strip().removeprefix("export ").strip(), v.strip().strip('"').strip("'"))
            return


def jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, "JPEG", quality=quality)
    return buf.getvalue()


@dataclass
class Ledger:
    fast_calls: int = 0
    fast_s: float = 0.0
    fast_usd: float = 0.0
    slow_calls: int = 0
    slow_s: float = 0.0
    slow_usd: float = 0.0
    by_use: dict[str, int] = field(default_factory=dict)

    def note(self, use: str) -> None:
        self.by_use[use] = self.by_use.get(use, 0) + 1


class SystemOne:
    """Typed questions about one frame and some state; one HTTP request answers them all."""

    def __init__(self, ledger: Ledger, decider: str = "jev", timeout_s: float = 120.0):
        import httpx

        self.ledger = ledger
        self.name = decider
        self.d = DECIDERS[decider]
        headers = {"TypeSafe-Danger-Zone-Images": "true"}
        if self.d["key"]:
            headers["Authorization"] = f"Bearer {os.environ[self.d['key']]}"
        self.http = httpx.Client(timeout=timeout_s, headers=headers)

    def ask(self, state: dict, frame: np.ndarray, questions: dict, use: str) -> dict:
        import httpx

        data = jpeg(frame)
        ref = f"wrist-{abs(hash(data)) % 10**12}"
        body = {"model": self.d["model"], "state": {**state, "wrist": ref}, "questions": questions,
                "assets": {ref: {"type": "image", "format": "jpeg", "data_base64": base64.b64encode(data).decode()}}}
        t0 = time.monotonic()
        for attempt in range(5):
            try:
                r = self.http.post(self.d["url"], json=body)
            except httpx.TransportError:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
                continue
            if r.status_code in (408, 429, 500, 502, 503, 529) and attempt < 4:
                time.sleep(2 ** attempt)
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"{self.name} {r.status_code}: {r.text[:300]}")
            break
        out = r.json()
        self.ledger.fast_calls += 1
        self.ledger.fast_s += time.monotonic() - t0
        self.ledger.fast_usd += out.get("usage", {}).get("input_tokens", 0) * self.d["usd_per_token"]
        self.ledger.note(f"{self.name}:{use}")
        return out["answers"]


class Claude:
    """Sets the goal and re-plans after repeated failures. Never picks low-level moves."""

    def __init__(self, ledger: Ledger, model: str = CLAUDE_MODEL):
        import anthropic

        self.ledger = ledger
        self.model = model
        self.client = anthropic.Anthropic()

    def ask(self, system: str, text: str, frame: np.ndarray | None, schema: dict, use: str) -> dict:
        content = []
        if frame is not None:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                        "data": base64.b64encode(jpeg(frame)).decode()}})
        content.append({"type": "text", "text": text})
        t0 = time.monotonic()
        r = self.client.messages.create(model=self.model, max_tokens=2000, system=system,
                                        output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
                                        messages=[{"role": "user", "content": content}])
        self.ledger.slow_calls += 1
        self.ledger.slow_s += time.monotonic() - t0
        pin, pout = CLAUDE_USD_PER_TOKEN.get(self.model, CLAUDE_USD_PER_TOKEN["claude-opus-5-5"])
        self.ledger.slow_usd += r.usage.input_tokens * pin + r.usage.output_tokens * pout
        self.ledger.note(f"claude:{use}")
        return json.loads(next(b.text for b in r.content if b.type == "text"))
