"""Jev, TypeSafe's System One model: typed answers to narrow questions about wrist frames.

Questions are raw System One dicts ({"type": "noul" | "choice" | "score" | "box", "instructions", "criteria"?}).
Their ids are for code only; the text carries the whole meaning and names the frame as `wrist`."""

from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image as PILImage

JEV_USD_PER_INPUT_TOKEN = 0.042 / 1e6
IMAGES_HEADER = {"TypeSafe-Danger-Zone-Images": "true"}


def load_env() -> None:
    """Read KEY=value lines from the nearest .env above this file (the repository's, or the checkout's that holds
    this worktree) into the environment, without echoing them. Variables already set win."""
    for d in Path(__file__).resolve().parents:
        p = d / ".env"
        if p.is_file():
            for line in p.read_text().splitlines():
                k, sep, v = line.strip().partition("=")
                if sep and k and not k.startswith("#"):
                    os.environ.setdefault(k, v.strip().strip('"').strip("'"))
            return


def jpeg(image: PILImage.Image, max_side: int = 640, quality: int = 88) -> bytes:
    im = image.copy()
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=quality)
    return buf.getvalue()


@dataclass
class Meter:
    free: bool = False  # a local model: tokens are counted, nothing is billed
    calls: int = 0
    input_tokens: int = 0
    seconds: float = 0.0
    log: list[dict] = field(default_factory=list)

    @property
    def usd(self) -> float:
        return 0.0 if self.free else self.input_tokens * JEV_USD_PER_INPUT_TOKEN


class Jev:
    def __init__(self, model: str = "jev-latest", timeout: float = 20.0):
        from typesafe_sdk import TypeSafeClient

        load_env()
        self.client = TypeSafeClient()
        self.model = model
        self.timeout = timeout
        self.meter = Meter()

    def ask(self, frames: dict[str, PILImage.Image], questions: dict[str, dict], facts: dict[str, Any] | None = None) -> dict[str, dict]:
        """Every question about the same frames in one request. Answers are plain dicts:
        noul -> {"p"}, box -> {"box": {top,bottom,left,right}, "confidence"}, choice -> {"choice", "probabilities", "confidence"},
        score -> {"score", "probabilities", "confidence"}."""
        from typesafe_sdk import Image

        state: dict[str, Any] = {name: Image(jpeg(im)) for name, im in frames.items()}
        if facts:
            state.update(facts)
        t0 = time.monotonic()
        for attempt in range(3):
            try:
                r = self.client.system_one(state=state, questions=questions, model=self.model, timeout=self.timeout,
                                           extra_headers=IMAGES_HEADER)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1.0 + attempt)
        dt = time.monotonic() - t0
        usage = getattr(r, "usage", None)
        tokens = int(getattr(usage, "input_tokens", 0) or 0)
        self.meter.calls += 1
        self.meter.input_tokens += tokens
        self.meter.seconds += dt
        out = {qid: _plain(r.answers[qid]) for qid in questions}
        self.meter.log.append({"questions": {k: q["instructions"] for k, q in questions.items()}, "answers": out,
                               "tokens": tokens, "s": round(dt, 3)})
        return out


def _plain(a) -> dict:
    t = a.type
    if t == "noul":
        return {"p": round(float(a.noul), 4)}
    if t == "box":
        b = a.box
        conf = getattr(a, "confidence", None)
        return {"box": {"top": b.top, "bottom": b.bottom, "left": b.left, "right": b.right},
                "confidence": None if conf is None else round(float(conf), 3)}
    if t == "choice":
        return {"choice": a.choice, "confidence": round(float(a.confidence), 3),
                "probabilities": {k: round(float(v), 4) for k, v in dict(a.probabilities).items()}}
    if t == "score":
        return {"score": round(float(a.score), 3), "confidence": round(float(a.confidence), 3),
                "probabilities": {str(k): round(float(v), 4) for k, v in dict(a.probabilities).items()}}
    raise TypeError(f"unexpected answer type {t}")


class LocalSystemOne:
    """The same System One API served locally (H2O-Lightning-4B on 127.0.0.1:8741, no key), over plain HTTP.
    Same interface as Jev, so the policy can swap its per-frame model."""

    def __init__(self, url: str = "http://127.0.0.1:8741/v1/systemone", model: str = "h2o-lightning-4b", timeout: float = 600.0):
        self.url, self.model, self.timeout = url, model, timeout
        self.meter = Meter(free=True)

    def ask(self, frames: dict[str, PILImage.Image], questions: dict[str, dict], facts: dict[str, Any] | None = None) -> dict[str, dict]:
        import base64
        import json
        import urllib.request
        import uuid

        state: dict[str, Any] = {}
        assets = {}
        for name, im in frames.items():
            ref = f"{name}-{uuid.uuid4().hex[:12]}"
            state[name] = ref
            assets[ref] = {"type": "image", "format": "jpeg", "data_base64": base64.b64encode(jpeg(im)).decode()}
        if facts:
            state.update(facts)
        body = json.dumps({"model": self.model, "state": state, "questions": questions, "assets": assets}).encode()
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json", **IMAGES_HEADER})
        t0 = time.monotonic()
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read())
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2.0 + 2 * attempt)
        dt = time.monotonic() - t0
        tokens = int((data.get("usage") or {}).get("input_tokens", 0) or 0)
        self.meter.calls += 1
        self.meter.input_tokens += tokens
        self.meter.seconds += dt
        out = {qid: _plain_dict(data["answers"][qid]) for qid in questions}
        self.meter.log.append({"questions": {k: q["instructions"] for k, q in questions.items()}, "answers": out,
                               "tokens": tokens, "s": round(dt, 3)})
        return out


def _plain_dict(a: dict) -> dict:
    t = a.get("type")
    if t == "noul":
        return {"p": round(float(a["noul"]), 4)}
    if t == "box":
        b = a["box"]
        conf = a.get("confidence")
        return {"box": {k: float(b[k]) for k in ("top", "bottom", "left", "right")},
                "confidence": None if conf is None else round(float(conf), 3), "present": a.get("present")}
    if t == "choice":
        return {"choice": a["choice"], "confidence": round(float(a.get("confidence") or 0), 3),
                "probabilities": {k: round(float(v), 4) for k, v in (a.get("probabilities") or {}).items()}}
    if t == "score":
        return {"score": round(float(a["score"]), 3), "confidence": round(float(a.get("confidence") or 0), 3),
                "probabilities": {str(k): round(float(v), 4) for k, v in (a.get("probabilities") or {}).items()}}
    raise TypeError(f"unexpected answer type {t}")
