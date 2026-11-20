"""Finding a described thing in one frame: a bounding box in pixels, from Jev (TypeSafe Box question) or Claude.

Answers are cached on disk by (backend, description, hash of the frame), so re-running a program on the same
scene from the same pose costs nothing; a frame that differs by one pixel is a new question."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from . import ledger

CACHE_DIR = Path(__file__).resolve().parents[2] / "state" / "localize_cache"
CLAUDE_MODEL = os.environ.get("ASPIRE_LOCALIZE_MODEL", "claude-opus-5-5")


@dataclass(frozen=True)
class Found:
    found: bool
    bbox: tuple[float, float, float, float] | None  # left, top, right, bottom in pixels
    confidence: float | None
    backend: str
    cached: bool = False


def frame_key(raw: np.ndarray) -> str:
    return hashlib.sha256(raw.tobytes()).hexdigest()[:24]


def jpeg(image: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def locate(image: Image.Image, text: str, key: str, backend: str = "jev") -> Found:
    path = CACHE_DIR / backend / hashlib.sha256(f"{text}\n{key}".encode()).hexdigest()[:32]
    if path.exists():
        d = json.loads(path.read_text())
        return Found(d["found"], tuple(d["bbox"]) if d["bbox"] else None, d["confidence"], backend, cached=True)
    if backend == "jev":
        out = _jev(image, text)
    elif backend == "claude":
        out = _claude(image, text)
    else:
        raise ValueError(f"unknown localize backend {backend!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**asdict(out), "text": text}))
    return out


def _jev(image: Image.Image, text: str) -> Found:
    from typesafe_sdk import Box, Image as TsImage, TypeSafeClient

    ledger.check(ledger.JEV_USD_PER_CALL)
    client = TypeSafeClient()
    try:
        r = client.system_one(
            state={"frame": TsImage(jpeg(image))},
            questions={"where": Box(instructions=f"Draw a box around {text} in `frame`.")},
            timeout=30.0,
            extra_headers={"TypeSafe-Danger-Zone-Images": "true"},
        )
    finally:
        client.close()
    ledger.record("localize:jev", "jev", ledger.JEV_USD_PER_CALL)
    a = r.answers["where"]
    b = a.box
    w, h = image.size
    conf = None if a.confidence is None else float(a.confidence)
    if b is None:
        return Found(False, None, conf, "jev")
    bbox = (b.left * w, b.top * h, b.right * w, b.bottom * h)
    return Found(bool(conf is None or conf >= 0.3) and bbox[2] > bbox[0] and bbox[3] > bbox[1], bbox, conf, "jev")


CLAUDE_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "left": {"type": "integer"},
        "top": {"type": "integer"},
        "right": {"type": "integer"},
        "bottom": {"type": "integer"},
    },
    "required": ["found", "left", "top", "right", "bottom"],
    "additionalProperties": False,
}


def _claude(image: Image.Image, text: str) -> Found:
    import anthropic

    ledger.check(0.02)
    w, h = image.size
    client = anthropic.Anthropic()
    r = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4000,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": CLAUDE_SCHEMA}},
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(jpeg(image)).decode()}},
            {"type": "text", "text": f"This image is {w} x {h} pixels. Give the tight bounding box of {text}, in pixels "
                                     f"(left, top, right, bottom; origin at the top left). If it is not visible, set found to false."},
        ]}],
    )
    ledger.record_claude("localize:claude", CLAUDE_MODEL, r.usage)
    d = json.loads(next(b.text for b in r.content if b.type == "text"))
    if not d["found"]:
        return Found(False, None, None, "claude")
    return Found(True, (float(d["left"]), float(d["top"]), float(d["right"]), float(d["bottom"])), None, "claude")
