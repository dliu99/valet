"""Jev (TypeSafe System One) box questions, for comparison with SAM 3 on the same frames. Request shape as in
point-and-reach's `models.py`. The key comes from valet's .env and is never printed."""

from __future__ import annotations

import base64
import io
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
USD_PER_MTOK = 0.042


def load_env() -> None:
    p = next((d / ".env" for d in Path(__file__).resolve().parents if (d / ".env").is_file()), None)
    if p is None:
        p = Path.home() / "projects" / "valet" / ".env"
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@dataclass
class JevBox:
    left: float  # pixels
    top: float
    right: float
    bottom: float
    confidence: float

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.left + self.right) / 2, (self.top + self.bottom) / 2)


@dataclass
class Jev:
    calls: list[dict] = field(default_factory=list)

    def __post_init__(self):
        import httpx

        load_env()
        self.http = httpx.Client(timeout=60.0)
        self.key = os.environ["TYPESAFE_API_KEY"]

    def boxes(self, image: Image.Image, things: dict[str, str]) -> dict[str, JevBox]:
        buf = io.BytesIO()
        image.save(buf, "JPEG", quality=90)
        ref = f"frame-{uuid.uuid4().hex[:12]}"
        body = {"model": MODEL, "state": {"frame": ref},
                "assets": {ref: {"type": "image", "format": "jpg", "data_base64": base64.b64encode(buf.getvalue()).decode()}},
                "questions": {k: {"type": "box", "instructions": v} for k, v in things.items()}}
        headers = {"Authorization": f"Bearer {self.key}", "TypeSafe-Danger-Zone-Images": "true"}
        t0 = time.time()
        for attempt in range(4):
            r = self.http.post(URL, json=body, headers=headers)
            if r.status_code < 500 and r.status_code != 429:
                break
            time.sleep(2 ** attempt)
        r.raise_for_status()
        out = r.json()
        tok = out.get("usage", {}).get("input_tokens", 0)
        self.calls.append({"s": round(time.time() - t0, 3), "usd": tok * USD_PER_MTOK / 1e6})
        w, h = image.size
        res = {}
        for k in things:
            a = out["answers"][k]
            b = a["box"]
            res[k] = JevBox(b["left"] * w, b["top"] * h, b["right"] * w, b["bottom"] * h, float(a.get("confidence", 0.0)))
        return res
