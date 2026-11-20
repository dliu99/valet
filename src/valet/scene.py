"""Where things were last seen. Observed, not inferred: each entry keeps when and how well."""

from __future__ import annotations

import time
from dataclasses import dataclass

from .robot import Vec3


@dataclass(frozen=True)
class Seen:
    xyz: Vec3
    span_m: float  # larger side of the box on the table
    p_present: float
    box_confidence: float | None
    offset: float  # box centre's distance from the picture centre; small = less distortion
    t: float


class SceneMap:
    def __init__(self) -> None:
        self._seen: dict[str, Seen] = {}

    def update(self, name: str, seen: Seen) -> None:
        self._seen[name] = seen

    def get(self, name: str, max_age_s: float | None = None) -> Seen | None:
        s = self._seen.get(name)
        if s is None:
            return None
        if max_age_s is not None and time.monotonic() - s.t > max_age_s:
            return None
        return s

    def forget(self, name: str) -> None:
        self._seen.pop(name, None)

    def as_dict(self) -> dict[str, dict]:
        return {
            k: {"xyz": [round(c, 4) for c in s.xyz], "span_m": round(s.span_m, 4), "p_present": round(s.p_present, 3)}
            for k, s in self._seen.items()
        }
