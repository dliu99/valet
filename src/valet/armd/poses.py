from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import numpy as np

from .kinematics import ARM

MAX_NAME_LEN = 64


class PoseBook:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._poses: dict[str, dict[str, float]] = json.loads(self.path.read_text()) if self.path.is_file() else {}

    def get(self, name: str) -> np.ndarray | None:
        with self._lock:
            joints = self._poses.get(name)
        return None if joints is None else np.array([joints[m] for m in ARM], float)

    def remember(self, name: str, arm_joints: np.ndarray) -> None:
        if not isinstance(name, str) or not name or len(name) > MAX_NAME_LEN:
            raise ValueError(f"a pose name is a non-empty string of at most {MAX_NAME_LEN} characters")
        with self._lock:
            self._poses[name] = {m: round(float(v), 2) for m, v in zip(ARM, arm_joints)}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._poses, indent=1))
            os.replace(tmp, self.path)

    def as_json(self) -> dict:
        with self._lock:
            return {name: {"joints": dict(j)} for name, j in self._poses.items()}
