from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class Settings:
    url: str = "http://127.0.0.1:8750"
    port: str = "/dev/tty.usbmodem5B901520711"
    robot_id: str = "valet_follower"
    allow_motion: bool = False
    table_z: float = 0.0
    speed_dps: float = 25.0
    cameras: dict[str, Path] = field(default_factory=lambda: {"wrist": Path("runs/live/wrist.jpg"), "side": Path("runs/live/side.jpg")})

    @property
    def listen(self) -> tuple[str, int]:
        u = urlparse(self.url)
        return u.hostname or "127.0.0.1", u.port or 8750


def load(path: Path | None) -> Settings:
    if path is None:
        return Settings()
    doc = tomllib.loads(Path(path).read_text())
    armd = doc.get("armd", {})
    known = {k: armd[k] for k in ("url", "port", "robot_id", "allow_motion", "table_z", "speed_dps") if k in armd}
    cams = {cam: Path(p) for cam, p in doc.get("cameras", {}).items() if cam in ("wrist", "side")}
    return Settings(**known, **({"cameras": cams} if cams else {}))
