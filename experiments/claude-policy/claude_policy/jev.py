"""Jev (TypeSafe System One) answers one yes/no question about the current camera frames. Needs
TYPESAFE_API_KEY. Images are a TypeSafe preview feature ("Danger Zone"): fine for a prototype."""

from __future__ import annotations

IMAGES_HEADER = {"TypeSafe-Danger-Zone-Images": "true"}
CAMERAS = {"wrist": "`wrist` is a frame from the camera on the robot arm's gripper, looking out past the jaws",
           "side": "`side` is from a fixed camera beside the table"}


class Jev:
    def __init__(self, model: str | None = None, timeout: float = 15.0):
        from typesafe_sdk import TypeSafeClient

        self.client = TypeSafeClient()
        self.model = model
        self.timeout = timeout

    def ask(self, frames: dict[str, bytes], question: str) -> tuple[float, dict]:
        from typesafe_sdk import Image, Noul

        r = self.client.system_one(
            state={cam: Image(jpeg) for cam, jpeg in frames.items()},
            questions={"q": Noul(instructions=f"{question} ({'; '.join(CAMERAS[c] for c in frames)}.)")},
            model=self.model,
            timeout=self.timeout,
            extra_headers=IMAGES_HEADER,
        )
        usage = getattr(r, "usage", None)
        return float(r.answers["q"].noul), {"input_tokens": getattr(usage, "input_tokens", None)}
