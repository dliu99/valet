"""Jev (TypeSafe System One) answers one yes/no question about the wrist frame. Needs TYPESAFE_API_KEY."""

from __future__ import annotations

IMAGES_HEADER = {"TypeSafe-Danger-Zone-Images": "true"}
FRAME_NOTE = " (`wrist` is a frame from the camera on the robot's gripper, looking out past the jaws.)"


class Jev:
    def __init__(self, model: str | None = None, timeout: float = 15.0):
        from typesafe_sdk import TypeSafeClient

        self.client = TypeSafeClient()
        self.model = model
        self.timeout = timeout

    def ask(self, jpeg: bytes, question: str) -> tuple[float, dict]:
        from typesafe_sdk import Image, Noul

        r = self.client.system_one(
            state={"wrist": Image(jpeg)},
            questions={"q": Noul(instructions=question + FRAME_NOTE)},
            model=self.model,
            timeout=self.timeout,
            extra_headers=IMAGES_HEADER,
        )
        usage = getattr(r, "usage", None)
        return float(r.answers["q"].noul), {"input_tokens": getattr(usage, "input_tokens", None)}
