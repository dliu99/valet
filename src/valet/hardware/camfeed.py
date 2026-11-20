"""Keep the newest wrist-camera frame in a file.

macOS only lets a process with camera permission open a camera, and an agent's shell usually
has none. Run this from a terminal that does; everything else reads the file.

    python -m valet.hardware.camfeed --name Innomaker --out /path/wrist.jpg
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time


def find_index(name: str) -> int:
    """AVFoundation's index of the first camera whose name contains `name`."""
    out = subprocess.run(
        ["system_profiler", "SPCameraDataType"], capture_output=True, text=True, check=False
    ).stdout
    names = [ln.strip()[:-1] for ln in out.splitlines() if ln.startswith("    ") and not ln.startswith("     ") and ln.strip().endswith(":")]
    for i, n in enumerate(names):
        if name.lower() in n.lower():
            return i
    raise SystemExit(f"no camera named like {name!r} among {names}")


def main() -> None:
    import cv2

    p = argparse.ArgumentParser()
    p.add_argument("--index", type=int)
    p.add_argument("--name", default="Innomaker")
    p.add_argument("--out", required=True)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--hz", type=float, default=5.0)
    a = p.parse_args()
    index = a.index if a.index is not None else find_index(a.name)
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    if not cap.isOpened():
        raise SystemExit(f"camera {index} did not open (camera permission?)")
    tmp = a.out + ".tmp.jpg"
    n, last = 0, 0.0
    while True:
        ok, frame = cap.read()  # read every frame so the buffer never goes stale
        if not ok:
            time.sleep(0.05)
            continue
        if time.time() - last >= 1.0 / a.hz:
            cv2.imwrite(tmp, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            os.replace(tmp, a.out)
            last = time.time()
            n += 1
            if n == 1 or n % 300 == 0:
                print(f"camera {index}: {frame.shape[1]}x{frame.shape[0]}, {n} frames written", flush=True)


if __name__ == "__main__":
    main()
