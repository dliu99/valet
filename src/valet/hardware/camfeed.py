"""Keep the newest frame of one camera in a file.

macOS only lets a process with camera permission open a camera, and an agent's shell usually
has none. Run this from a terminal that does; everything else reads the file.

    python -m valet.hardware.camfeed --name Innomaker --out runs/live/wrist.jpg

The camera is opened by its exact AVFoundation name through ffmpeg. OpenCV numbers cameras in
its own order, which differs from every listing, so an index can open the wrong camera.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import select
import subprocess
import time

REFUSED = ("facetime", "iphone", "capture screen")
SOI, EOI = b"\xff\xd8", b"\xff\xd9"


def cameras() -> list[str]:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                         capture_output=True, text=True, check=False).stderr
    video = out.split("AVFoundation video devices:", 1)[-1].split("AVFoundation audio devices:", 1)[0]
    return re.findall(r"\[\d+\] (.+)", video)


def pick(name: str) -> str:
    names = cameras()
    hits = [n for n in names if name.lower() in n.lower()]
    if len(hits) != 1:
        raise SystemExit(f"{len(hits)} cameras named like {name!r} among {names}")
    if any(r in hits[0].lower() for r in REFUSED):
        raise SystemExit(f"refusing {hits[0]!r}: only the robot's cameras are opened")
    return hits[0]


GARBLED_ROW_RATIO = 0.9
GARBLED_MIN_DETAIL = 2.0
GARBLED_RESTART_AFTER = 3


def garbled(jpeg: bytes) -> bool:
    """A misread pixel format shows as two side-by-side copies striped every other row: neighbouring rows then
    differ as much as rows two apart, where in a real image they differ less."""
    import numpy as np
    from PIL import Image

    a = np.asarray(Image.open(io.BytesIO(jpeg)).convert("L"), np.float32)
    d1 = float(np.abs(a[1:] - a[:-1]).mean())
    d2 = float(np.abs(a[2:] - a[:-2]).mean())
    return d2 > GARBLED_MIN_DETAIL and d1 / d2 > GARBLED_ROW_RATIO


def stream(device: str, a: argparse.Namespace) -> subprocess.Popen:
    return subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "avfoundation", "-pixel_format", "uyvy422",
         "-framerate", str(a.fps), "-video_size", f"{a.width}x{a.height}", "-i", f"{device}:none",
         "-vf", f"fps={a.hz}", "-f", "image2pipe", "-c:v", "mjpeg", "-q:v", "3", "-"],
        stdout=subprocess.PIPE)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="Innomaker")
    p.add_argument("--out", required=True)
    # Raw frames: 1280x720 at 10 fps is about 150 Mb/s; 720p30 nearly fills a USB 2.0 hub the arm's
    # serial adapter shares, and the adapter drops off. armd's locate is fitted on 16:9 frames.
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--stall-s", type=float, default=3.0)
    a = p.parse_args()
    tmp = a.out + ".tmp.jpg"
    n = 0
    while True:
        try:
            device = pick(a.name)
        except SystemExit as e:
            print(f"{e}; retrying", flush=True)
            time.sleep(2)
            continue
        ffmpeg, buf, bad = stream(device, a), b"", 0
        fd = ffmpeg.stdout.fileno()
        why = f"no frame for {a.stall_s:.0f} s"
        while bad < GARBLED_RESTART_AFTER and select.select([fd], [], [], a.stall_s)[0] and (chunk := os.read(fd, 65536)):
            buf += chunk
            while (start := buf.find(SOI)) >= 0 and (end := buf.find(EOI, start + 2)) >= 0:
                jpeg, buf = buf[start:end + 2], buf[end + 2:]
                if garbled(jpeg):
                    bad += 1
                    why = f"{bad} garbled frames in a row"
                    continue
                bad = 0
                with open(tmp, "wb") as f:
                    f.write(jpeg)
                os.replace(tmp, a.out)
                n += 1
                if n == 1 or n % 300 == 0:
                    print(f"{device}: {a.width}x{a.height}, {n} frames written", flush=True)
        ffmpeg.kill()
        ffmpeg.wait()
        print(f"{why}; restarting the camera", flush=True)


if __name__ == "__main__":
    main()
