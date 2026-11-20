"""The sim over HTTP, in real time, for harnesses in any language. Loopback only.

  GET  /state                      joints, goals, amps, torque, grasp point
  POST /goals   {joints: {...}}    joint goals in degrees (LeRobot calibrated); the servos slew to them
  POST /move    {xyz, pitch_deg, roll_deg?, gripper?}   IK for the grasp point, then goals; 422 if unreachable
  POST /torque  {on: bool}
  POST /frames  {cams: ["wrist", "side"], max_side?}    upright JPEG frames, base64
  POST /reset   {seed}             a benchmark episode (task.episode); the scene's layout is not returned
  GET  /score                      the episode's score from the true box pose: for evaluation, never for control
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .ik import IK, Unreachable
from .sim import Sim, SimSpec, realtime, with_box
from .task import episode, score

LOOPBACK = {"127.0.0.1", "localhost", "[::1]"}


class App:
    def __init__(self, seed: int):
        self.lock = threading.Lock()
        self.sim: Sim | None = None
        self.reset(seed)

    def reset(self, seed: int) -> dict:
        ep = episode(int(seed))
        with self.lock:
            if self.sim is None:
                self.sim = realtime(with_box(SimSpec(), ep.box, ep.target_xy))
                self.ik = IK(self.sim.model)
            else:
                self.sim.reset(ep.box, ep.target_xy)
            self.seed = ep.seed
        return {"seed": ep.seed}

    def frames(self, cams: list[str], max_side: int | None) -> dict:
        out = {}
        for cam in cams:
            image = self.sim.frame(cam)
            if max_side:
                image.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            image.save(buf, "JPEG", quality=90)
            out[cam] = {"mime": "image/jpeg", "width": image.width, "height": image.height, "data_b64": base64.b64encode(buf.getvalue()).decode()}
        return {"frames": out}

    def move(self, body: dict) -> dict:
        q = self.ik.solve(body["xyz"], float(body["pitch_deg"]), float(body.get("roll_deg", -90.0)),
                          seed=self.sim.joints(), gripper=body.get("gripper"))
        self.sim.set_goals(q)
        return {"goals": q}

    def handle(self, method: str, path: str, body: dict) -> tuple[int, dict]:
        sim = self.sim
        if (method, path) == ("GET", "/state"):
            return 200, sim.state()
        if (method, path) == ("POST", "/goals"):
            sim.set_goals(body["joints"])
            return 200, sim.state()
        if (method, path) == ("POST", "/move"):
            try:
                return 200, self.move(body)
            except Unreachable as e:
                return 422, {"error": str(e)}
        if (method, path) == ("POST", "/torque"):
            sim.torque(bool(body["on"]))
            return 200, sim.state()
        if (method, path) == ("POST", "/frames"):
            return 200, self.frames(body.get("cams", ["wrist"]), body.get("max_side"))
        if (method, path) == ("POST", "/reset"):
            return 200, self.reset(body.get("seed", 0))
        if (method, path) == ("GET", "/score"):
            return 200, {"seed": self.seed, **score(sim.truth())}
        return 404, {"error": f"no route {method} {path}"}


def make_server(app: App, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def _go(self, method: str) -> None:
            if self.headers.get("Host", "").rsplit(":", 1)[0] not in LOOPBACK:
                return self._send(403, {"error": "loopback only"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}") if method == "POST" else {}
                self._send(*app.handle(method, self.path.split("?")[0], body))
            except (KeyError, TypeError, ValueError) as e:
                self._send(400, {"error": f"{type(e).__name__}: {e}"})

        def do_GET(self) -> None:
            self._go("GET")

        def do_POST(self) -> None:
            self._go("POST")

        def _send(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="valet-sim", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    server = make_server(App(a.seed), a.port)
    print(f"valet-sim on http://127.0.0.1:{a.port}, episode seed {a.seed}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
