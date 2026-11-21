"""POSTs must carry `Content-Type: application/json` and the Host must be loopback, so a web page
in a browser on this machine cannot send the arm a motion.
"""

from __future__ import annotations

import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable
from urllib.parse import urlparse

import numpy as np

from .app import Armd
from .errors import ArmdError

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
MAX_BODY_BYTES = 1 << 20

ROUTES: dict[tuple[str, str], Callable[[Armd, dict], dict]] = {
    ("GET", "/state"): lambda app, body: app.state(),
    ("GET", "/poses"): lambda app, body: app.pose_book(),
    ("GET", "/fake/truth"): lambda app, body: app.fake_truth(),
    ("POST", "/frames"): lambda app, body: app.frames(body.get("cams"), body.get("max_side")),
    ("POST", "/act"): lambda app, body: app.act(body.get("motion")),
    ("POST", "/hold"): lambda app, body: app.hold(),
    ("POST", "/relax"): lambda app, body: app.relax(),
    ("POST", "/locate"): lambda app, body: app.locate(body),
    ("POST", "/triangulate"): lambda app, body: app.triangulate(body),
    ("POST", "/segment"): lambda app, body: app.segment(body),
    ("POST", "/fake/reset"): lambda app, body: app.fake_reset(body.get("box")),
}
LOGGED = {"/act", "/hold", "/relax", "/fake/reset"}


def _plain(o: object) -> object:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON")


class Handler(BaseHTTPRequestHandler):
    app: Armd
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        self._route("GET")

    def do_POST(self) -> None:
        self._route("POST")

    def _route(self, method: str) -> None:
        path = urlparse(self.path).path.rstrip("/") or "/"
        try:
            raw = self._drain_body()
            self._check_origin(method)
            body = self._parse(raw)
            route = ROUTES.get((method, path))
            if route is None:
                raise ArmdError(404, "not_found", f"no route {method} {path}")
            self._send(200, route(self.app, body))
        except ArmdError as e:
            self._send(e.status, {"error": str(e), "kind": e.kind})
        except Exception as e:
            traceback.print_exc()
            self._send(500, {"error": f"{type(e).__name__}: {e}", "kind": "internal"})

    def _check_origin(self, method: str) -> None:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        if host not in LOOPBACK_HOSTS:
            raise ArmdError(403, "forbidden_host", f"Host {host!r} is not loopback")
        if method == "POST" and (self.headers.get("Content-Type") or "").split(";")[0].strip() != "application/json":
            raise ArmdError(400, "bad_request", "POST bodies are JSON: send Content-Type: application/json")

    def _drain_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            self.close_connection = True
            raise ArmdError(400, "bad_request", "body too large")
        return self.rfile.read(length) if length else b""

    def _parse(self, raw: bytes) -> dict:
        if not raw.strip():
            return {}
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ArmdError(400, "bad_request", f"body is not JSON: {e}") from None
        if not isinstance(body, dict):
            raise ArmdError(400, "bad_request", "body is a JSON object")
        return body

    def _send(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, default=_plain).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        path = urlparse(self.path).path.rstrip("/")
        if path in LOGGED or (isinstance(code, int) and code >= 400):
            sys.stderr.write(f"armd: {self.command} {path} -> {code}\n")


def make_server(app: Armd, host: str, port: int) -> ThreadingHTTPServer:
    handler = type("ArmdHandler", (Handler,), {"app": app})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server
