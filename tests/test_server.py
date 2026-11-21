import json
import threading
import urllib.error
import urllib.request

import pytest

from valet.armd.server import make_server

ARM_STATE_KEYS = {"t", "kind", "allow_motion", "joints", "tip", "pan_deg", "reach_m", "height_m", "pitch_deg", "roll_deg",
                  "gripper_pct", "grip", "amps", "peak_amps", "torque"}


class Client:
    def __init__(self, port: int):
        self.base = f"http://127.0.0.1:{port}"

    def call(self, method: str, path: str, body: dict | None = None, headers: dict | None = None) -> tuple[int, dict]:
        data = None if method == "GET" else json.dumps(body or {}).encode()
        h = {} if method == "GET" else {"Content-Type": "application/json"}
        req = urllib.request.Request(self.base + path, data=data, method=method, headers={**h, **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def get(self, path: str) -> tuple[int, dict]:
        return self.call("GET", path)

    def post(self, path: str, body: dict | None = None) -> tuple[int, dict]:
        return self.call("POST", path, body)

    def act(self, **motion) -> tuple[int, dict]:
        return self.post("/act", {"motion": motion})


def serving(app):
    server = make_server(app, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
    return server, Client(server.server_address[1])


@pytest.fixture
def client(make_app):
    server, c = serving(make_app())
    c.app = server.RequestHandlerClass.app
    yield c
    server.shutdown()


@pytest.fixture
def still_client(make_app):
    server, c = serving(make_app(allow_motion=False))
    yield c
    server.shutdown()


def test_state_has_every_arm_state_field(client):
    status, s = client.get("/state")
    assert status == 200 and set(s) == ARM_STATE_KEYS
    assert s["kind"] == "fake" and set(s["joints"]) == {"shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"}


def test_motion_before_hold_is_refused_in_the_result(client):
    status, r = client.act(kind="delta", pan_deg=5)
    assert status == 200 and r["ok"] is False and "hold" in r["refused"]


def test_hold_act_and_result_shape(client):
    assert client.post("/hold")[1]["torque"] is True
    status, r = client.act(kind="delta", reach_m=0.01)
    assert status == 200 and r["ok"] is True
    assert set(r) >= {"ok", "before", "after", "ms"} and set(r["after"]) == ARM_STATE_KEYS


def test_motion_not_allowed_is_403_but_remember_state_and_frames_work(still_client):
    status, err = still_client.act(kind="delta", pan_deg=5)
    assert (status, err["kind"]) == (403, "motion_not_allowed")
    assert still_client.act(kind="grip", to="close")[0] == 403
    assert still_client.act(kind="remember", name="start")[1]["ok"] is True
    assert "start" in still_client.get("/poses")[1]
    assert still_client.get("/state")[0] == 200
    assert set(still_client.post("/frames", {"cams": ["wrist", "side"]})[1]["frames"]) == {"wrist", "side"}


def test_unreachable_is_422(client):
    client.post("/hold")
    status, err = client.act(kind="delta", reach_m=0.4)
    assert (status, err["kind"]) == (422, "unreachable")


def test_bad_requests_and_unknown_routes(client):
    assert client.get("/nowhere")[0] == 404
    assert client.act(kind="pose", name="nowhere")[0] == 404
    assert client.act(kind="dance")[0] == 400
    assert client.call("POST", "/hold", headers={"Content-Type": "text/plain"})[0] == 400
    status, err = client.call("GET", "/state", headers={"Host": "evil.example"})
    assert status == 403


def test_frames_endpoint(client):
    status, out = client.post("/frames", {"cams": ["wrist", "side", "laptop"], "max_side": 200})
    assert status == 200 and set(out["frames"]) == {"wrist", "side"} and set(out["missing"]) == {"laptop"}
    wrist = out["frames"]["wrist"]
    assert max(wrist["width"], wrist["height"]) == 200 and len(wrist["sha256"]) == 64 and wrist["upright"] is True


def test_relax_drops_torque(client):
    client.post("/hold")
    status, s = client.post("/relax")
    assert status == 200 and s["torque"] is False


def test_fake_truth_and_reset(client):
    status, truth = client.post("/fake/reset", {"box": {"x": 0.25, "y": 0.05, "w": 0.03, "d": 0.03, "h": 0.04}})
    assert status == 200 and truth["held"] is False
    assert truth["box"] == pytest.approx({"x": 0.25, "y": 0.05, "z": 0.02, "w": 0.03, "d": 0.03, "h": 0.04})
    assert set(client.get("/fake/truth")[1]) == {"box", "held", "tip"}
