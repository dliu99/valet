import base64
import hashlib
import io
import threading
import time

import numpy as np
import pytest
from PIL import Image

from test_server import serving
from valet.armd.segment import FOUND_SCORE, Instance, Segmenter, describe, edge_band, mask_stats


def rect(h, w, x0, y0, x1, y1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_stats_of_a_rectangle_in_view():
    h, w = 100, 200
    mask = rect(h, w, 40, 20, 80, 60)
    s = mask_stats(mask, edge_band(np.ones((h, w), bool)))
    assert s["whole"] is True and s["touches"] == []
    assert s["bbox_px"] == [40, 20, 80, 60]
    assert s["box"] == {"left": 0.2, "top": 0.2, "right": 0.4, "bottom": 0.6}
    assert s["area"] == pytest.approx(40 * 40 / (h * w))
    assert s["centroid"] == {"u": 0.3, "v": 0.4}
    assert s["top"] == {"u": 0.3, "v": 0.205}
    assert s["bottom"] == {"u": 0.3, "v": 0.595}
    assert s["left"] == {"u": 0.2025, "v": 0.4}
    assert s["right"] == {"u": 0.3975, "v": 0.4}


def test_extremes_are_the_middle_of_the_outermost_row_and_column():
    h, w = 100, 100
    mask = np.zeros((h, w), bool)
    for y in range(10, 60):  # a triangle: apex at the top, wide base at the bottom
        half = (y - 10) // 2
        mask[y, 50 - half:51 + half] = True
    s = mask_stats(mask, edge_band(np.ones((h, w), bool)))
    assert s["top"] == {"u": 0.505, "v": 0.105}
    assert s["bottom"]["u"] == pytest.approx(0.505) and s["bottom"]["v"] == 0.595
    assert s["left"]["v"] == s["right"]["v"] == pytest.approx(0.585, abs=0.006)
    assert s["centroid"]["v"] > 0.4, "the wide base pulls the centroid down"


def test_a_mask_cut_off_by_the_frame_is_not_whole_and_says_where():
    h, w = 100, 200
    band = edge_band(np.ones((h, w), bool))
    assert mask_stats(rect(h, w, 0, 30, 30, 60), band)["touches"] == ["left"]
    s = mask_stats(rect(h, w, 150, 70, 200, 100), band)
    assert s["whole"] is False and s["touches"] == ["right", "bottom"]


def test_the_black_corners_of_a_turned_frame_count_as_frame_edge():
    h, w = 120, 200
    sensor = np.asarray(Image.new("L", (w, h), 255).rotate(15, resample=Image.Resampling.BILINEAR)) > 127
    assert not sensor[0, 0], "turning leaves the corners black"
    band = edge_band(sensor)
    corner = rect(h, w, 10, 10, 30, 30)
    assert mask_stats(corner, band)["whole"] is False
    assert mask_stats(rect(h, w, 90, 50, 110, 70), band)["whole"] is True


def test_describe_reports_the_best_instance_and_counts_the_confident_ones():
    h, w = 50, 50
    band = edge_band(np.ones((h, w), bool))
    a, b = rect(h, w, 5, 5, 15, 15), rect(h, w, 30, 30, 40, 40)
    out = describe([Instance(a, 0.9), Instance(b, 0.6), Instance(b, 0.2)], band, with_mask=True)
    assert out["found"] is True and out["score"] == 0.9 and out["count"] == 2
    assert out["bbox_px"] == [5, 5, 15, 15]
    png = Image.open(io.BytesIO(base64.b64decode(out["mask_png_b64"])))
    assert np.array_equal(np.asarray(png.convert("L")) > 0, a)
    weak = describe([Instance(a, FOUND_SCORE - 0.01)], band)
    assert weak == {"found": False, "score": round(FOUND_SCORE - 0.01, 3), "count": 0}
    assert describe([], band) == {"found": False, "score": 0.0, "count": 0}


class WhiteThings:
    """Stands in for SAM 3: "white box" is every near-white pixel; nothing else is ever found."""

    def __init__(self, delay_s=0.0):
        self.delay_s = delay_s
        self.seen: list[tuple[tuple[int, int], list[str], int]] = []

    def segment(self, image, prompts, size):
        self.seen.append((image.size, prompts, size))
        time.sleep(self.delay_s)
        white = np.asarray(image.convert("L")) > 230
        return {p: [Instance(white, 0.93)] if p == "white box" and white.any() else [] for p in prompts}


@pytest.fixture
def seg_client(make_app):
    app = make_app()
    loads = []
    model = WhiteThings()
    app.segmenter = Segmenter(lambda: loads.append(1) or model)
    server, c = serving(app)
    c.model, c.loads = model, loads
    yield c
    server.shutdown()


def test_segment_route_finds_the_fake_box_in_the_upright_wrist_frame(seg_client):
    status, out = seg_client.post("/segment", {"prompts": ["white box", "red cup"], "frame": True, "mask": True})
    assert status == 200, out
    frame = out["frame"]
    data = base64.b64decode(frame["data_b64"])
    assert hashlib.sha256(data).hexdigest() == frame["sha256"] and frame["cam"] == "wrist"
    image = Image.open(io.BytesIO(data))
    assert seg_client.model.seen == [(image.size, ["white box", "red cup"], 504)]
    box = out["results"]["white box"]
    assert box["found"] is True and box["score"] == 0.93 and box["whole"] is True
    white = np.asarray(image.convert("L")) > 230
    ys, xs = np.nonzero(white)
    assert box["centroid"]["u"] == pytest.approx((xs.mean() + 0.5) / image.width, abs=0.01)
    assert box["centroid"]["v"] == pytest.approx((ys.mean() + 0.5) / image.height, abs=0.01)
    assert out["results"]["red cup"] == {"found": False, "score": 0.0, "count": 0}
    assert out["load_ms"] >= 0 and out["ms"] >= 0 and out["size"] == 504


def test_segment_loads_the_model_once_and_leaves_the_frame_out_unless_asked(seg_client):
    for size in (504, 336):
        status, out = seg_client.post("/segment", {"prompts": ["white box"], "size": size, "max_side": 320})
        assert status == 200 and "data_b64" not in out["frame"] and "mask_png_b64" not in out["results"]["white box"]
        assert out["frame"]["width"] == 320
    assert seg_client.loads == [1] and [s for _, _, s in seg_client.model.seen] == [504, 336]
    assert "load_ms" not in out


@pytest.mark.parametrize("body", [{}, {"prompts": []}, {"prompts": "white box"}, {"prompts": [""]}, {"prompts": ["x"] * 9},
                                  {"prompts": ["x"], "size": 500}, {"prompts": ["x"], "size": 504.0}, {"prompts": ["x"], "cam": 3},
                                  {"prompts": ["x"], "max_side": 8}])
def test_segment_rejects_bad_requests(seg_client, body):
    status, err = seg_client.post("/segment", body)
    assert (status, err["kind"]) == (400, "bad_request")
    assert seg_client.loads == []


def test_segment_from_a_missing_camera_is_503(seg_client):
    status, err = seg_client.post("/segment", {"prompts": ["x"], "cam": "laptop"})
    assert (status, err["kind"]) == (503, "no_frame")


def test_state_answers_while_a_segment_runs(make_app):
    app = make_app()
    app.segmenter = Segmenter(lambda: WhiteThings(delay_s=0.6))
    server, c = serving(app)
    try:
        slow = threading.Thread(target=c.post, args=("/segment", {"prompts": ["white box"]}))
        slow.start()
        time.sleep(0.1)
        t0 = time.monotonic()
        assert c.get("/state")[0] == 200
        assert time.monotonic() - t0 < 0.3
        assert slow.is_alive(), "the segment was still running"
        slow.join()
    finally:
        server.shutdown()
