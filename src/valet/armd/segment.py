"""SAM 3 masks of named objects in a camera frame, and the few numbers a skill needs from each mask.

SAM 3 is Meta's open-vocabulary segmenter. It runs here on Apple silicon through mlx-vlm's port,
`mlx-community/sam3-bf16` (Meta's weights converted to MLX; `facebook/sam3` itself is gated): 1.7 GB of
weights, 3-4 GB while running. One backbone pass per frame serves every prompt; each prompt then runs
the detector's encoder, decoder and mask head. mlx-vlm is optional (the `sam` extra) and imported only
when the first request loads the model.
"""

from __future__ import annotations

import base64
import io
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from PIL import Image, ImageFilter

from .errors import ArmdError

MODEL = "mlx-community/sam3-bf16"
SIZES = (336, 504, 672, 1008)
DEFAULT_SIZE = 504
KEEP_SCORE = 0.1
FOUND_SCORE = 0.5
SAME_OBJECT_IOU = 0.7
EDGE_MARGIN = 0.005


@dataclass(frozen=True)
class Instance:
    mask: np.ndarray  # (H, W) bool, in the frame's pixels
    score: float


class Model(Protocol):
    def segment(self, image: Image.Image, prompts: list[str], size: int) -> dict[str, list[Instance]]: ...


class Sam3:
    def __init__(self, repo: str = MODEL):
        try:
            import mlx.core as mx
            from mlx_vlm.models.sam3 import generate
            from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor
            from mlx_vlm.utils import get_model_path, load_model
        except ImportError as e:
            raise ArmdError(501, "unavailable", f"SAM 3 needs mlx-vlm, see the `sam` extra in pyproject.toml ({e})") from None
        try:
            path = get_model_path(repo, local_files_only=True)
        except Exception:
            path = get_model_path(repo)
        self.mx, self.gen = mx, generate
        self.model = load_model(path)
        self.processor = Sam3Processor.from_pretrained(str(path))
        self.predictor = generate.Sam3Predictor(self.model, self.processor, score_threshold=KEEP_SCORE)

    def segment(self, image: Image.Image, prompts: list[str], size: int) -> dict[str, list[Instance]]:
        mx, gen = self.mx, self.gen
        self.processor.image_size = size
        pixels = mx.array(self.processor.preprocess_image(image)["pixel_values"])
        src, pos, fpn, (hf, wf) = gen._get_det_features(self.model, gen._get_backbone_features(self.model, pixels))
        det = self.model.detector_model
        w, h = image.size
        out: dict[str, list[Instance]] = {}
        for prompt in prompts:
            embeds, attn = self.predictor._get_input_embeddings(prompt)
            encoded = det.detr_encoder(src, pos, embeds, attn)
            hs, _, presence = det.detr_decoder(vision_features=encoded, inputs_embeds=embeds, vision_pos_encoding=pos,
                                               text_mask=attn, spatial_shape=(hf, wf))
            scores = mx.sigmoid(det.dot_product_scoring(hs, embeds, attn)[-1].squeeze(-1)[0].squeeze())
            if presence is not None:
                scores = scores * mx.sigmoid(presence[-1][0])
            logits = det.mask_decoder(hs[-1], list(fpn), encoder_hidden_states=encoded, prompt_features=embeds,
                                      prompt_mask=attn)["pred_masks"]
            mx.eval(scores, logits)
            s = np.array(scores.astype(mx.float32)).reshape(-1)
            keep = np.flatnonzero(s > KEEP_SCORE)
            keep = keep[np.argsort(-s[keep])]
            found: list[Instance] = []
            if len(keep):
                masks = gen._resize_masks(np.array(logits[0].astype(mx.float32))[keep], (h, w)) > 0
                for k, mask in zip(keep, masks):
                    if mask.any() and not any(_iou(mask, i.mask) > SAME_OBJECT_IOU for i in found):
                        found.append(Instance(mask, float(s[k])))
            out[prompt] = found
        return out


class Segmenter:
    """The model, loaded by the first request, and the one thread that runs it: MLX ties its GPU streams
    to the thread that made them, and one request at a time keeps memory flat."""

    def __init__(self, load: Callable[[], Model] = Sam3):
        self._load = load
        self._model: Model | None = None
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sam3")

    def segment(self, image: Image.Image, prompts: list[str], size: int) -> tuple[dict[str, list[Instance]], dict]:
        return self._worker.submit(self._run, image, prompts, size).result()

    def _run(self, image: Image.Image, prompts: list[str], size: int) -> tuple[dict[str, list[Instance]], dict]:
        timing = {}
        if self._model is None:
            t0 = time.monotonic()
            self._model = self._load()
            timing["load_ms"] = round((time.monotonic() - t0) * 1000.0)
        t0 = time.monotonic()
        found = self._model.segment(image, prompts, size)
        timing["ms"] = round((time.monotonic() - t0) * 1000.0)
        return found, timing


def edge_band(sensor: np.ndarray) -> np.ndarray:
    """The pixels within a small margin of the frame's edge, or of the corners the upright turn filled
    with black (`sensor` is False where no camera pixel landed). A mask that reaches into it may be cut off."""
    h, w = sensor.shape
    m = max(2, round(EDGE_MARGIN * max(w, h)))
    band = np.zeros((h, w), bool)
    if not sensor.all():
        outside = Image.fromarray(np.where(sensor, 0, 255).astype(np.uint8))
        band = np.asarray(outside.filter(ImageFilter.MaxFilter(2 * m + 1))) > 0
    band[:m], band[-m:], band[:, :m], band[:, -m:] = True, True, True, True
    return band


def mask_stats(mask: np.ndarray, band: np.ndarray) -> dict:
    """Where the mask is, in u, v of 0..1 across and down the frame (pixel centres), and whether all of it
    is in view. Each extreme point is the middle of the mask's outermost row or column on that side."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())

    def at(x: float, y: float) -> dict:
        return {"u": round((float(x) + 0.5) / w, 4), "v": round((float(y) + 0.5) / h, 4)}

    cut = mask & band
    ty, tx = np.nonzero(cut)
    nearest = np.argmin(np.stack([tx, w - 1 - tx, ty, h - 1 - ty]), axis=0) if len(tx) else np.array([], int)
    touches = [side for k, side in enumerate(("left", "right", "top", "bottom")) if (nearest == k).any()]
    return {
        "whole": not touches,
        "touches": touches,
        "area": round(float(len(xs)) / (w * h), 5),
        "bbox_px": [x0, y0, x1 + 1, y1 + 1],
        "box": {"left": round(x0 / w, 4), "top": round(y0 / h, 4), "right": round((x1 + 1) / w, 4), "bottom": round((y1 + 1) / h, 4)},
        "centroid": at(xs.mean(), ys.mean()),
        "top": at(xs[ys == y0].mean(), y0),
        "bottom": at(xs[ys == y1].mean(), y1),
        "left": at(x0, ys[xs == x0].mean()),
        "right": at(x1, ys[xs == x1].mean()),
    }


def mask_png_b64(mask: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(mask).convert("1").save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def describe(found: list[Instance], band: np.ndarray, with_mask: bool = False) -> dict:
    """The best-scoring instance of one prompt. `count` is how many instances cleared FOUND_SCORE."""
    count = sum(i.score >= FOUND_SCORE for i in found)
    if not found or found[0].score < FOUND_SCORE:
        return {"found": False, "score": round(found[0].score, 3) if found else 0.0, "count": 0}
    best = found[0]
    out = {"found": True, "score": round(best.score, 3), "count": count, **mask_stats(best.mask, band)}
    if with_mask:
        out["mask_png_b64"] = mask_png_b64(best.mask)
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0
