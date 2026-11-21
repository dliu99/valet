"""SAM 3 (Meta's open-vocabulary segmenter) on Apple silicon through mlx-vlm's port, `mlx-community/sam3-bf16`
(facebook/sam3 is gated; the MLX conversion carries the same weights under the SAM License). `version="3.1"` loads
SAM 3.1 (`mlx-community/sam3.1-bf16`), whose detector differs only in its neck.

One vision-backbone pass per frame serves every text prompt; each prompt then runs SAM 3's DETR encoder, decoder
and mask head. `Segmenter.segment` returns, per prompt, the instances above a score floor, best first."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

MODELS = {"3": "mlx-community/sam3-bf16", "3.1": "mlx-community/sam3.1-bf16"}


@dataclass
class Instance:
    mask: np.ndarray  # (H, W) bool, in the frame's pixels
    score: float
    box: tuple[float, float, float, float]  # xyxy pixels

    @property
    def area(self) -> int:
        return int(self.mask.sum())


@dataclass
class Segmenter:
    resolution: int = 1008
    floor: float = 0.1
    version: str = "3"
    calls: list[dict] = field(default_factory=list)

    def __post_init__(self):
        import mlx.core as mx
        from mlx_vlm.models.sam3.generate import Sam3Predictor
        from mlx_vlm.utils import get_model_path, load_model

        if self.version == "3.1":
            from mlx_vlm.models.sam3_1 import generate as gen
            from mlx_vlm.models.sam3_1.processing_sam3_1 import Sam31Processor as Processor
        else:
            from mlx_vlm.models.sam3 import generate as gen
            from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor as Processor
        self._backbone, self._det_features = gen._get_backbone_features, gen._get_det_features
        path = get_model_path(MODELS[self.version])
        self.mx = mx
        self.model = load_model(path)
        self.processor = Processor.from_pretrained(str(path))
        self.processor.image_size = self.resolution
        self.predictor = Sam3Predictor(self.model, self.processor, score_threshold=self.floor)
        self._last = None

    def segment(self, image: Image.Image, prompts: list[str]) -> dict[str, list[Instance]]:
        from mlx_vlm.models.sam3.generate import _resize_masks

        mx = self.mx
        t0 = time.time()
        same = self._last is not None and self._last[0] is image and self._last[1] == self.processor.image_size
        if same:  # more prompts on a frame already seen: reuse its backbone pass
            src, pos_flat, fpn, (hf, wf) = self._last[2]
        else:
            pixel_values = mx.array(self.processor.preprocess_image(image)["pixel_values"])
            backbone = self._backbone(self.model, pixel_values)
            src, pos_flat, fpn, (hf, wf) = self._det_features(self.model, backbone)  # fpn: the levels the mask head reads
            self._last = (image, self.processor.image_size, (src, pos_flat, fpn, (hf, wf)))
        det = self.model.detector_model
        w, h = image.size
        out: dict[str, list[Instance]] = {}
        for prompt in prompts:
            embeds, attn = self.predictor._get_input_embeddings(prompt)
            encoded = det.detr_encoder(src, pos_flat, embeds, attn)
            hs, ref_boxes, presence = det.detr_decoder(vision_features=encoded, inputs_embeds=embeds,
                                                       vision_pos_encoding=pos_flat, text_mask=attn, spatial_shape=(hf, wf))
            cx, cy, bw, bh = (ref_boxes[-1][..., k] for k in range(4))
            boxes = mx.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1)[0] * mx.array([w, h, w, h])
            logits = det.dot_product_scoring(hs, embeds, attn)[-1].squeeze(-1)
            scores = mx.sigmoid(logits[0].squeeze())
            if presence is not None:
                scores = scores * mx.sigmoid(presence[-1][0])
            seg = det.mask_decoder(hs[-1], list(fpn), encoder_hidden_states=encoded, prompt_features=embeds, prompt_mask=attn)
            mx.eval(scores, boxes, seg["pred_masks"])
            s = np.array(scores.astype(mx.float32)).reshape(-1)
            keep = np.flatnonzero(s > self.floor)
            keep = keep[np.argsort(-s[keep])]
            insts = []
            if len(keep):
                logit_masks = np.array(seg["pred_masks"][0].astype(mx.float32))[keep]
                masks = _resize_masks(logit_masks, (h, w)) > 0
                b = np.array(boxes.astype(mx.float32))[keep]
                for k in range(len(keep)):
                    if masks[k].any() and not any(_iou(masks[k], i.mask) > 0.7 for i in insts):
                        insts.append(Instance(masks[k], float(s[keep[k]]), tuple(float(v) for v in b[k])))
            out[prompt] = insts
        self.calls.append({"s": round(time.time() - t0, 3), "prompts": len(prompts), "new_frame": not same})
        return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0
