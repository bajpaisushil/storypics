"""Stage 10 — evaluate a generated image: text alignment, quality, character identity consistency.

* alignment   — Model C head on frozen SigLIP features (falls back to calibrated zero-shot SigLIP cosine)
* quality     — Model D head on frozen SigLIP + DINOv2 features + OpenCV statistics (None without checkpoint)
* consistency — DINOv2 cosine similarity between each character's anchor and its region of the scene
                (zero-shot metric; regions come from the same layout used for masked IP-Adapter conditioning)

Scores drive automatic retries and are shown in the UI next to each image.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import Settings
from app.schemas.generation import Region
from app.schemas.jobs import ImageScores

log = logging.getLogger(__name__)

SIGLIP = "google/siglip-base-patch16-224"
DINO = "facebook/dinov2-small"
ALIGN_CKPT = "scene_alignment.pt"
QUALITY_CKPT = "quality_evaluator.pt"


@dataclass
class IdentityRef:
    character_id: str
    anchor_path: Path
    region: Region | None


def image_statistics(image: Image.Image) -> np.ndarray:
    """Cheap OpenCV diagnostics used as extra quality features: sharpness, contrast, colourfulness, exposure."""
    import cv2

    arr = np.asarray(image.convert("RGB").resize((256, 256)), dtype=np.float32)
    gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    sharp = cv2.Laplacian(gray, cv2.CV_32F).var()
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    rg, yb = r - g, 0.5 * (r + g) - b
    colorful = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    hist = np.histogram(gray, bins=32, range=(0, 255))[0] / gray.size
    entropy = -(hist[hist > 0] * np.log2(hist[hist > 0])).sum()
    clipped = ((gray < 5) | (gray > 250)).mean()
    return np.array(
        [np.log1p(sharp) / 8, gray.std() / 64, colorful / 100, entropy / 5, gray.mean() / 255, clipped],
        dtype=np.float32,
    )


def crop_region(image: Image.Image, region: Region | None) -> Image.Image:
    if region is None:
        return image
    w, h = image.size
    return image.crop((int(region.x0 * w), int(region.y0 * h), int(region.x1 * w), int(region.y1 * h)))


class ImageEvaluator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self._loaded = False
        self.error: str | None = None
        self.align_head = None
        self.quality_head = None

    @property
    def available(self) -> bool:
        return self.settings.enable_evaluator and self.error is None

    def describe(self) -> dict:
        return {
            "enabled": self.settings.enable_evaluator,
            "loaded": self._loaded,
            "alignment": "model-c" if (self.settings.checkpoints_dir / ALIGN_CKPT).exists() else "zero-shot-siglip",
            "quality": "model-d" if (self.settings.checkpoints_dir / QUALITY_CKPT).exists() else "unavailable",
            "consistency": "dinov2-zero-shot",
            "error": self.error,
        }

    def _load(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModel, AutoProcessor

        from app.models.quality_evaluator import QualityHead
        from app.models.scene_alignment import AlignmentHead

        local = not self.settings.allow_downloads
        self.torch = torch
        self.siglip = AutoModel.from_pretrained(SIGLIP, local_files_only=local).eval()
        self.siglip_proc = AutoProcessor.from_pretrained(SIGLIP, local_files_only=local)
        self.dino = AutoModel.from_pretrained(DINO, local_files_only=local).eval()
        self.dino_proc = AutoImageProcessor.from_pretrained(DINO, local_files_only=local)
        ckpt = self.settings.checkpoints_dir
        if (ckpt / ALIGN_CKPT).exists():
            self.align_head, _ = AlignmentHead.load(ckpt / ALIGN_CKPT)
        if (ckpt / QUALITY_CKPT).exists():
            self.quality_head, _ = QualityHead.load(ckpt / QUALITY_CKPT)
        self._loaded = True

    # --- features ------------------------------------------------------------------------------
    def siglip_image(self, images: list[Image.Image]):
        inputs = self.siglip_proc(images=images, return_tensors="pt")
        feats = self.siglip.get_image_features(**inputs)
        feats = getattr(feats, "pooler_output", feats)
        return self.torch.nn.functional.normalize(feats, dim=-1)

    def siglip_text(self, texts: list[str]):
        inputs = self.siglip_proc(text=texts, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        feats = self.siglip.get_text_features(**inputs)
        feats = getattr(feats, "pooler_output", feats)
        return self.torch.nn.functional.normalize(feats, dim=-1)

    def dino_embed(self, images: list[Image.Image]):
        inputs = self.dino_proc(images=images, return_tensors="pt")
        out = self.dino(**inputs)
        return self.torch.nn.functional.normalize(out.pooler_output, dim=-1)

    # --- scoring -------------------------------------------------------------------------------
    def evaluate(self, image: Image.Image, text: str, identities: list[IdentityRef]) -> ImageScores | None:
        if not self.settings.enable_evaluator or self.error:
            return None
        try:
            with self._lock, __import__("torch").inference_mode():
                self._load()
                return self._evaluate(image, text, identities)
        except Exception as exc:  # evaluation must never break generation
            self.error = f"evaluator failed: {exc}"
            log.exception("evaluator failed; disabling it for this process")
            return None

    def _evaluate(self, image: Image.Image, text: str, identities: list[IdentityRef]) -> ImageScores:
        torch = self.torch
        img_feat = self.siglip_image([image])
        txt_feat = self.siglip_text([text])
        cos = float((img_feat * txt_feat).sum())
        if self.align_head is not None:
            alignment = float(self.align_head.score(img_feat, txt_feat)[0])
        else:
            # SigLIP cosines for matching pairs sit roughly in [-0.05, 0.2]; map to [0, 1] for display
            alignment = float(np.clip((cos + 0.05) / 0.25, 0, 1))
        quality = None
        if self.quality_head is not None:
            dino = self.dino_embed([image])
            stats = torch.from_numpy(image_statistics(image))[None]
            quality = float(self.quality_head.score(img_feat, dino, stats)[0])
        consistency: dict[str, float] = {}
        for ref in identities:
            if not ref.anchor_path.exists():
                continue
            anchor = Image.open(ref.anchor_path).convert("RGB")
            emb = self.dino_embed([anchor, crop_region(image, ref.region)])
            consistency[ref.character_id] = round(float((emb[0] * emb[1]).sum()), 4)
        parts = [(alignment, 0.45)]
        if quality is not None:
            parts.append((quality, 0.25))
        if consistency:
            parts.append((float(np.mean(list(consistency.values()))), 0.30))
        overall = sum(v * w for v, w in parts) / sum(w for _, w in parts)
        return ImageScores(
            alignment=round(alignment, 4),
            quality=None if quality is None else round(quality, 4),
            consistency=consistency,
            overall=round(overall, 4),
        )
