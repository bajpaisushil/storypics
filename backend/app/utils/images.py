"""Image file helpers: saving originals + web thumbnails, validating uploads."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}
MAX_UPLOAD_SIDE = 1024
THUMB_SIDE = 384


class InvalidImage(ValueError):
    pass


def save_with_thumb(image: Image.Image, path: Path) -> tuple[Path, Path]:
    """Save a PNG original and a WebP thumbnail next to it; returns both paths."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)
    thumb = image.copy()
    thumb.thumbnail((THUMB_SIDE, THUMB_SIDE))
    thumb_path = path.with_name(path.stem + "_thumb.webp")
    thumb.save(thumb_path, format="WEBP", quality=82)
    return path, thumb_path


def load_upload(data: bytes, max_bytes: int) -> Image.Image:
    if len(data) > max_bytes:
        raise InvalidImage(f"Image is larger than {max_bytes // (1024 * 1024)} MB.")
    try:
        img = Image.open(io.BytesIO(data))
        fmt = img.format
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImage("File is not a readable image.") from exc
    if fmt not in ALLOWED_FORMATS:
        raise InvalidImage(f"Unsupported image format {fmt}; use PNG, JPEG or WebP.")
    img = ImageOps.exif_transpose(img).convert("RGB")
    if min(img.size) < 64:
        raise InvalidImage("Image is too small (minimum 64 px).")
    img.thumbnail((MAX_UPLOAD_SIDE, MAX_UPLOAD_SIDE))
    return img
