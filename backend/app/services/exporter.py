"""Stage 11 — ordered storybook output: PDF (reportlab, built-in fonts) and ZIP (images + manifest)."""

from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph

from app.schemas.jobs import Job
from app.services.styles import get_style

INK = colors.HexColor("#1f2430")
MUTED = colors.HexColor("#6b7280")
ACCENT = colors.HexColor("#6d5dfc")


class NothingToExport(ValueError):
    pass


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48] or "scene"


def _selected_images(job: Job, media_dir: Path) -> list[tuple[int, str, str, Path]]:
    out = []
    for s in job.scenes:
        v = s.selected()
        if v is not None and (media_dir / v.file).exists():
            out.append((s.index, s.title, s.text, media_dir / v.file))
    if not out:
        raise NothingToExport("No scenes have been illustrated yet.")
    return out


def _page_size(aspect: str) -> tuple[float, float]:
    if aspect == "9:16":
        return A4
    if aspect == "1:1":
        return (8.5 * inch, 8.5 * inch)
    return landscape(A4)


def _jpeg_reader(path: Path, max_side: int = 1600) -> ImageReader:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return ImageReader(buf)


def _draw_image(c: Canvas, path: Path, x: float, y: float, w: float, h: float) -> None:
    reader = _jpeg_reader(path)
    iw, ih = reader.getSize()
    scale = min(w / iw, h / ih)
    dw, dh = iw * scale, ih * scale
    c.drawImage(reader, x + (w - dw) / 2, y + (h - dh) / 2, dw, dh)


def build_pdf(job: Job, media_dir: Path) -> bytes:
    scenes = _selected_images(job, media_dir)
    page_w, page_h = _page_size(job.settings.aspect_ratio)
    margin = 14 * mm
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=(page_w, page_h))
    c.setTitle(job.title)
    c.setAuthor("StoryPix")
    style = get_style(job.settings.style)
    title_style = ParagraphStyle(
        "t", fontName="Times-Bold", fontSize=30, leading=36, alignment=TA_CENTER, textColor=INK
    )
    body = ParagraphStyle("b", fontName="Times-Roman", fontSize=13, leading=18, textColor=INK)
    heading = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=10, leading=13, textColor=ACCENT)

    # cover
    c.setFillColor(colors.HexColor("#faf9ff"))
    c.rect(0, 0, page_w, page_h, fill=1, stroke=0)
    p = Paragraph(job.title, title_style)
    _, ph = p.wrap(page_w - 2 * margin, page_h)
    p.drawOn(c, margin, page_h - margin - ph - 10 * mm)
    cover_top = page_h - margin - ph - 18 * mm
    _draw_image(c, scenes[0][3], margin, 26 * mm, page_w - 2 * margin, cover_top - 26 * mm)
    c.setFont("Helvetica", 9)
    c.setFillColor(MUTED)
    c.drawCentredString(
        page_w / 2, 14 * mm, f"An illustrated story · {style.name} · made with StoryPix · {datetime.now(UTC):%B %Y}"
    )
    c.showPage()

    # scene pages
    for index, title, text, path in scenes:
        text_block = Paragraph(text.replace("\n", "<br/>"), body)
        head = Paragraph(f"SCENE {index} · {title.upper()}", heading)
        tw = page_w - 2 * margin
        _, hh = head.wrap(tw, page_h)
        _, bh = text_block.wrap(tw, page_h)
        text_h = hh + bh + 8 * mm
        image_h = page_h - 2 * margin - text_h
        _draw_image(c, path, margin, margin + text_h, tw, image_h)
        head.drawOn(c, margin, margin + bh + 4 * mm)
        text_block.drawOn(c, margin, margin)
        c.setFont("Helvetica", 8)
        c.setFillColor(MUTED)
        c.drawRightString(page_w - margin, 6 * mm, str(index))
        c.showPage()

    # cast page
    cast = [ch for ch in job.characters if ch.anchor() is not None and (media_dir / ch.anchor().file).exists()]
    if cast:
        c.setFont("Times-Bold", 22)
        c.setFillColor(INK)
        c.drawString(margin, page_h - margin - 12 * mm, "The Characters")
        cols = min(4, len(cast))
        cell_w = (page_w - 2 * margin) / cols
        img = min(cell_w - 8 * mm, 60 * mm)
        y = page_h - margin - 24 * mm - img
        small = ParagraphStyle("s", fontName="Helvetica", fontSize=8, leading=10, textColor=MUTED)
        for i, ch in enumerate(cast):
            col, row = i % cols, i // cols
            x = margin + col * cell_w
            yy = y - row * (img + 32 * mm)
            if yy < margin:
                break
            _draw_image(c, media_dir / ch.anchor().file, x, yy, img, img)
            c.setFont("Helvetica-Bold", 10)
            c.setFillColor(INK)
            c.drawString(x, yy - 5 * mm, ch.name)
            para = Paragraph(ch.descriptor, small)
            _, h = para.wrap(cell_w - 6 * mm, 30 * mm)
            para.drawOn(c, x, yy - 7 * mm - h)
        c.showPage()
    c.save()
    return buf.getvalue()


def build_zip(job: Job, media_dir: Path) -> bytes:
    scenes = _selected_images(job, media_dir)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for index, title, _, path in scenes:
            zf.write(path, f"{index:02d}-{_slug(title)}.png")
        for ch in job.characters:
            anchor = ch.anchor()
            if anchor is not None and (media_dir / anchor.file).exists():
                zf.write(media_dir / anchor.file, f"characters/{_slug(ch.name)}.png")
        story_txt = [job.title, "=" * len(job.title), ""]
        for s in job.scenes:
            story_txt += [f"Scene {s.index}: {s.title}", s.text, ""]
        zf.writestr("story.txt", "\n".join(story_txt))
        manifest = {
            "title": job.title,
            "created_at": job.created_at.isoformat(),
            "settings": job.settings.model_dump(),
            "analysis": job.analysis.model_dump(),
            "characters": [
                {
                    "id": ch.id,
                    "name": ch.name,
                    "species": ch.species,
                    "descriptor": ch.descriptor,
                    "attributes": ch.attributes,
                    "anchor": ch.anchor().model_dump(mode="json") if ch.anchor() else None,
                }
                for ch in job.characters
            ],
            "scenes": [
                {
                    "index": s.index,
                    "title": s.title,
                    "text": s.text,
                    "summary": s.summary,
                    "characters": s.character_ids,
                    "location": s.location.model_dump() if s.location else None,
                    "selected": s.selected().model_dump(mode="json") if s.selected() else None,
                }
                for s in job.scenes
            ],
        }
        zf.writestr("storybook.json", json.dumps(manifest, indent=2))
    return buf.getvalue()
