"""Command-line interface — the same pipeline as the API, without the web server.

    python -m app.cli analyze story.txt [--scenes 5] [--json]
    python -m app.cli generate story.txt --style watercolor --scenes 5 --aspect 4:3 --out out/ [--seed 42]
    python -m app.cli previews            # real style-preview thumbnails for the frontend (needs the image model)
    python -m app.cli benchmark           # latency + memory of one 1-MP scene on this machine

`generate` is the batch mode used by the free-GPU notebook: it runs analysis → anchors → scenes → evaluation
and writes images, storybook.pdf and storybook.zip into --out.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from app.config import get_settings

FRONTEND_STYLES = Path(__file__).resolve().parents[2] / "frontend" / "public" / "styles"


def _read(path: str) -> str:
    return sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")


def cmd_analyze(args) -> None:
    from app.services.story_parser import StoryAnalyzer

    analysis = StoryAnalyzer(get_settings()).analyze(_read(args.story), args.scenes)
    if args.json:
        print(analysis.model_dump_json(indent=2))
        return
    print(
        f"{analysis.title}  ·  {len(analysis.scenes)} scenes  ·  segmenter={analysis.meta.segmenter}  "
        f"tagger={analysis.meta.tagger}"
    )
    for c in analysis.characters:
        print(f"  {c.id} {c.name} ({c.species}) scenes={c.scene_indices} → {c.descriptor}")
    for s in analysis.scenes:
        print(
            f"\n[{s.index}] {s.title}\n    {s.summary}\n    visual: {s.visual_description}\n    "
            f"setting: {s.location.description if s.location else '-'} · {s.time_of_day or 'day'} · {s.composition}"
        )
    for w in analysis.warnings:
        print(f"! {w}")


async def _generate(args) -> int:
    from app.schemas.jobs import CreateJobRequest
    from app.services.events import EventBus
    from app.services.exporter import build_pdf, build_zip
    from app.services.image_evaluator import ImageEvaluator
    from app.services.image_generator import create_generator
    from app.services.job_manager import JobManager
    from app.services.job_store import JobStore
    from app.services.story_parser import StoryAnalyzer

    out = Path(args.out).resolve()
    settings = get_settings().model_copy(update={"data_dir": out / "data"})
    manager = JobManager(
        settings,
        JobStore(settings.db_path),
        EventBus(),
        StoryAnalyzer(settings),
        create_generator(settings),
        ImageEvaluator(settings) if settings.enable_evaluator else None,
    )
    await manager.start()
    try:
        job = await manager.create_job(
            CreateJobRequest(
                story=_read(args.story),
                style=args.style,
                scene_count=args.scenes,
                aspect_ratio=args.aspect,
                seed=args.seed,
            )
        )
        print(
            f"job {job.id}: {len(job.scenes)} scenes, "
            f"{sum(c.anchor_status == 'queued' for c in job.characters)} character anchors"
        )
        last = ""
        while job.status in ("queued", "generating"):
            await asyncio.sleep(0.5)
            cur = job.current
            line = f"{cur.label} · step {cur.step}/{cur.total_steps}" if cur else ""
            if line and line != last:
                print("  " + line, flush=True)
                last = line
        print(f"status: {job.status}")
        for s in job.scenes:
            v = s.selected()
            score = f" overall={v.scores.overall}" if v and v.scores else ""
            print(f"  [{s.index}] {s.status:9s} {s.title}{score} {s.error or ''}")
        if any(s.selected() for s in job.scenes):
            (out / "storybook.pdf").write_bytes(build_pdf(job, settings.media_dir))
            (out / "storybook.zip").write_bytes(build_zip(job, settings.media_dir))
            print(
                f"wrote {out / 'storybook.pdf'} and {out / 'storybook.zip'} (images in {settings.media_dir / job.id})"
            )
        return 0 if job.status == "completed" else 1
    finally:
        await manager.stop()


def cmd_previews(args) -> None:
    from PIL import Image

    from app.schemas.generation import GenerationRequest
    from app.services.image_generator import create_generator
    from app.services.prompt_builder import PromptBuilder
    from app.services.styles import STYLES

    settings = get_settings()
    gen = create_generator(settings)
    FRONTEND_STYLES.mkdir(parents=True, exist_ok=True)
    for style in STYLES.values():
        if args.only and style.id not in args.only:
            continue
        built = PromptBuilder(style).preview_prompt()
        image, info = gen.generate(
            GenerationRequest(
                kind="preview",
                prompt=built.prompt,
                negative_prompt=built.negative_prompt,
                width=1024,
                height=1024,
                seed=args.seed,
                guidance_scale=style.guidance_scale,
            )
        )
        image = image.convert("RGB")
        image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        path = FRONTEND_STYLES / f"{style.id}.webp"
        image.save(path, format="WEBP", quality=85)
        print(f"{style.id:18s} {info.duration_s:6.1f}s → {path}")


def cmd_benchmark(args) -> None:
    import platform

    from app.schemas.generation import GenerationRequest
    from app.services.image_generator import create_generator

    settings = get_settings()
    gen = create_generator(settings)
    t0 = time.perf_counter()
    gen.load()
    load_s = time.perf_counter() - t0
    req = GenerationRequest(
        kind="scene",
        prompt="children's picture book illustration, a hen meeting a cow on a farm",
        width=1152,
        height=896,
        seed=1,
    )
    gen.generate(req)  # warm-up (MPS kernels compile on first use)
    runs = []
    for seed in range(2, 2 + args.runs):
        _, info = gen.generate(req.model_copy(update={"seed": seed}))
        runs.append(info)
    result = {
        "machine": platform.platform(),
        "processor": platform.processor(),
        "device": runs[0].device,
        "profile": runs[0].profile,
        "steps": runs[0].steps,
        "size": f"{req.width}x{req.height}",
        "load_seconds": round(load_s, 1),
        "seconds_per_image": [r.duration_s for r in runs],
        "peak_memory_gb": max((r.peak_memory_gb or 0) for r in runs),
    }
    print(json.dumps(result, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="storypix", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("story", help="path to a .txt file, or - for stdin")
    a.add_argument("--scenes", default="auto", type=lambda v: v if v == "auto" else int(v))
    a.add_argument("--json", action="store_true")
    g = sub.add_parser("generate")
    g.add_argument("story")
    g.add_argument("--style", default="storybook")
    g.add_argument("--scenes", default="auto", type=lambda v: v if v == "auto" else int(v))
    g.add_argument("--aspect", default="4:3", choices=["1:1", "4:3", "16:9", "9:16"])
    g.add_argument("--seed", type=int)
    g.add_argument("--out", default="out")
    p = sub.add_parser("previews")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--only", nargs="*")
    b = sub.add_parser("benchmark")
    b.add_argument("--runs", type=int, default=2)
    args = ap.parse_args()
    if args.cmd == "analyze":
        cmd_analyze(args)
    elif args.cmd == "generate":
        sys.exit(asyncio.run(_generate(args)))
    elif args.cmd == "previews":
        cmd_previews(args)
    else:
        cmd_benchmark(args)


if __name__ == "__main__":
    main()
