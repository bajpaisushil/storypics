"""Ablation: does reference conditioning actually keep characters consistent?

The spec asks not to assume that repeating a text description guarantees the same-looking character. This script
measures it. For the two-character scenes of the example story it renders each scene three ways, with the same
seeds and prompts:

  text-only   canonical descriptors in the prompt, no image reference
  unmasked    both character anchors as IP-Adapter references over the whole canvas
  masked      each anchor confined to its own region with IP-Adapter attention masks (StoryPix default)

Identity consistency = DINOv2 cosine similarity between a character's anchor portrait and the scene. Two
variants are reported: the region the layout assigns to the character, and the better of the two halves (which does
not penalise a method for swapping sides). Alignment uses the production evaluator.

    python -m training.ablation_consistency --seeds 3 --style storybook
Outputs checkpoints/ablation_consistency.json and docs/images/ablation_consistency.jpg (contact sheet).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from PIL import Image, ImageDraw

from app.config import get_settings
from app.schemas.generation import GenerationRequest, ReferenceImage, Region
from app.services.image_evaluator import IdentityRef, ImageEvaluator
from app.services.image_generator import LocalDiffusersGenerator
from app.services.prompt_builder import PromptBuilder, image_size
from app.services.story_parser import StoryAnalyzer
from app.services.styles import get_style

ROOT = Path(__file__).resolve().parents[1]
STORY = (
    "Once upon a time, there was a hen who lived on a farm. One day, she met a cow. They became good friends and "
    "explored the farm together. One evening, they helped a lost little rabbit find its home."
)
HALVES = [Region(x0=0, y0=0, x1=0.5, y1=1), Region(x0=0.5, y0=0, x1=1, y1=1)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--style", default="storybook")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "ablation")
    args = ap.parse_args()
    settings = get_settings()
    gen = LocalDiffusersGenerator(settings)
    if not gen.status().reference_conditioning:
        raise SystemExit("IP-Adapter weights are required for this ablation (python scripts/download_models.py)")
    evaluator = ImageEvaluator(settings)
    analysis = StoryAnalyzer(settings).analyze(STORY, 5)
    style = get_style(args.style)
    builder = PromptBuilder(style)
    args.out.mkdir(parents=True, exist_ok=True)

    # anchors
    anchors: dict[str, Path] = {}
    for char in [c for c in analysis.characters if c.is_recurring]:
        built = builder.anchor_prompt(char)
        image, _ = gen.generate(
            GenerationRequest(
                kind="anchor",
                prompt=built.prompt,
                width=1024,
                height=1024,
                seed=11,
                guidance_scale=style.guidance_scale,
            )
        )
        anchors[char.id] = args.out / f"anchor_{char.id}.png"
        image.save(anchors[char.id])
        print(f"anchor {char.name}: {anchors[char.id]}")

    width, height = image_size("4:3", settings.image_resolution)
    scenes = [s for s in analysis.scenes if len([c for c in s.character_ids if c in anchors]) == 2]
    rows, results = [], []
    for mode in ("text-only", "unmasked", "masked"):
        row = []
        for scene in scenes:
            built = builder.scene_prompt(scene, analysis.characters, set(anchors))
            for seed in range(1, args.seeds + 1):
                refs = (
                    []
                    if mode == "text-only"
                    else [
                        ReferenceImage(
                            character_id=s.character_id,
                            path=str(anchors[s.character_id]),
                            region=s.region if mode == "masked" else None,
                        )
                        for s in built.slots
                    ]
                )
                t0 = time.perf_counter()
                image, info = gen.generate(
                    GenerationRequest(
                        kind="scene",
                        prompt=built.prompt,
                        negative_prompt=built.negative_prompt,
                        width=width,
                        height=height,
                        seed=seed,
                        references=refs,
                        ip_adapter_scale=built.ip_adapter_scale,
                        guidance_scale=style.guidance_scale,
                    )
                )
                row.append(image)
                assigned = [IdentityRef(s.character_id, anchors[s.character_id], s.region) for s in built.slots]
                scores = evaluator.evaluate(image, scene.visual_description, assigned)
                best_half = {}
                for s in built.slots:
                    sims = [
                        evaluator.evaluate(
                            image, scene.visual_description, [IdentityRef(s.character_id, anchors[s.character_id], h)]
                        ).consistency[s.character_id]
                        for h in HALVES
                    ]
                    best_half[s.character_id] = max(sims)
                results.append(
                    {
                        "mode": mode,
                        "scene": scene.index,
                        "seed": seed,
                        "reference_mode": info.reference_mode,
                        "consistency_assigned": scores.consistency,
                        "consistency_best_half": best_half,
                        "alignment": scores.alignment,
                        "seconds": round(time.perf_counter() - t0, 1),
                    }
                )
                print(json.dumps(results[-1]))
        rows.append((mode, row))

    summary = {}
    for mode, _ in rows:
        rs = [r for r in results if r["mode"] == mode]
        summary[mode] = {
            "identity_assigned_region": round(
                statistics.mean(v for r in rs for v in r["consistency_assigned"].values()), 4
            ),
            "identity_best_half": round(statistics.mean(v for r in rs for v in r["consistency_best_half"].values()), 4),
            "alignment": round(statistics.mean(r["alignment"] for r in rs), 4),
            "images": len(rs),
        }
    out = {
        "story": STORY,
        "style": args.style,
        "seeds": args.seeds,
        "scenes": [s.index for s in scenes],
        "summary": summary,
        "runs": results,
    }
    (ROOT / "checkpoints" / "ablation_consistency.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))

    # contact sheet: one row per mode
    thumb_w = 320
    thumb_h = int(thumb_w * height / width)
    cols = max(len(r) for _, r in rows)
    sheet = Image.new("RGB", (140 + cols * thumb_w, len(rows) * thumb_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (mode, images) in enumerate(rows):
        draw.text((10, i * thumb_h + thumb_h // 2), mode, fill=(30, 30, 30))
        for j, img in enumerate(images):
            sheet.paste(img.resize((thumb_w, thumb_h)), (140 + j * thumb_w, i * thumb_h))
    docs = ROOT.parent / "docs" / "images"
    docs.mkdir(parents=True, exist_ok=True)
    sheet.save(docs / "ablation_consistency.jpg", quality=85)
    print(f"contact sheet → {docs / 'ablation_consistency.jpg'}")


if __name__ == "__main__":
    main()
