"""Build the weakly-supervised training set for Model A (scene boundary detection) from TinyStories.

Each story is flattened into one paragraph — users usually paste a single block of text — and split into
event units with the production pipeline. Every unit transition gets a *soft* label from:

  * paragraph structure of the original story (the writer's own beat boundaries), and
  * labelling functions: strong temporal cues, place changes with movement, ending cues, dialogue turns.

The model never sees paragraph information as input, so it must learn to recover these boundaries from
the text itself. Output: training/data/processed/scene_{split}.pt

    python -m training.datasets.prepare_scene_data --train 30000 --dev 2000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from app.models.encoders import get_text_encoder
from app.models.scene_segmenter import ENDING_CUES, MOVEMENT, STRONG_TIME_CUES, UnitInfo, unit_features
from app.services.character_manager import CharacterResolver, RuleMentionDetector
from app.services.story_parser import unit_infos
from app.services.text_units import get_nlp, normalize_text, split_units

ROOT = Path(__file__).resolve().parents[1] / "data"
RAW = ROOT / "raw" / "tinystories"
OUT = ROOT / "processed"


def read_stories(path: Path, limit: int, skip: int = 0) -> list[list[str]]:
    """Return stories as lists of paragraphs (TinyStories V2 uses one paragraph per line)."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    stories = []
    for raw in text.split("<|endoftext|>")[skip:]:
        paragraphs = [normalize_text(p) for p in raw.strip().split("\n") if p.strip()]
        words = sum(len(p.split()) for p in paragraphs)
        if len(paragraphs) >= 2 and 40 <= words <= 450:
            stories.append(paragraphs)
        if len(stories) >= limit:
            break
    return stories


def soft_labels(units, infos: list[UnitInfo], par_starts: set[int]) -> tuple[list[float], list[int]]:
    soft, hard = [], []
    current_place = None
    for i, (u, info) in enumerate(zip(units, infos, strict=False)):
        if i == 0:
            if info.places:
                current_place = info.places[-1]
            continue
        text = u.text.lstrip('"').strip().lower()
        par = u.start_char in par_starts
        dialogue_pair = '"' in u.text and '"' in units[i - 1].text
        time_cue = bool(STRONG_TIME_CUES.search(text)) or bool(ENDING_CUES.search(text))
        new_place = bool(info.places) and info.places[-1] != current_place
        place_move = new_place and (info.main_verb_lemma in MOVEMENT or "arriv" in text)
        if par and not dialogue_pair:
            y = 0.92 if (time_cue or place_move) else 0.78
        elif par:
            y = 0.35
        elif time_cue:
            y = 0.7
        elif place_move:
            y = 0.55
        else:
            y = 0.06
        soft.append(y)
        hard.append(int(par and not dialogue_pair))
        if info.places:
            current_place = info.places[-1]
    return soft, hard


def build(stories: list[list[str]], name: str, n_process: int) -> None:
    nlp = get_nlp()
    detector = RuleMentionDetector()
    texts, par_offsets = [], []
    for paragraphs in stories:
        offsets, pos = set(), 0
        for p in paragraphs:
            offsets.add(pos)
            pos += len(p) + 1
        texts.append(" ".join(paragraphs))
        par_offsets.append(offsets)
    records = []
    t0 = time.time()
    for doc, offsets in tqdm(
        zip(nlp.pipe(texts, n_process=n_process, batch_size=64), par_offsets, strict=False),
        total=len(texts),
        desc=f"parse {name}",
    ):
        story = split_units("", doc=doc)
        if not 3 <= len(story.units) <= 60:
            continue
        resolution = CharacterResolver(detector).resolve(story)
        infos = unit_infos(story, resolution)
        soft, hard = soft_labels(story.units, infos, offsets)
        records.append(
            {
                "texts": [u.text for u in story.units],
                "features": unit_features(infos).half(),
                "soft": torch.tensor(soft, dtype=torch.float16),
                "hard": torch.tensor(hard, dtype=torch.int8),
            }
        )
    print(f"{name}: {len(records)} stories parsed in {time.time() - t0:.0f}s")
    encoder = get_text_encoder()
    all_texts = [t for r in records for t in r["texts"]]
    emb = []
    for i in tqdm(range(0, len(all_texts), 2048), desc=f"embed {name}"):
        emb.append(encoder.encode(all_texts[i : i + 2048], batch_size=256).half())
    emb = torch.cat(emb)
    pos = 0
    for r in records:
        n = len(r["texts"])
        r["embeddings"] = emb[pos : pos + n].clone()
        pos += n
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(records, OUT / f"scene_{name}.pt")
    n_pairs = sum(len(r["soft"]) for r in records)
    pos_rate = sum(int(r["hard"].sum()) for r in records) / max(n_pairs, 1)
    print(
        f"saved {OUT / f'scene_{name}.pt'}: {len(records)} stories, {n_pairs} transitions, "
        f"paragraph-boundary rate {pos_rate:.3f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=30000)
    ap.add_argument("--dev", type=int, default=2000)
    ap.add_argument("--processes", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(4)
    build(read_stories(RAW / "TinyStoriesV2-GPT4-valid.txt", args.dev), "dev", args.processes)
    build(read_stories(RAW / "TinyStoriesV2-GPT4-train.head60mb.txt", args.train), "train", args.processes)


if __name__ == "__main__":
    main()
