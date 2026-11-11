"""Story understanding orchestrator: text → units → characters → scene boundaries → scenes + profiles."""

from __future__ import annotations

import itertools
import logging
import time
from pathlib import Path

import torch

from app.config import Settings
from app.models.encoders import TextEncoder, get_text_encoder
from app.models.scene_segmenter import (
    BoundaryScorer,
    HeuristicBoundaryScorer,
    LearnedBoundaryScorer,
    UnitInfo,
    decode_boundaries,
)
from app.schemas.story import AnalysisMeta, CharacterProfile, StoryAnalysis
from app.services.character_manager import (
    CharacterResolution,
    CharacterResolver,
    MentionDetector,
    RuleMentionDetector,
    TaggerMentionDetector,
    build_profile,
)
from app.services.scene_builder import SceneDraft, build_scenes
from app.services.text_units import ProcessedStory, get_nlp, split_units

log = logging.getLogger(__name__)

SEGMENTER_CKPT = "scene_segmenter.pt"
TAGGER_CKPT = "character_tagger.pt"
PARAGRAPH_PRIOR = 0.6
SHORT_STORY_UNITS = 6


def _depictable(units, i: int) -> bool:
    unit, prev = units[i], units[i - 1]
    if '"' in unit.text and '"' in prev.text:
        return False
    roots = [t for t in unit.tokens if t.pos_ in ("VERB", "AUX") and (t.head not in unit.tokens or t.head == t)]
    if not roots:
        return False
    root = roots[0]
    reflective = (
        root.lemma_ in ("be", "feel", "seem")
        and any(c.dep_ == "acomp" for c in root.children)
        and not any(c.dep_ in ("attr", "dobj", "prep") for c in root.children)
    )
    return not reflective


def unit_infos(story: ProcessedStory, resolution: CharacterResolution) -> list[UnitInfo]:
    infos = []
    for u in story.units:
        verbs = [t for t in u.tokens if t.pos_ in ("VERB", "AUX") and (t.head not in u.tokens or t.head == t)]
        infos.append(
            UnitInfo(
                text=u.text,
                is_continuation=u.is_continuation,
                characters=resolution.unit_characters[u.index],
                places=resolution.unit_places[u.index],
                main_verb_lemma=verbs[0].lemma_.lower() if verbs else None,
                lemmas={t.lemma_.lower() for t in u.tokens},
            )
        )
    return infos


class StoryAnalyzer:
    def __init__(self, settings: Settings, encoder: TextEncoder | None = None):
        self.settings = settings
        get_nlp(settings.spacy_model)
        self.encoder = encoder or get_text_encoder(settings.text_encoder, local_files_only=not settings.allow_downloads)
        ckpt_dir: Path = settings.checkpoints_dir
        self.detector: MentionDetector
        if (ckpt_dir / TAGGER_CKPT).exists():
            self.detector = TaggerMentionDetector(ckpt_dir / TAGGER_CKPT, self.encoder)
        else:
            log.warning("No %s checkpoint — using rule-based mention detection", TAGGER_CKPT)
            self.detector = RuleMentionDetector()
        self.scorer: BoundaryScorer
        if (ckpt_dir / SEGMENTER_CKPT).exists():
            self.scorer = LearnedBoundaryScorer(ckpt_dir / SEGMENTER_CKPT)
        else:
            log.warning("No %s checkpoint — using heuristic scene segmentation", SEGMENTER_CKPT)
            self.scorer = HeuristicBoundaryScorer()

    def analyze(self, text: str, scene_count: int | str = "auto", seed_key: str = "") -> StoryAnalysis:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        story = split_units(text)
        timings["parse_ms"] = (time.perf_counter() - t0) * 1000
        warnings: list[str] = []
        if not story.units:
            raise ValueError("The story has no readable sentences.")

        t1 = time.perf_counter()
        resolution = CharacterResolver(self.detector).resolve(story)
        timings["characters_ms"] = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        infos = unit_infos(story, resolution)
        with torch.inference_mode():
            embeddings = self.encoder.encode([u.text for u in story.units])
        probs = self.scorer.boundary_probs(infos, embeddings)
        # The writer's own paragraph breaks are strong evidence of a scene change (noisy-OR prior).
        for i in range(1, len(story.units)):
            if story.units[i].paragraph_index != story.units[i - 1].paragraph_index:
                probs[i - 1] = probs[i - 1] + (1 - probs[i - 1]) * PARAGRAPH_PRIOR
        threshold = getattr(self.scorer, "threshold", 0.5)
        starts = decode_boundaries(
            probs, scene_count, story.word_count, threshold=threshold, max_scenes=self.settings.max_scenes
        )
        if scene_count == "auto" and len(story.units) <= SHORT_STORY_UNITS:
            # Product rule for very short stories: every depictable event gets its own picture; only
            # reflective states ("She was happy.") and dialogue continuations stay with the previous unit.
            starts = [i for i in range(1, len(story.units)) if i in starts or _depictable(story.units, i)]
        timings["segmentation_ms"] = (time.perf_counter() - t2) * 1000
        if scene_count != "auto" and int(scene_count) > len(story.units):
            warnings.append(
                f"The story contains {len(story.units)} distinct moments, so {len(story.units)} scenes were "
                f"created instead of {scene_count}. Add more events to the story for more scenes."
            )

        bounds = [0, *starts, len(story.units)]
        drafts = [
            SceneDraft(units=story.units[a:b], boundary_confidence=(probs[a - 1] if a > 0 else None))
            for a, b in itertools.pairwise(bounds)
        ]

        t3 = time.perf_counter()
        scene_of_unit = {u.index: si + 1 for si, d in enumerate(drafts) for u in d.units}
        profiles: dict[str, CharacterProfile] = {}
        for cluster in resolution.clusters:
            scene_ids = sorted({scene_of_unit[u] for u in cluster.units if u in scene_of_unit})
            profiles[cluster.id] = build_profile(cluster, seed_key or text[:200], scene_ids)
        scenes = build_scenes(story, drafts, resolution, profiles)
        # a character that appears in only one scene but is the story's protagonist still gets an anchor
        if profiles:
            lead = max(profiles.values(), key=lambda p: p.mention_count)
            if lead.mention_count >= 2:
                lead.is_recurring = True
        timings["scenes_ms"] = (time.perf_counter() - t3) * 1000
        if not profiles:
            warnings.append("No characters were detected; scenes will be illustrated from the text alone.")

        ordered = sorted(profiles.values(), key=lambda p: (-int(p.is_recurring), int(p.id[1:])))
        return StoryAnalysis(
            title=story_title(ordered, scenes[0].summary if scenes else text),
            scenes=scenes,
            characters=ordered,
            units=[u.schema() for u in story.units],
            boundary_probabilities=[round(p, 4) for p in probs],
            warnings=warnings,
            meta=AnalysisMeta(
                segmenter=self.scorer.name,
                tagger=getattr(self.detector, "name", "rules"),
                unit_count=len(story.units),
                word_count=story.word_count,
                requested_scenes=scene_count if scene_count == "auto" else int(scene_count),
                timings_ms={k: round(v, 1) for k, v in timings.items()},
            ),
        )


def story_title(characters: list[CharacterProfile], fallback: str) -> str:
    main = [c for c in characters if c.category != "group"][:2]
    if not main:
        words = fallback.split()
        return " ".join(words[:6]).rstrip(".,!?") or "My Story"

    def label(c: CharacterProfile) -> str:
        return c.name if c.name.lower() != c.species else f"the {c.name}"

    names = [label(c) for c in main]
    title = " and ".join(names)
    return title[:1].upper() + title[1:]
