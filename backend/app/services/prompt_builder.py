"""Stage 7 — scene + characters + style (+ user instructions) → image-generation request parameters.

Ordering matters: CLIP attends most to early tokens and SDXL's pooled conditioning comes from the first
77-token window, so the prompt leads with the medium and the action, then the canonical character
descriptors, setting, lighting, composition and style details. Longer prompts are encoded in 77-token
chunks by the generator (see image_generator.encode_long_prompt) instead of being silently truncated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.schemas.generation import Region
from app.schemas.story import CharacterProfile, Scene
from app.services import appearance as kb
from app.services.styles import GLOBAL_NEGATIVE, StyleSpec

# canvas regions for masked multi-reference conditioning, with a small overlap between neighbours
LAYOUTS: dict[int, list[tuple[Region, str]]] = {
    1: [(Region(x0=0, y0=0, x1=1, y1=1), "")],
    2: [(Region(x0=0.0, y0=0, x1=0.56, y1=1), "on the left"), (Region(x0=0.44, y0=0, x1=1.0, y1=1), "on the right")],
    3: [
        (Region(x0=0.0, y0=0, x1=0.38, y1=1), "on the left"),
        (Region(x0=0.31, y0=0, x1=0.69, y1=1), "in the middle"),
        (Region(x0=0.62, y0=0, x1=1.0, y1=1), "on the right"),
    ],
}
MAX_REFERENCED = 3
SIZES = {"1:1": (1024, 1024), "4:3": (1152, 896), "16:9": (1344, 768), "9:16": (768, 1344)}
IMPERATIVES = re.compile(
    r"^(please\s+)?(make|show|draw|depict|add|give|let|have|put|change|turn|render|paint|"
    r"include|place)\s+",
    re.I,
)


# SD1.5-family models (CPU "lcm" profile) are trained around 512 px; sizes are multiples of 8.
SD15_SIZES = {"1:1": (512, 512), "4:3": (640, 480), "16:9": (768, 432), "9:16": (432, 768)}


def image_size(aspect_ratio: str, resolution: int = 1024, profile: str | None = None) -> tuple[int, int]:
    """Native canvas for the model family. SDXL: its ~1-megapixel bucket scaled so resolution² ≈ pixel count."""
    if profile == "lcm":
        return SD15_SIZES[aspect_ratio]
    w, h = SIZES[aspect_ratio]
    scale = resolution / 1024
    return int(round(w * scale / 64) * 64), int(round(h * scale / 64) * 64)


@dataclass
class CharacterSlot:
    character_id: str
    region: Region | None
    position: str


@dataclass
class BuiltPrompt:
    prompt: str
    negative_prompt: str
    slots: list[CharacterSlot] = field(default_factory=list)
    ip_adapter_scale: float = 0.6


def rewrite_instruction(instruction: str) -> str:
    """'Make the hen happier' → 'the hen very happy'; 'Show the cow near the tree' → 'the cow near the tree'."""
    text = instruction.strip().rstrip(".!")
    text = IMPERATIVES.sub("", text)
    text = re.sub(r"^(it|this|the image|the picture|the scene)\s+(look\s+|be\s+)?", "", text, flags=re.I)
    comparatives = {
        "happier": "happy",
        "sadder": "sad",
        "bigger": "big",
        "smaller": "small",
        "brighter": "bright",
        "darker": "dark",
        "cuter": "cute",
        "friendlier": "friendly",
        "scarier": "spooky",
        "calmer": "calm",
        "older": "old",
        "younger": "young",
        "taller": "tall",
        "warmer": "warm",
        "funnier": "funny",
        "closer": "close",
        "sunnier": "sunny",
        "more colorful": "colorful",
    }
    for comp, base in comparatives.items():
        text = re.sub(rf"\b{comp}\b", f"very {base}", text, flags=re.I)
    text = re.sub(r"\bmore (\w+)", r"very \1", text)
    return text.strip()


class PromptBuilder:
    def __init__(self, style: StyleSpec):
        self.style = style

    def _names(self, chars: list[CharacterProfile]) -> dict[str, str]:
        return {c.id: (c.name if c.name.lower() != c.species else f"the {c.species}") for c in chars}

    def scene_prompt(
        self, scene: Scene, characters: list[CharacterProfile], anchored: set[str], instructions: str | None = None
    ) -> BuiltPrompt:
        s = self.style
        present = [c for c in characters if c.id in scene.character_ids]
        present.sort(key=lambda c: scene.character_ids.index(c.id))
        referenced = [c for c in present if c.id in anchored][:MAX_REFERENCED]
        layout = LAYOUTS.get(len(referenced), [])
        slots = [
            CharacterSlot(c.id, region if len(referenced) > 1 else None, position)
            for c, (region, position) in zip(referenced, layout, strict=False)
        ]
        position_of = {slot.character_id: slot.position for slot in slots}

        parts: list[str] = [s.lead]
        if instructions:
            parts.append(rewrite_instruction(instructions))
        if scene.visual_description:
            parts.append(scene.visual_description)
        for c in present:
            label = self._names([c])[c.id]
            desc = c.descriptor
            where = position_of.get(c.id, "")
            expression = c.personality[0] if c.personality else ""
            detail = ", ".join(x for x in (desc, expression, where) if x)
            parts.append(f"{label}: {detail}" if label.lower() not in desc.lower()[: len(label) + 4] else detail)
        if scene.location:
            parts.append(f"setting: {scene.location.description}")
        lighting = kb.TIME_OF_DAY.get(scene.time_of_day or "", s.lighting)
        parts.append(lighting)
        if scene.weather:
            parts.append(kb.WEATHER.get(scene.weather, scene.weather))
        if scene.composition:
            parts.append(scene.composition)
        if scene.mood:
            parts.append(scene.mood)
        parts += [s.medium, s.character_rendering, s.palette]
        if len(present) <= 3 and present:
            parts.append(
                f"only {len(present)} character{'s' if len(present) > 1 else ''}"
                if len(present) > 1
                else "single character"
            )
        prompt = _clean(", ".join(p for p in parts if p))
        scale = s.ip_adapter_scale + (0.1 if len(slots) > 1 else 0.0)
        return BuiltPrompt(prompt=prompt, negative_prompt=self.negative(), slots=slots, ip_adapter_scale=scale)

    def anchor_prompt(self, character: CharacterProfile, from_upload: bool = False) -> BuiltPrompt:
        s = self.style
        expression = character.personality[0] if character.personality else "friendly expression"
        parts = [
            s.lead,
            "character reference sheet",
            f"full body portrait of {character.descriptor}",
            expression,
            "standing, facing the viewer, centered, whole body visible",
            s.anchor_background,
            s.medium,
            s.character_rendering,
            s.palette,
            "single character, simple clean composition",
        ]
        slots = [CharacterSlot(character.id, None, "")] if from_upload else []
        return BuiltPrompt(
            prompt=_clean(", ".join(parts)),
            negative_prompt=self.negative(),
            slots=slots,
            ip_adapter_scale=0.55 if from_upload else 0.0,
        )

    def preview_prompt(self) -> BuiltPrompt:
        s = self.style
        prompt = (
            f"{s.lead}, a friendly little fox reading a book under a big oak tree in a flowery meadow, "
            f"{s.lighting}, {s.medium}, {s.character_rendering}, {s.palette}"
        )
        return BuiltPrompt(prompt=_clean(prompt), negative_prompt=self.negative())

    def negative(self) -> str:
        return f"{self.style.negative}, {GLOBAL_NEGATIVE}"


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*,+", ",", text)
    return text.strip(" ,")
