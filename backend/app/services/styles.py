"""Illustration style catalogue.

Each style is a full set of generation instructions, not a UI label: it changes the prompt lead
and medium descriptors, how characters and lighting are described, the negative prompt (quality
profile), classifier-free guidance, and how strongly character references are applied. Artist names
and studio trademarks are deliberately avoided.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from app.schemas.story import StyleId


@dataclass(frozen=True)
class StyleSpec:
    id: StyleId
    name: str
    tagline: str
    lead: str  # first words of the prompt: the medium (CLIP weights early tokens most)
    medium: str  # technique / rendering descriptors
    character_rendering: str  # how characters should look in this style
    lighting: str  # default lighting when the scene implies none
    palette: str
    anchor_background: str  # background used for character reference sheets
    negative: str  # style-specific negative prompt (used when CFG > 1)
    guidance_scale: float  # CFG for the "quality" profile
    ip_adapter_scale: float  # identity-reference strength for this style

    def public(self) -> dict:
        data = asdict(self)
        return {k: data[k] for k in ("id", "name", "tagline")} | {"preview_url": f"/styles/{self.id}.webp"}


GLOBAL_NEGATIVE = (
    "lowres, blurry, jpeg artifacts, deformed, disfigured, bad anatomy, extra limbs, extra heads, "
    "fused bodies, mutated, duplicate characters, text, letters, watermark, signature, logo, frame, border, nsfw"
)

STYLES: dict[str, StyleSpec] = {
    s.id: s
    for s in [
        StyleSpec(
            id="cartoon",
            name="Cartoon",
            tagline="Bold outlines, bright flat colours",
            lead="cartoon illustration",
            medium="bold clean outlines, flat vibrant colors, simple cel shading, playful rounded shapes",
            character_rendering="cute expressive cartoon characters with big eyes",
            lighting="bright even lighting",
            palette="saturated cheerful colors",
            anchor_background="plain white background",
            negative="photo, photorealistic, 3d render, realistic texture, muted colors, grainy",
            guidance_scale=6.0,
            ip_adapter_scale=0.6,
        ),
        StyleSpec(
            id="realistic",
            name="Realistic",
            tagline="Lifelike, photographic detail",
            lead="photorealistic photograph",
            medium="natural textures, true-to-life detail, shallow depth of field, 50mm lens, high dynamic range",
            character_rendering="lifelike proportions and natural fur, feathers and skin",
            lighting="soft natural daylight",
            palette="natural colors",
            anchor_background="plain light grey studio background",
            negative="cartoon, illustration, drawing, painting, anime, 3d render, cgi, plastic, oversaturated",
            guidance_scale=5.0,
            ip_adapter_scale=0.5,
        ),
        StyleSpec(
            id="anime",
            name="Anime",
            tagline="Clean line art and cel shading",
            lead="anime illustration",
            medium="clean lineart, cel shading, detailed painted background, vibrant colors",
            character_rendering="anime-style characters with large expressive eyes",
            lighting="soft glowing light",
            palette="vivid sky blues and warm accents",
            anchor_background="plain white background",
            negative="photo, realistic, 3d render, western cartoon, sketch, low quality",
            guidance_scale=6.5,
            ip_adapter_scale=0.55,
        ),
        StyleSpec(
            id="watercolor",
            name="Watercolor",
            tagline="Soft washes on textured paper",
            lead="watercolor painting",
            medium="soft wet-on-wet washes, visible paper texture, delicate ink outlines, gentle color bleeding",
            character_rendering="gently painted characters with soft edges",
            lighting="soft diffused light",
            palette="pastel palette",
            anchor_background="plain white paper background",
            negative="photo, 3d render, digital art, hard edges, neon colors, oversaturated",
            guidance_scale=6.0,
            ip_adapter_scale=0.5,
        ),
        StyleSpec(
            id="3d",
            name="3D Animation",
            tagline="Feature-film CGI look",
            lead="3d animated movie still",
            medium="stylized 3d render, soft global illumination, subsurface scattering, smooth materials, "
            "cinematic depth of field",
            character_rendering="appealing stylized 3d characters with big expressive eyes",
            lighting="warm cinematic lighting",
            palette="rich saturated colors",
            anchor_background="plain soft grey studio background",
            negative="2d, flat, sketch, drawing, painting, photo, grainy, lowres",
            guidance_scale=6.0,
            ip_adapter_scale=0.55,
        ),
        StyleSpec(
            id="storybook",
            name="Children's Storybook",
            tagline="Warm, whimsical picture-book art",
            lead="children's picture book illustration",
            medium="soft gouache and colored pencil textures, gentle rounded shapes, whimsical cozy details",
            character_rendering="friendly cute characters",
            lighting="warm soft light",
            palette="warm gentle colors",
            anchor_background="plain cream paper background",
            negative="photo, realistic, 3d render, dark, scary, gritty, harsh shadows",
            guidance_scale=6.0,
            ip_adapter_scale=0.55,
        ),
        StyleSpec(
            id="digital_painting",
            name="Digital Painting",
            tagline="Painterly, atmospheric concept art",
            lead="digital painting",
            medium="painterly brush strokes, rich color, atmospheric depth, detailed concept art",
            character_rendering="expressive painterly characters",
            lighting="dramatic volumetric lighting",
            palette="rich harmonious colors",
            anchor_background="plain neutral background",
            negative="photo, flat colors, cartoon, sketch, lowres, blurry",
            guidance_scale=6.5,
            ip_adapter_scale=0.5,
        ),
        StyleSpec(
            id="comic",
            name="Comic Book",
            tagline="Inked lines, halftone, dynamic panels",
            lead="comic book art",
            medium="bold ink lines, halftone shading, dynamic composition, vivid flat colors",
            character_rendering="expressive comic characters with strong silhouettes",
            lighting="high contrast lighting",
            palette="vivid primary colors",
            anchor_background="plain white background",
            negative="photo, realistic, 3d render, watercolor, speech bubble, text, blurry",
            guidance_scale=6.5,
            ip_adapter_scale=0.55,
        ),
    ]
}


def get_style(style_id: str) -> StyleSpec:
    try:
        return STYLES[style_id]
    except KeyError as exc:  # pragma: no cover - guarded by the StyleId Literal at the API layer
        raise ValueError(f"unknown style {style_id!r}") from exc
