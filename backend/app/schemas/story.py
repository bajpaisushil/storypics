"""Structured outputs of the story-understanding stages (JSON contract between pipeline stages)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

StyleId = Literal["cartoon", "realistic", "anime", "watercolor", "3d", "storybook", "digital_painting", "comic"]
AspectRatio = Literal["1:1", "4:3", "16:9", "9:16"]
CharacterCategory = Literal["human", "animal", "creature", "object", "group"]
Gender = Literal["female", "male", "unknown"]


class EventUnit(BaseModel):
    """A clause-level narrative unit; the atoms that scene segmentation groups together."""

    index: int
    text: str
    start: int
    end: int
    sentence_index: int
    paragraph_index: int
    is_continuation: bool = Field(description="unit was split off a coordinated clause of the previous unit")


class CharacterMention(BaseModel):
    text: str
    start: int
    end: int
    unit_index: int
    kind: Literal["name", "nominal", "pronoun"]
    source: Literal["tagger", "rules", "resolver"]


class CharacterProfile(BaseModel):
    id: str
    name: str = Field(description="display name, e.g. 'Hen' or 'Lily'")
    species: str = Field(description="what the character is, e.g. 'hen', 'girl', 'robot'")
    category: CharacterCategory
    gender: Gender = "unknown"
    attributes: dict[str, str] = Field(default_factory=dict, description="visual attributes, ordered")
    explicit_traits: list[str] = Field(default_factory=list, description="adjectives found in the story text")
    personality: list[str] = Field(default_factory=list)
    descriptor: str = Field(description="canonical prompt phrase reused in every scene")
    mention_count: int = 0
    mentions: list[CharacterMention] = Field(default_factory=list)
    scene_indices: list[int] = Field(default_factory=list)
    is_recurring: bool = Field(description="appears in 2+ scenes or is a main character → gets an anchor image")
    reference_upload_id: str | None = None


class SceneLocation(BaseModel):
    name: str
    description: str
    carried_over: bool = Field(description="not mentioned in this scene; inherited from earlier scenes")


class SceneAction(BaseModel):
    actor_ids: list[str]
    verb: str
    phrase: str = Field(description="gerund phrase from the story, e.g. 'meeting a cow'")
    short_phrase: str = Field("", description="verb + first object, used for titles")
    target_ids: list[str] = Field(default_factory=list)
    subject_text: str | None = Field(None, description="non-character subject, e.g. 'the wooden chest'")


class Scene(BaseModel):
    index: int = Field(ge=1)
    title: str
    text: str = Field(description="original story text covered by this scene")
    summary: str = Field(description="standalone, pronoun-resolved description")
    unit_indices: list[int]
    character_ids: list[str]
    actions: list[SceneAction] = Field(default_factory=list)
    location: SceneLocation | None = None
    time_of_day: str | None = None
    weather: str | None = None
    emotion: str = "neutral"
    mood: str = ""
    composition: str = ""
    visual_description: str = Field("", description="depictable description used to build the image prompt")
    previous_context: str | None = None
    boundary_confidence: float | None = Field(None, description="P(scene starts here) from the boundary model")


class AnalysisMeta(BaseModel):
    segmenter: str
    tagger: str
    unit_count: int
    word_count: int
    requested_scenes: int | Literal["auto"]
    timings_ms: dict[str, float] = Field(default_factory=dict)


class StoryAnalysis(BaseModel):
    title: str
    scenes: list[Scene]
    characters: list[CharacterProfile]
    units: list[EventUnit]
    boundary_probabilities: list[float] = Field(
        default_factory=list, description="P(boundary before unit i) for i = 1..n-1"
    )
    warnings: list[str] = Field(default_factory=list)
    meta: AnalysisMeta


class AnalyzeRequest(BaseModel):
    story: str = Field(min_length=20)
    scene_count: int | Literal["auto"] = "auto"
    style: StyleId = "storybook"
