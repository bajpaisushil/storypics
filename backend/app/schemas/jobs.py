"""Job / scene state exposed by the API and persisted in SQLite."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, computed_field, field_validator

from app.schemas.story import AnalysisMeta, AspectRatio, CharacterProfile, Scene, StyleId

JobStatus = Literal["queued", "generating", "completed", "partial", "failed", "cancelled", "interrupted"]
TaskStatus = Literal["pending", "queued", "generating", "completed", "failed", "cancelled", "skipped"]
VersionMode = Literal["initial", "keep_composition", "new_variation", "retry", "upload"]
SceneCountChoice = int | Literal["auto"]


def utcnow() -> datetime:
    return datetime.now(UTC)


class ImageScores(BaseModel):
    """All scores are in [0, 1]; None means the evaluator was unavailable."""

    alignment: float | None = Field(None, description="Model C: does the image depict the scene text?")
    quality: float | None = Field(None, description="Model D: predicted human fidelity rating")
    consistency: dict[str, float] = Field(
        default_factory=dict, description="character id → DINOv2 identity similarity to its anchor"
    )
    overall: float | None = None


class ImageVersion(BaseModel):
    id: str
    url: str
    thumb_url: str
    file: str = Field(description="path relative to the media directory")
    created_at: datetime = Field(default_factory=utcnow)
    mode: VersionMode
    seed: int
    prompt: str
    prompt_2: str | None = None
    negative_prompt: str = ""
    instructions: str | None = None
    scores: ImageScores | None = None
    duration_s: float = 0.0
    attempts: int = 1
    generation: dict = Field(default_factory=dict, description="GenerationInfo of the kept candidate")


class TaskProgress(BaseModel):
    kind: Literal["anchor", "scene"]
    target: str = Field(description="character id or scene index")
    label: str
    step: int = 0
    total_steps: int = 0
    attempt: int = 1


class SceneState(Scene):
    status: TaskStatus = "pending"
    versions: list[ImageVersion] = Field(default_factory=list)
    selected_version: str | None = None
    error: str | None = None

    def selected(self) -> ImageVersion | None:
        return next((v for v in self.versions if v.id == self.selected_version), None)


class CharacterState(CharacterProfile):
    anchor_status: TaskStatus = "skipped"
    anchor_versions: list[ImageVersion] = Field(default_factory=list)
    selected_anchor: str | None = None
    anchor_error: str | None = None

    def anchor(self) -> ImageVersion | None:
        return next((v for v in self.anchor_versions if v.id == self.selected_anchor), None)


class JobSettings(BaseModel):
    style: StyleId
    aspect_ratio: AspectRatio
    scene_count: SceneCountChoice
    seed: int
    width: int
    height: int
    profile: str = ""


class Job(BaseModel):
    id: str
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    status: JobStatus = "queued"
    title: str
    story: str
    settings: JobSettings
    characters: list[CharacterState]
    scenes: list[SceneState]
    current: TaskProgress | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    analysis: AnalysisMeta

    @computed_field  # type: ignore[prop-decorator]
    @property
    def progress(self) -> float:
        tasks = [c.anchor_status for c in self.characters if c.anchor_status != "skipped"]
        tasks += [s.status for s in self.scenes]
        if not tasks:
            return 0.0
        done = sum(1 for t in tasks if t in ("completed", "failed", "cancelled"))
        return done / len(tasks)

    def scene(self, index: int) -> SceneState:
        for s in self.scenes:
            if s.index == index:
                return s
        raise KeyError(index)

    def character(self, character_id: str) -> CharacterState:
        for c in self.characters:
            if c.id == character_id:
                return c
        raise KeyError(character_id)


class JobSummary(BaseModel):
    id: str
    title: str
    status: JobStatus
    style: StyleId
    scene_count: int
    created_at: datetime
    cover_url: str | None = None


class ReferenceAssignment(BaseModel):
    upload_id: str = Field(pattern=r"^[a-f0-9]{16,32}$")
    character_name: str = Field(min_length=1, max_length=60, description="e.g. 'Hen' or 'Lily'")


class CreateJobRequest(BaseModel):
    story: str = Field(min_length=20)
    style: StyleId = "storybook"
    scene_count: SceneCountChoice = "auto"
    aspect_ratio: AspectRatio = "4:3"
    references: list[ReferenceAssignment] = Field(default_factory=list, max_length=6)
    seed: int | None = Field(None, ge=0, le=2**31 - 1)
    title: str | None = Field(None, max_length=120)

    @field_validator("scene_count")
    @classmethod
    def _scene_count_range(cls, v: SceneCountChoice) -> SceneCountChoice:
        if v != "auto" and not 1 <= v <= 12:
            raise ValueError("scene_count must be 'auto' or between 1 and 12")
        return v


class RegenerateSceneRequest(BaseModel):
    instructions: str | None = Field(None, max_length=300)
    mode: Literal["keep_composition", "new_variation"] = "new_variation"


class SelectVersionRequest(BaseModel):
    version_id: str


class UpdateCharacterRequest(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=60)
    descriptor: str | None = Field(None, min_length=3, max_length=300)
    attributes: dict[str, str] | None = None
    regenerate: bool = True


class UploadResponse(BaseModel):
    upload_id: str
    url: str
    width: int
    height: int
