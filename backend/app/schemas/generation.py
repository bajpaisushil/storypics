"""Contract between the orchestration layer and an image generator (local Diffusers or a remote worker)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Region(BaseModel):
    """Normalised rectangle (0–1) of the canvas that a reference image should influence."""

    x0: float = Field(ge=0, le=1)
    y0: float = Field(ge=0, le=1)
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)


class ReferenceImage(BaseModel):
    character_id: str
    # Exactly one of path / data_b64 is set. Local generators read files; remote workers receive base64.
    path: str | None = None
    data_b64: str | None = None
    region: Region | None = Field(None, description="None = condition the whole canvas")


class GenerationRequest(BaseModel):
    kind: Literal["anchor", "scene", "preview"]
    prompt: str
    prompt_2: str | None = Field(None, description="SDXL second text encoder prompt (style/detail)")
    negative_prompt: str = ""
    width: int = Field(ge=256, le=2048)
    height: int = Field(ge=256, le=2048)
    seed: int = Field(ge=0, le=2**32 - 1)
    references: list[ReferenceImage] = Field(default_factory=list)
    ip_adapter_scale: float = Field(0.6, ge=0, le=1.5)
    # Optional overrides; generators fall back to their profile defaults.
    steps: int | None = Field(None, ge=1, le=100)
    guidance_scale: float | None = Field(None, ge=0, le=20)


class GenerationInfo(BaseModel):
    seed: int
    steps: int
    guidance_scale: float
    width: int
    height: int
    profile: str
    device: str
    model: str
    reference_mode: Literal["none", "single", "masked", "unmasked"]
    duration_s: float
    peak_memory_gb: float | None = None


class GeneratorStatus(BaseModel):
    backend: Literal["local", "remote", "disabled"]
    available: bool
    loaded: bool
    device: str | None = None
    profile: str | None = None
    model: str | None = None
    reason: str | None = Field(None, description="why the generator is unavailable, with the fix")
    reference_conditioning: bool = Field(True, description="IP-Adapter character references are active")
    notice: str | None = Field(None, description="non-fatal limitation, e.g. IP-Adapter weights still missing")
