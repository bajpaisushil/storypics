"""Application settings, loaded from environment variables (prefix ``STORYPIX_``) or ``backend/.env``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STORYPIX_", env_file=BACKEND_ROOT / ".env", extra="ignore")

    # --- storage -----------------------------------------------------------------------------
    data_dir: Path = BACKEND_ROOT / "data"
    checkpoints_dir: Path = BACKEND_ROOT / "checkpoints"

    # --- web ---------------------------------------------------------------------------------
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    # e.g. "https://storypics.*\\.vercel\\.app" to allow a Vercel deployment and its preview URLs
    cors_origin_regex: str | None = None
    # Public demos: require this key (header X-StoryPix-Key) for anything that starts work or changes data,
    # and cap how many jobs may be queued/running at once (0 = unlimited).
    access_key: str | None = None
    max_active_jobs: int = Field(default=0, ge=0)
    # Free compute included in the hosting plan each month (Modal Starter: $30), shown as a meter in the web app.
    monthly_credit_usd: float = 30.0
    max_story_chars: int = 6000
    max_upload_mb: int = 10
    max_scenes: int = 12

    # --- image generation -------------------------------------------------------------------
    # local  = run Diffusers in this process (MPS/CUDA/CPU)
    # remote = send generation requests to a StoryPix worker (worker/server.py) on a GPU machine
    # disabled = analysis only; job creation returns 503 with an explanation
    generator: Literal["local", "remote", "disabled"] = "local"
    remote_generator_url: str | None = None
    worker_token: str | None = None
    device: Literal["auto", "mps", "cuda", "cpu"] = "auto"
    # lightning = SDXL + SDXL-Lightning LoRA (4/8 steps, no CFG) — default on Apple silicon
    # quality   = SDXL, 28 steps, CFG + negative prompts — default on CUDA GPUs
    # lcm       = LCM-DreamShaper v7 (SD1.5 family, 4 steps, 512 px) — default on CPU-only servers
    profile: Literal["auto", "lightning", "quality", "lcm"] = "auto"
    lightning_steps: int = 8  # 4 or 8 (the published SDXL-Lightning LoRAs)
    quality_steps: int = 28
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    base_variant: str | None = "fp16"
    vae_model: str | None = "madebyollin/sdxl-vae-fp16-fix"  # None → the base model's own VAE
    lightning_repo: str = "ByteDance/SDXL-Lightning"
    ip_adapter_repo: str = "h94/IP-Adapter"
    ip_adapter_weight: str = "ip-adapter-plus_sdxl_vit-h.safetensors"
    ip_adapter_sd15_weight: str = "ip-adapter-plus_sd15.safetensors"
    lcm_model: str = "SimianLuo/LCM_Dreamshaper_v7"  # MIT licence
    lcm_steps: int = Field(default=4, ge=1, le=12)
    lcm_guidance: float = 8.0
    safety_checker: bool = True  # discard images the Stable Diffusion safety checker flags (when its weights exist)
    torch_threads: int = 0  # CPU threads for PyTorch (0 = library default)
    enable_ip_adapter: bool = True  # character reference conditioning (needs the IP-Adapter weights)
    # Never download multi-GB weights implicitly at request time; run scripts/download_models.py.
    allow_downloads: bool = False
    preload_models: bool = False
    attention_slicing: bool = False
    # ≈ sqrt(pixels): 1024 uses SDXL's native ~1 MP buckets (e.g. 1344×768 for 16:9); 768 is ~2× faster
    image_resolution: int = Field(default=1024, ge=512, le=1536)

    # --- evaluation / reranking ----------------------------------------------------------------
    enable_evaluator: bool = True
    candidates_per_scene: int = Field(default=1, ge=1, le=4)
    auto_retry: bool = True
    max_retries: int = Field(default=1, ge=0, le=3)
    retry_below_score: float = 0.45

    # --- NLP ---------------------------------------------------------------------------------
    text_encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    spacy_model: str = "en_core_web_sm"

    @field_validator("lightning_steps")
    @classmethod
    def _lightning_steps(cls, v: int) -> int:
        # an int field (not Literal[4, 8]) so values from environment variables, which arrive as text, validate
        if v not in (4, 8):
            raise ValueError("lightning_steps must be 4 or 8")
        return v

    @property
    def media_dir(self) -> Path:
        return self.data_dir / "media"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "storypix.sqlite3"


@lru_cache
def get_settings() -> Settings:
    return Settings()
