"""StoryPix generation worker — runs the Diffusers backend on a GPU machine you control.

    STORYPIX_WORKER_TOKEN=change-me uvicorn worker.server:app --host 0.0.0.0 --port 8001

The web backend then uses it with STORYPIX_GENERATOR=remote, STORYPIX_REMOTE_GENERATOR_URL=http://<host>:8001
and the same STORYPIX_WORKER_TOKEN. Requests and responses use the GenerationRequest / GenerationInfo schemas,
with reference images and the result inlined as base64 PNG.
"""

from __future__ import annotations

import asyncio
import base64
import io
import secrets

from fastapi import Depends, FastAPI, Header, HTTPException

from app.config import get_settings
from app.schemas.generation import GenerationRequest
from app.services.image_generator import GeneratorUnavailable, LocalDiffusersGenerator

settings = get_settings()
generator = LocalDiffusersGenerator(settings)
app = FastAPI(title="StoryPix generation worker", version="1.0.0")
_gpu = asyncio.Lock()


def authorize(authorization: str | None = Header(None)) -> None:
    token = settings.worker_token
    if token and not (authorization and secrets.compare_digest(authorization, f"Bearer {token}")):
        raise HTTPException(status_code=401, detail="invalid worker token")


@app.get("/v1/health", dependencies=[Depends(authorize)])
async def health() -> dict:
    return generator.status().model_dump()


@app.post("/v1/generate", dependencies=[Depends(authorize)])
async def generate(req: GenerationRequest) -> dict:
    async with _gpu:  # one generation at a time per GPU
        try:
            image, info = await asyncio.to_thread(generator.generate, req)
        except GeneratorUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {"image_b64": base64.b64encode(buf.getvalue()).decode(), "info": info.model_dump()}
