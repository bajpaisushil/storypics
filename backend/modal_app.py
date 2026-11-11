"""Deploy the StoryPix backend on Modal: FastAPI + the four custom models + SDXL-Lightning on a GPU.

    modal setup                                             # once: log in to your Modal account (browser)
    cd backend && modal deploy modal_app.py                 # GPU (T4): SDXL-Lightning — needs a card on file
    STORYPIX_MODAL_GPU=none modal deploy modal_app.py       # CPU only: LCM-DreamShaper, 512 px, slower

The container image is built on Modal and the model weights (~12 GB) download into a Modal Volume on first start,
so nothing large is stored on your machine. The GPU container scales to zero after a few idle minutes and is
billed per second only while it runs (an L4 is about $0.80/hour), which keeps a demo inside the free plan's
monthly credits. Set a workspace budget in the Modal dashboard as a safety net.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
# "T4"/"L4"/…: SDXL-Lightning on a GPU (Modal requires a payment method on file for GPUs, even on free credits).
# "none": CPU-only container running the LCM-DreamShaper profile (slower, 512 px), allowed without a card.
GPU = os.environ.get("STORYPIX_MODAL_GPU", "T4")
CPU_ONLY = GPU.lower() == "none"
IDLE_SECONDS = 240  # scale to zero after 4 idle minutes

hf_cache = modal.Volume.from_name("storypix-hf-cache", create_if_missing=True)  # model weights
data = modal.Volume.from_name("storypix-data", create_if_missing=True)  # jobs database + generated images

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libglib2.0-0", "libgomp1")
    .uv_pip_install(requirements=[str(ROOT / "requirements.txt")])
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "STORYPIX_DATA_DIR": "/data",
            "STORYPIX_MODAL_GPU": GPU,  # same choice inside the container when this module is re-imported
            "STORYPIX_DEVICE": "cpu" if CPU_ONLY else "cuda",
            "STORYPIX_PROFILE": "lcm" if CPU_ONLY else "lightning",
            "STORYPIX_LIGHTNING_STEPS": "8",
            "STORYPIX_ALLOW_DOWNLOADS": "true",
            "STORYPIX_PRELOAD_MODELS": "true",
            "STORYPIX_SAFETY_CHECKER": "true",
            "STORYPIX_MAX_ACTIVE_JOBS": "2",
            "STORYPIX_CORS_ORIGINS": '["http://localhost:3000","http://127.0.0.1:3000"]',
            # no backslashes: Modal's image builder treats them as escapes ([.] matches a literal dot)
            "STORYPIX_CORS_ORIGIN_REGEX": "https://storypics[a-z0-9-]*[.]vercel[.]app",
        }
    )
    .add_local_dir(ROOT / "app", "/root/app", ignore=["**/__pycache__/**"])
    .add_local_dir(ROOT / "checkpoints", "/root/checkpoints")
)

app = modal.App("storypix", image=image)


@app.cls(
    gpu=None if CPU_ONLY else GPU,
    cpu=8 if CPU_ONLY else 4,  # CPU-only: diffusion runs on these cores; GPU: language models + evaluator
    memory=16384,
    volumes={"/cache": hf_cache, "/data": data},
    scaledown_window=IDLE_SECONDS,
    max_containers=1,  # job state lives in one process; one GPU worker draws images in order
    timeout=3600,  # long-lived progress streams; the frontend reconnects if one is cut
)
@modal.concurrent(max_inputs=200)
class StoryPixServer:
    @modal.enter()
    def start(self) -> None:
        self._stop = threading.Event()
        threading.Thread(target=self._commit_loop, daemon=True).start()

    def _commit_loop(self) -> None:
        # Persist new storybooks and downloaded weights so the next cold start sees them.
        while not self._stop.wait(30):
            self._commit()

    @staticmethod
    def _commit() -> None:
        for volume in (data, hf_cache):
            try:
                volume.commit()
            except Exception as exc:  # a failed commit is retried on the next tick
                print(f"volume commit failed: {exc}")

    @modal.exit()
    def stop(self) -> None:
        self._stop.set()
        self._commit()

    @modal.asgi_app()
    def web(self):
        from app.main import create_app

        return create_app()
