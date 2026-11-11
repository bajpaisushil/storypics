"""Shared fixtures. The FakeGenerator below is a TEST DOUBLE used only by the test-suite: it draws flat
coloured placeholder images so the job workflow can be tested without a GPU. It is never used by the app."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.config import Settings
from app.schemas.generation import GenerationInfo, GenerationRequest, GeneratorStatus
from app.services.story_parser import StoryAnalyzer

HEN_STORY = (
    "Once upon a time, there was a hen who lived on a farm. One day, she met a cow. They became good "
    "friends and explored the farm together. One evening, they helped a lost little rabbit find its home."
)


class FakeGenerator:
    """Test double — deterministic flat images, records every request it receives."""

    def __init__(self, fail_kinds: set[str] | None = None, delay: float = 0.0):
        self.requests: list[GenerationRequest] = []
        self.fail_kinds = fail_kinds or set()
        self.delay = delay
        self.loaded = False

    def status(self) -> GeneratorStatus:
        return GeneratorStatus(
            backend="local", available=True, loaded=self.loaded, device="cpu", profile="test-double", model="fake"
        )

    def load(self) -> None:
        self.loaded = True

    def generate(self, request: GenerationRequest, progress=None, cancel: threading.Event | None = None):
        self.requests.append(request)
        if request.kind in self.fail_kinds:
            raise RuntimeError(f"simulated {request.kind} failure")
        for step in range(1, 3):
            if self.delay:
                time.sleep(self.delay)
            if progress:
                progress(step, 2)
        img = Image.new("RGB", (request.width // 8, request.height // 8), (request.seed % 255, 120, 180))
        ImageDraw.Draw(img).text((4, 4), request.kind, fill=(255, 255, 255))
        info = GenerationInfo(
            seed=request.seed,
            steps=2,
            guidance_scale=0,
            width=request.width,
            height=request.height,
            profile="test-double",
            device="cpu",
            model="fake",
            reference_mode="none" if not request.references else "single",
            duration_s=0.01,
        )
        return img, info


@pytest.fixture(scope="session")
def analyzer(tmp_path_factory) -> StoryAnalyzer:
    # rules + heuristic only, so tests do not depend on trained checkpoints
    return StoryAnalyzer(Settings(checkpoints_dir=tmp_path_factory.mktemp("no_ckpt")))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data", checkpoints_dir=tmp_path / "ckpt", enable_evaluator=False, auto_retry=False
    )


@pytest.fixture
def make_client(settings: Settings, analyzer: StoryAnalyzer):
    from app.main import create_app

    clients = []

    def factory(generator=None) -> tuple[TestClient, FakeGenerator]:
        gen = generator or FakeGenerator()
        client = TestClient(create_app(settings, generator=gen, analyzer=analyzer))
        client.__enter__()
        clients.append(client)
        return client, gen

    yield factory
    for c in clients:
        c.__exit__(None, None, None)


def wait_for(client: TestClient, job_id: str, timeout: float = 20.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "partial", "failed", "cancelled"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish: {job['status']}")
