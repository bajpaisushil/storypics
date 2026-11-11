"""Asynchronous job orchestration.

* A job = one story: analysis (synchronous, < 1 s) → character anchor tasks → scene tasks, in order.
* One background worker consumes a priority queue and runs one GPU task at a time in a thread, so the API
  stays responsive. Regeneration requests jump ahead of remaining initial work.
* All job mutations happen on the event-loop thread; worker threads only compute and report progress via
  ``loop.call_soon_threadsafe``. Every state change is persisted (SQLite) and published (SSE).
"""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import itertools
import logging
import secrets
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from app.config import Settings
from app.schemas.generation import GenerationInfo, GenerationRequest, ReferenceImage
from app.schemas.jobs import (
    CharacterState,
    CreateJobRequest,
    ImageScores,
    ImageVersion,
    Job,
    JobSettings,
    RegenerateSceneRequest,
    SceneState,
    TaskProgress,
    UpdateCharacterRequest,
    VersionMode,
    utcnow,
)
from app.schemas.story import CharacterProfile, StoryAnalysis
from app.services.character_manager import compose_descriptor
from app.services.events import EventBus
from app.services.image_evaluator import IdentityRef, ImageEvaluator
from app.services.image_generator import GenerationCancelled, GeneratorUnavailable, ImageGenerator
from app.services.job_store import JobStore
from app.services.prompt_builder import BuiltPrompt, PromptBuilder, image_size
from app.services.story_parser import StoryAnalyzer
from app.services.styles import get_style
from app.utils.ids import new_id
from app.utils.images import save_with_thumb

log = logging.getLogger(__name__)

PRIORITY_REGENERATE = 0
PRIORITY_INITIAL = 10
SEED_MAX = 2**32 - 1


class JobNotFound(KeyError):
    pass


class InvalidRequest(ValueError):
    pass


class ServerBusy(RuntimeError):
    pass


@dataclass(order=True)
class Task:
    priority: int
    seq: int
    job_id: str = field(compare=False)
    kind: str = field(compare=False)  # "anchor" | "scene"
    target: str = field(compare=False)  # character id | scene index
    mode: VersionMode = field(compare=False, default="initial")
    instructions: str | None = field(compare=False, default=None)


def match_character(name: str, characters: list[CharacterProfile]) -> CharacterProfile | None:
    key = name.strip().lower()
    for c in characters:
        if key in (c.name.lower(), c.species.lower(), f"the {c.species.lower()}"):
            return c
    choices = {c.name.lower(): c for c in characters} | {c.species.lower(): c for c in characters}
    close = difflib.get_close_matches(key, list(choices), n=1, cutoff=0.75)
    return choices[close[0]] if close else None


class JobManager:
    def __init__(
        self,
        settings: Settings,
        store: JobStore,
        bus: EventBus,
        analyzer: StoryAnalyzer,
        generator: ImageGenerator,
        evaluator: ImageEvaluator | None,
    ):
        self.settings = settings
        self.store = store
        self.bus = bus
        self.analyzer = analyzer
        self.generator = generator
        self.evaluator = evaluator
        self.jobs: dict[str, Job] = {}
        self.queue: asyncio.PriorityQueue[Task] = asyncio.PriorityQueue()
        self.cancel_flags: dict[str, threading.Event] = {}
        self._seq = itertools.count()
        self._worker: asyncio.Task | None = None
        self._preload: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self.media_dir = settings.media_dir
        self.media_dir.mkdir(parents=True, exist_ok=True)
        settings.uploads_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        for job in self.store.all():
            if job.status in ("queued", "generating"):
                self._mark_interrupted(job)
            self.jobs[job.id] = job
        self._worker = asyncio.create_task(self._worker_loop(), name="storypix-worker")
        if self.settings.preload_models:
            self._preload = asyncio.create_task(asyncio.to_thread(self._safe_preload))

    def _safe_preload(self) -> None:
        try:
            self.generator.load()
        except Exception as exc:  # reported through /api/health
            log.warning("preload failed: %s", exc)

    async def stop(self) -> None:
        for flag in self.cancel_flags.values():
            flag.set()
        if self._worker:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker

    def _mark_interrupted(self, job: Job) -> None:
        for c in job.characters:
            if c.anchor_status in ("queued", "generating"):
                c.anchor_status, c.anchor_error = "failed", "Interrupted by a server restart."
        for s in job.scenes:
            if s.status in ("queued", "generating"):
                s.status, s.error = "failed", "Interrupted by a server restart."
        job.status, job.current = "interrupted", None
        self.store.save(job)

    # ------------------------------------------------------------------ helpers
    def get(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            raise JobNotFound(job_id)
        return job

    def _save(self, job: Job, publish: bool = True) -> None:
        job.updated_at = utcnow()
        self.store.save(job)
        if publish:
            self.bus.publish(job.id, {"type": "job", "job": job.model_dump(mode="json")})

    def _enqueue(
        self,
        job: Job,
        kind: str,
        target: str,
        priority: int = PRIORITY_INITIAL,
        mode: VersionMode = "initial",
        instructions: str | None = None,
    ) -> None:
        self.queue.put_nowait(Task(priority, next(self._seq), job.id, kind, target, mode, instructions))

    def _require_generator(self) -> None:
        status = self.generator.status()
        if not status.available:
            raise GeneratorUnavailable(status.reason or "Image generator unavailable")

    # ------------------------------------------------------------------ API operations
    async def analyze(self, story: str, scene_count: int | str, seed_key: str = "") -> StoryAnalysis:
        if len(story) > self.settings.max_story_chars:
            raise InvalidRequest(f"Story is longer than {self.settings.max_story_chars} characters.")
        return await asyncio.to_thread(self.analyzer.analyze, story, scene_count, seed_key)

    async def create_job(self, req: CreateJobRequest) -> Job:
        self._require_generator()
        limit = self.settings.max_active_jobs
        if limit and sum(j.status in ("queued", "generating") for j in self.jobs.values()) >= limit:
            raise ServerBusy(f"The server is busy ({limit} storybook(s) in progress). Please try again shortly.")
        seed = req.seed if req.seed is not None else secrets.randbelow(2**31)
        analysis = await self.analyze(req.story, req.scene_count, seed_key=str(seed))
        profile = self.generator.status().profile
        width, height = image_size(req.aspect_ratio, self.settings.image_resolution, profile)
        warnings = list(analysis.warnings)

        characters = [CharacterState(**c.model_dump()) for c in analysis.characters if c.scene_indices]
        for ref in req.references:
            path = self.settings.uploads_dir / f"{ref.upload_id}.png"
            if not path.exists():
                raise InvalidRequest(f"Unknown upload {ref.upload_id}.")
            target = match_character(ref.character_name, characters)  # type: ignore[arg-type]
            if target is None:
                warnings.append(f"No character called “{ref.character_name}” was found; its reference was not used.")
                continue
            target.reference_upload_id = ref.upload_id
            target.is_recurring = True
            # let the photo decide colours: keep only species + details stated in the story
            target.descriptor = compose_descriptor(
                target.species, {}, [], target.name if target.name.lower() != target.species else None, target.category
            )
        for c in characters:
            c.anchor_status = "queued" if c.is_recurring else "skipped"
        scenes = [SceneState(**s.model_dump(), status="queued") for s in analysis.scenes]

        status = self.generator.status()
        job = Job(
            id=new_id(),
            title=req.title or analysis.title,
            story=req.story,
            settings=JobSettings(
                style=req.style,
                aspect_ratio=req.aspect_ratio,
                scene_count=req.scene_count,
                seed=seed,
                width=width,
                height=height,
                profile=status.profile or "",
            ),
            characters=characters,
            scenes=scenes,
            warnings=warnings,
            analysis=analysis.meta,
            status="queued",
        )
        self.jobs[job.id] = job
        self.cancel_flags[job.id] = threading.Event()
        for c in characters:
            if c.anchor_status == "queued":
                self._enqueue(job, "anchor", c.id)
        for s in scenes:
            self._enqueue(job, "scene", str(s.index))
        self._save(job)
        return job

    async def regenerate_scene(self, job_id: str, index: int, req: RegenerateSceneRequest) -> Job:
        self._require_generator()
        job = self.get(job_id)
        try:
            scene = job.scene(index)
        except KeyError as exc:
            raise InvalidRequest(f"Scene {index} does not exist.") from exc
        if scene.status in ("queued", "generating"):
            raise InvalidRequest(f"Scene {index} is already being generated.")
        mode: VersionMode = req.mode if scene.selected() else "new_variation"
        scene.status, scene.error = "queued", None
        self._reopen(job)
        self._enqueue(job, "scene", str(index), PRIORITY_REGENERATE, mode, req.instructions)
        self._save(job)
        return job

    async def regenerate_character(self, job_id: str, character_id: str, cascade: bool = False) -> Job:
        self._require_generator()
        job = self.get(job_id)
        try:
            char = job.character(character_id)
        except KeyError as exc:
            raise InvalidRequest(f"Character {character_id} does not exist.") from exc
        if char.anchor_status in ("queued", "generating"):
            raise InvalidRequest(f"{char.name} is already being generated.")
        char.anchor_status, char.anchor_error, char.is_recurring = "queued", None, True
        self._reopen(job)
        self._enqueue(job, "anchor", char.id, PRIORITY_REGENERATE)
        if cascade:
            for s in job.scenes:
                if char.id in s.character_ids and s.status not in ("queued", "generating"):
                    s.status, s.error = "queued", None
                    self._enqueue(job, "scene", str(s.index), PRIORITY_REGENERATE + 1, "keep_composition")
        self._save(job)
        return job

    async def update_character(self, job_id: str, character_id: str, req: UpdateCharacterRequest) -> Job:
        job = self.get(job_id)
        try:
            char = job.character(character_id)
        except KeyError as exc:
            raise InvalidRequest(f"Character {character_id} does not exist.") from exc
        if req.name:
            char.name = req.name.strip()
        if req.attributes is not None:
            char.attributes = {k.strip(): v.strip() for k, v in req.attributes.items() if k.strip() and v.strip()}
        if req.descriptor:
            char.descriptor = req.descriptor.strip()
        elif req.attributes is not None:
            name = char.name if char.name.lower() != char.species.lower() else None
            char.descriptor = compose_descriptor(char.species, char.attributes, [], name, char.category)
        self._save(job)
        if req.regenerate:
            return await self.regenerate_character(job_id, character_id, cascade=True)
        return job

    def select_version(self, job_id: str, index: int, version_id: str) -> Job:
        job = self.get(job_id)
        scene = job.scene(index)
        if not any(v.id == version_id for v in scene.versions):
            raise InvalidRequest(f"Version {version_id} not found for scene {index}.")
        scene.selected_version = version_id
        self._save(job)
        return job

    async def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        self.cancel_flags.setdefault(job_id, threading.Event()).set()
        for c in job.characters:
            if c.anchor_status in ("queued", "generating"):
                c.anchor_status = "cancelled"
        for s in job.scenes:
            if s.status in ("queued", "generating"):
                s.status = "cancelled"
        job.status, job.current = "cancelled", None
        self._save(job)
        return job

    async def resume(self, job_id: str) -> Job:
        """Re-queue everything that did not complete (after a cancel, failure or restart)."""
        self._require_generator()
        job = self.get(job_id)
        self.cancel_flags[job_id] = threading.Event()
        for c in job.characters:
            if c.is_recurring and c.anchor_status in ("failed", "cancelled", "pending"):
                c.anchor_status, c.anchor_error = "queued", None
                self._enqueue(job, "anchor", c.id)
        for s in job.scenes:
            if s.status in ("failed", "cancelled", "pending"):
                s.status, s.error = "queued", None
                self._enqueue(job, "scene", str(s.index))
        self._reopen(job)
        self._save(job)
        return job

    def delete(self, job_id: str) -> None:
        job = self.get(job_id)
        self.cancel_flags.setdefault(job_id, threading.Event()).set()
        self.jobs.pop(job.id, None)
        self.store.delete(job.id)
        shutil.rmtree(self.media_dir / job.id, ignore_errors=True)

    def _reopen(self, job: Job) -> None:
        if job.status in ("completed", "partial", "failed", "cancelled", "interrupted"):
            job.status = "queued"
        flag = self.cancel_flags.get(job.id)
        if flag is None or flag.is_set():
            self.cancel_flags[job.id] = threading.Event()

    # ------------------------------------------------------------------ worker
    async def _worker_loop(self) -> None:
        while True:
            task = await self.queue.get()
            job = self.jobs.get(task.job_id)
            if job is None or job.status == "cancelled":
                continue
            try:
                if task.kind == "anchor":
                    await self._run_anchor(job, task)
                else:
                    await self._run_scene(job, task)
            except Exception:  # never let one task kill the worker
                log.exception("task %s failed unexpectedly", task)
            finally:
                self._finish_if_done(job)

    def _progress_callback(self, job: Job) -> callable:
        loop = self._loop
        assert loop is not None

        def on_step(step: int, total: int) -> None:
            loop.call_soon_threadsafe(self._on_progress, job.id, step, total)

        return on_step

    def _on_progress(self, job_id: str, step: int, total: int) -> None:
        job = self.jobs.get(job_id)
        if job is None or job.current is None:
            return
        job.current.step, job.current.total_steps = step, total
        self.bus.publish(job_id, {"type": "progress", "current": job.current.model_dump(mode="json")})

    async def _ensure_loaded(self, job: Job, progress: TaskProgress) -> None:
        if not self.generator.status().loaded:
            job.current = progress.model_copy(update={"label": "Loading the image model (first run only)…"})
            self._save(job)
            await asyncio.to_thread(self.generator.load)

    def _seed(self, job: Job, offset: int, attempt: int) -> int:
        return (job.settings.seed + offset + attempt * 104_729) % SEED_MAX

    async def _generate(self, job: Job, request: GenerationRequest) -> tuple[Image.Image, GenerationInfo]:
        flag = self.cancel_flags.setdefault(job.id, threading.Event())
        return await asyncio.to_thread(self.generator.generate, request, self._progress_callback(job), flag)

    def _store_version(
        self,
        job: Job,
        stem: str,
        image: Image.Image,
        info: GenerationInfo,
        built: BuiltPrompt,
        mode: VersionMode,
        instructions: str | None,
        scores: ImageScores | None,
        attempts: int,
        existing: list[ImageVersion],
    ) -> ImageVersion:
        version_id = f"v{len(existing) + 1}"
        rel = Path(job.id) / f"{stem}_{version_id}.png"
        _original, thumb = save_with_thumb(image, self.media_dir / rel)
        return ImageVersion(
            id=version_id,
            url=f"/media/{rel.as_posix()}",
            thumb_url=f"/media/{(rel.parent / thumb.name).as_posix()}",
            file=rel.as_posix(),
            mode=mode,
            seed=info.seed,
            prompt=built.prompt,
            negative_prompt=built.negative_prompt,
            instructions=instructions,
            scores=scores,
            duration_s=info.duration_s,
            attempts=attempts,
            generation=info.model_dump(),
        )

    async def _run_anchor(self, job: Job, task: Task) -> None:
        char = job.character(task.target)
        if char.anchor_status != "queued":
            return
        style = get_style(job.settings.style)
        upload = self.settings.uploads_dir / f"{char.reference_upload_id}.png" if char.reference_upload_id else None
        built = PromptBuilder(style).anchor_prompt(char, from_upload=upload is not None)
        progress = TaskProgress(kind="anchor", target=char.id, label=f"Designing {char.name}")
        char.anchor_status, job.status, job.current = "generating", "generating", progress
        self._save(job)
        try:
            await self._ensure_loaded(job, progress)
            job.current = progress
            refs = [ReferenceImage(character_id=char.id, path=str(upload))] if upload else []
            side = image_size("1:1", self.settings.image_resolution, job.settings.profile)[0]
            attempt = len(char.anchor_versions)
            request = GenerationRequest(
                kind="anchor",
                prompt=built.prompt,
                negative_prompt=built.negative_prompt,
                width=side,
                height=side,
                seed=self._seed(job, 1000 * (job.characters.index(char) + 1), attempt),
                references=refs,
                ip_adapter_scale=built.ip_adapter_scale,
                guidance_scale=style.guidance_scale,
            )
            image, info = await self._generate(job, request)
            version = self._store_version(
                job,
                f"char_{char.id}",
                image,
                info,
                built,
                "upload" if upload else "initial",
                None,
                None,
                1,
                char.anchor_versions,
            )
            char.anchor_versions.append(version)
            char.selected_anchor = version.id
            char.anchor_status = "completed"
        except GenerationCancelled:
            char.anchor_status = "cancelled"
        except Exception as exc:
            log.exception("anchor generation failed")
            char.anchor_status, char.anchor_error = "failed", _friendly(exc)
        finally:
            job.current = None
            self._save(job)

    async def _run_scene(self, job: Job, task: Task) -> None:
        scene = job.scene(int(task.target))
        if scene.status != "queued":
            return
        style = get_style(job.settings.style)
        anchors = {c.id: c.anchor() for c in job.characters if c.anchor() is not None}
        built = PromptBuilder(style).scene_prompt(scene, job.characters, set(anchors), task.instructions)
        refs = [
            ReferenceImage(
                character_id=slot.character_id,
                path=str(self.media_dir / anchors[slot.character_id].file),
                region=slot.region,
            )
            for slot in built.slots
        ]
        progress = TaskProgress(kind="scene", target=str(scene.index), label=f"Illustrating scene {scene.index}")
        scene.status, job.status, job.current = "generating", "generating", progress
        self._save(job)
        previous = scene.selected()
        try:
            await self._ensure_loaded(job, progress)
            identities = [IdentityRef(r.character_id, Path(r.path or ""), r.region) for r in refs]
            best: tuple[Image.Image, GenerationInfo, ImageScores | None] | None = None
            max_attempts = 1 + (self.settings.max_retries if self.settings.auto_retry else 0)
            attempts = 0
            candidates = max(1, self.settings.candidates_per_scene)
            for attempt in range(max_attempts * candidates):
                attempts += 1
                job.current = progress.model_copy(update={"attempt": attempts})
                self._save(job)
                if task.mode == "keep_composition" and previous is not None and attempt == 0:
                    seed = previous.seed
                elif task.mode in ("new_variation", "keep_composition"):
                    seed = secrets.randbelow(SEED_MAX)
                else:
                    seed = self._seed(job, 7919 * scene.index, attempt)
                request = GenerationRequest(
                    kind="scene",
                    prompt=built.prompt,
                    negative_prompt=built.negative_prompt,
                    width=job.settings.width,
                    height=job.settings.height,
                    seed=seed,
                    references=refs,
                    ip_adapter_scale=built.ip_adapter_scale,
                    guidance_scale=style.guidance_scale,
                )
                image, info = await self._generate(job, request)
                scores = None
                if self.evaluator is not None:
                    text = f"{scene.visual_description}. {scene.location.description if scene.location else ''}"
                    scores = await asyncio.to_thread(self.evaluator.evaluate, image, text, identities)
                if best is None or _better(scores, best[2]):
                    best = (image, info, scores)
                enough = scores is None or scores.overall is None or scores.overall >= self.settings.retry_below_score
                if attempts >= candidates and enough:
                    break
            assert best is not None
            image, info, scores = best
            mode: VersionMode = task.mode if task.mode != "initial" else ("retry" if attempts > 1 else "initial")
            version = self._store_version(
                job,
                f"scene_{scene.index:02d}",
                image,
                info,
                built,
                mode,
                task.instructions,
                scores,
                attempts,
                scene.versions,
            )
            scene.versions.append(version)
            scene.selected_version = version.id
            scene.status, scene.error = "completed", None
        except GenerationCancelled:
            scene.status = "cancelled"
        except Exception as exc:
            log.exception("scene generation failed")
            scene.status, scene.error = "failed", _friendly(exc)
        finally:
            job.current = None
            self._save(job)

    def _finish_if_done(self, job: Job) -> None:
        tasks = [c.anchor_status for c in job.characters if c.anchor_status != "skipped"] + [
            s.status for s in job.scenes
        ]
        if any(t in ("queued", "generating") for t in tasks) or job.status == "cancelled":
            return
        scene_states = [s.status for s in job.scenes]
        if all(t == "completed" for t in scene_states):
            job.status = "completed"
        elif any(t == "completed" for t in scene_states):
            job.status = "partial"
        else:
            job.status = "failed"
            job.error = next((s.error for s in job.scenes if s.error), "Generation failed.")
        job.current = None
        self._save(job)


def _better(new: ImageScores | None, old: ImageScores | None) -> bool:
    if old is None or old.overall is None:
        return not (new is None or new.overall is None)
    return new is not None and new.overall is not None and new.overall > old.overall


def _friendly(exc: Exception) -> str:
    text = str(exc) or exc.__class__.__name__
    if "out of memory" in text.lower() or "MPS backend out of memory" in text:
        return (
            "The device ran out of memory. Close other apps, use a smaller aspect ratio, or set "
            "STORYPIX_IMAGE_RESOLUTION=768."
        )
    return text[:400]
