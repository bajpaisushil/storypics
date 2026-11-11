"""FastAPI application factory."""

from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import generation, health, scenes, story
from app.config import Settings, get_settings
from app.services.events import EventBus
from app.services.exporter import NothingToExport
from app.services.image_evaluator import ImageEvaluator
from app.services.image_generator import GeneratorUnavailable, ImageGenerator, create_generator
from app.services.job_manager import InvalidRequest, JobManager, JobNotFound, ServerBusy
from app.services.job_store import JobStore
from app.services.story_parser import StoryAnalyzer
from app.utils.images import InvalidImage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app(
    settings: Settings | None = None,
    generator: ImageGenerator | None = None,
    evaluator: ImageEvaluator | None = None,
    analyzer: StoryAnalyzer | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    settings.media_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manager = JobManager(
            settings,
            JobStore(settings.db_path),
            EventBus(),
            analyzer or StoryAnalyzer(settings),
            generator or create_generator(settings),
            evaluator if evaluator is not None else (ImageEvaluator(settings) if settings.enable_evaluator else None),
        )
        app.state.manager, app.state.settings = manager, settings
        await manager.start()
        yield
        await manager.stop()

    app = FastAPI(
        title="StoryPix API",
        version="1.0.0",
        lifespan=lifespan,
        description="Turn stories into consistently illustrated storybooks. See docs/API.md.",
    )

    # Starlette wraps later-added middleware around earlier ones. Effective order, outermost first:
    # private-network preflight header → CORS → access key → routes (so 401s still carry CORS headers).
    @app.middleware("http")
    async def require_access_key(request: Request, call_next):
        if (
            settings.access_key
            and request.method in ("POST", "PATCH", "PUT", "DELETE")
            and request.url.path.startswith("/api/")
            and request.url.path != "/api/story/analyze"
            and not secrets.compare_digest(request.headers.get("x-storypix-key", ""), settings.access_key)
        ):
            return JSONResponse(status_code=401, content={"detail": "This StoryPix server requires an access key."})
        return await call_next(request)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_origin_regex=settings.cors_origin_regex,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )

    @app.middleware("http")
    async def allow_private_network(request: Request, call_next):
        """Lets a page hosted elsewhere (e.g. the Vercel frontend) reach this backend on localhost.
        Browsers ask first with Access-Control-Request-Private-Network; the user's machine stays unexposed."""
        response = await call_next(request)
        if request.headers.get("access-control-request-private-network") == "true":
            response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response

    for router in (health.router, story.router, generation.router, scenes.router):
        app.include_router(router)
    app.mount("/media", StaticFiles(directory=settings.media_dir), name="media")
    app.mount("/uploads", StaticFiles(directory=settings.uploads_dir), name="uploads")

    def error(status: int, message: str) -> JSONResponse:
        return JSONResponse(status_code=status, content={"detail": message})

    @app.exception_handler(JobNotFound)
    async def _not_found(_: Request, exc: JobNotFound):
        return error(404, f"Job {exc.args[0]} not found.")

    @app.exception_handler(InvalidRequest)
    async def _invalid(_: Request, exc: InvalidRequest):
        return error(400, str(exc))

    @app.exception_handler(InvalidImage)
    async def _bad_image(_: Request, exc: InvalidImage):
        return error(400, str(exc))

    @app.exception_handler(GeneratorUnavailable)
    async def _unavailable(_: Request, exc: GeneratorUnavailable):
        return error(503, str(exc))

    @app.exception_handler(ServerBusy)
    async def _busy(_: Request, exc: ServerBusy):
        return error(429, str(exc))

    @app.exception_handler(NothingToExport)
    async def _nothing(_: Request, exc: NothingToExport):
        return error(409, str(exc))

    @app.exception_handler(ValueError)
    async def _value(_: Request, exc: ValueError):
        return error(400, str(exc))

    return app


app = create_app()
