from fastapi import APIRouter, Depends

from app.api.deps import get_manager
from app.services.job_manager import JobManager
from app.services.styles import STYLES
from app.services.usage import UsageReport, UsageTracker

_usage: UsageTracker | None = None

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health", summary="Service, model and device status")
async def health(manager: JobManager = Depends(get_manager)) -> dict:
    return {
        "status": "ok",
        "generator": manager.generator.status().model_dump(),
        "analyzer": {
            "segmenter": manager.analyzer.scorer.name,
            "tagger": getattr(manager.analyzer.detector, "name", "rules"),
        },
        "evaluator": manager.evaluator.describe() if manager.evaluator else {"enabled": False},
        "queue_size": manager.queue.qsize(),
        "requires_key": bool(manager.settings.access_key),
        "max_active_jobs": manager.settings.max_active_jobs,
    }


@router.get("/styles", summary="Illustration styles")
async def styles() -> list[dict]:
    return [s.public() for s in STYLES.values()]


@router.get("/usage", response_model=UsageReport, summary="Hosted compute used this month (Modal)")
async def usage(manager: JobManager = Depends(get_manager)) -> UsageReport:
    global _usage
    if _usage is None:
        _usage = UsageTracker(manager.settings.monthly_credit_usd)
    import asyncio

    return await asyncio.to_thread(_usage.report)
