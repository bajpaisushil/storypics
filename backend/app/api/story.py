from fastapi import APIRouter, Depends, File, UploadFile

from app.api.deps import get_app_settings, get_manager
from app.config import Settings
from app.schemas.jobs import UploadResponse
from app.schemas.story import AnalyzeRequest, StoryAnalysis
from app.services.job_manager import JobManager
from app.utils.ids import new_id
from app.utils.images import load_upload

router = APIRouter(prefix="/api", tags=["story"])


@router.post("/story/analyze", response_model=StoryAnalysis, summary="Analyse a story (no images)")
async def analyze(req: AnalyzeRequest, manager: JobManager = Depends(get_manager)) -> StoryAnalysis:
    return await manager.analyze(req.story, req.scene_count)


@router.post("/uploads", response_model=UploadResponse, summary="Upload a character reference image")
async def upload(file: UploadFile = File(...), settings: Settings = Depends(get_app_settings)) -> UploadResponse:
    data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    image = load_upload(data, settings.max_upload_mb * 1024 * 1024)
    upload_id = new_id()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    image.save(settings.uploads_dir / f"{upload_id}.png", format="PNG")
    return UploadResponse(upload_id=upload_id, url=f"/uploads/{upload_id}.png", width=image.width, height=image.height)
