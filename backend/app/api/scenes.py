from fastapi import APIRouter, Depends

from app.api.deps import get_manager
from app.schemas.jobs import Job, RegenerateSceneRequest, SelectVersionRequest, UpdateCharacterRequest
from app.services.job_manager import JobManager

router = APIRouter(prefix="/api/jobs/{job_id}", tags=["scenes"])


@router.post("/scenes/{index}/regenerate", response_model=Job, summary="Regenerate one scene")
async def regenerate_scene(
    job_id: str, index: int, req: RegenerateSceneRequest, manager: JobManager = Depends(get_manager)
) -> Job:
    return await manager.regenerate_scene(job_id, index, req)


@router.post("/scenes/{index}/select", response_model=Job, summary="Choose the version used in the storybook")
async def select_version(
    job_id: str, index: int, req: SelectVersionRequest, manager: JobManager = Depends(get_manager)
) -> Job:
    return manager.select_version(job_id, index, req.version_id)


@router.patch("/characters/{character_id}", response_model=Job, summary="Edit a character's appearance")
async def update_character(
    job_id: str, character_id: str, req: UpdateCharacterRequest, manager: JobManager = Depends(get_manager)
) -> Job:
    return await manager.update_character(job_id, character_id, req)


@router.post("/characters/{character_id}/regenerate", response_model=Job, summary="Redraw a character reference")
async def regenerate_character(job_id: str, character_id: str, manager: JobManager = Depends(get_manager)) -> Job:
    return await manager.regenerate_character(job_id, character_id)


def _attachment(path, filename: str):
    from fastapi.responses import FileResponse

    return FileResponse(path, media_type="image/png", filename=filename)  # sets Content-Disposition: attachment


@router.get("/scenes/{index}/download", summary="Download a scene image (selected version by default)")
async def download_scene(
    job_id: str, index: int, version: str | None = None, manager: JobManager = Depends(get_manager)
):
    from app.services.exporter import _slug
    from app.services.job_manager import InvalidRequest

    job = manager.get(job_id)
    try:
        scene = job.scene(index)
    except KeyError as exc:
        raise InvalidRequest(f"Scene {index} does not exist.") from exc
    chosen = next((v for v in scene.versions if v.id == version), None) if version else scene.selected()
    if chosen is None:
        raise InvalidRequest(f"Scene {index} has no image{' version ' + version if version else ''} yet.")
    suffix = "" if chosen.id == scene.selected_version and not version else f"-{chosen.id}"
    return _attachment(manager.media_dir / chosen.file, f"{index:02d}-{_slug(scene.title)}{suffix}.png")


@router.get("/characters/{character_id}/download", summary="Download a character reference image")
async def download_character(job_id: str, character_id: str, manager: JobManager = Depends(get_manager)):
    from app.services.exporter import _slug
    from app.services.job_manager import InvalidRequest

    job = manager.get(job_id)
    try:
        char = job.character(character_id)
    except KeyError as exc:
        raise InvalidRequest(f"Character {character_id} does not exist.") from exc
    anchor = char.anchor()
    if anchor is None:
        raise InvalidRequest(f"{char.name} has no reference image yet.")
    return _attachment(manager.media_dir / anchor.file, f"character-{_slug(char.name)}.png")
