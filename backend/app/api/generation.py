import asyncio
import json

from fastapi import APIRouter, Depends, Request, Response, status
from fastapi.responses import StreamingResponse

from app.api.deps import get_manager
from app.schemas.jobs import CreateJobRequest, Job, JobSummary
from app.services.exporter import _slug, build_pdf, build_zip
from app.services.job_manager import JobManager

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.post("", response_model=Job, status_code=status.HTTP_202_ACCEPTED, summary="Start generating a storybook")
async def create_job(req: CreateJobRequest, manager: JobManager = Depends(get_manager)) -> Job:
    return await manager.create_job(req)


@router.get("", response_model=list[JobSummary], summary="Recent storybooks")
async def list_jobs(manager: JobManager = Depends(get_manager)) -> list[JobSummary]:
    return manager.store.summaries()


@router.get("/{job_id}", response_model=Job)
async def get_job(job_id: str, manager: JobManager = Depends(get_manager)) -> Job:
    return manager.get(job_id)


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_job(job_id: str, manager: JobManager = Depends(get_manager)) -> Response:
    manager.delete(job_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{job_id}/cancel", response_model=Job)
async def cancel_job(job_id: str, manager: JobManager = Depends(get_manager)) -> Job:
    return await manager.cancel(job_id)


@router.post("/{job_id}/resume", response_model=Job, summary="Retry everything that did not complete")
async def resume_job(job_id: str, manager: JobManager = Depends(get_manager)) -> Job:
    return await manager.resume(job_id)


@router.get("/{job_id}/events", summary="Server-Sent Events: 'job' snapshots and 'progress' steps")
async def job_events(job_id: str, request: Request, manager: JobManager = Depends(get_manager)) -> StreamingResponse:
    job = manager.get(job_id)

    async def stream():
        queue = manager.bus.subscribe(job_id)
        try:
            yield f"event: job\ndata: {json.dumps({'type': 'job', 'job': job.model_dump(mode='json')})}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                except TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
        finally:
            manager.bus.unsubscribe(job_id, queue)

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@router.get("/{job_id}/export.pdf", summary="Download the storybook as PDF")
async def export_pdf(job_id: str, manager: JobManager = Depends(get_manager)) -> Response:
    job = manager.get(job_id)
    data = await asyncio.to_thread(build_pdf, job, manager.media_dir)
    return Response(
        data,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{_slug(job.title)}.pdf"'},
    )


@router.get("/{job_id}/export.zip", summary="Download all images + manifest as ZIP")
async def export_zip(job_id: str, manager: JobManager = Depends(get_manager)) -> Response:
    job = manager.get(job_id)
    data = await asyncio.to_thread(build_zip, job, manager.media_dir)
    return Response(
        data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{_slug(job.title)}.zip"'},
    )
