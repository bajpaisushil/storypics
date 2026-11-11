"""SQLite persistence for jobs (stdlib sqlite3; one JSON document per job)."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from app.schemas.jobs import Job, JobSummary


class JobStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, created_at TEXT, updated_at TEXT, "
            "status TEXT, data TEXT NOT NULL)"
        )
        self._lock = threading.Lock()

    def save(self, job: Job) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO jobs (id, created_at, updated_at, status, data) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at, status=excluded.status, "
                "data=excluded.data",
                (job.id, job.created_at.isoformat(), job.updated_at.isoformat(), job.status, job.model_dump_json()),
            )

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            row = self._db.execute("SELECT data FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return Job.model_validate_json(row[0]) if row else None

    def all(self) -> list[Job]:
        with self._lock:
            rows = self._db.execute("SELECT data FROM jobs ORDER BY created_at DESC").fetchall()
        return [Job.model_validate_json(r[0]) for r in rows]

    def summaries(self, limit: int = 50) -> list[JobSummary]:
        out = []
        for job in self.all()[:limit]:
            cover = next((s.selected().thumb_url for s in job.scenes if s.selected()), None)
            out.append(
                JobSummary(
                    id=job.id,
                    title=job.title,
                    status=job.status,
                    style=job.settings.style,
                    scene_count=len(job.scenes),
                    created_at=job.created_at,
                    cover_url=cover,
                )
            )
        return out

    def delete(self, job_id: str) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
