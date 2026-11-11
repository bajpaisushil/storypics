"""In-process pub/sub for job progress (feeds the SSE endpoint)."""

from __future__ import annotations

import asyncio
import contextlib
from collections import defaultdict


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, set[asyncio.Queue]] = defaultdict(set)

    def subscribe(self, job_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._subs[job_id].add(q)
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue) -> None:
        self._subs[job_id].discard(q)
        if not self._subs[job_id]:
            self._subs.pop(job_id, None)

    def publish(self, job_id: str, event: dict) -> None:
        """Must be called from the event-loop thread."""
        for q in list(self._subs.get(job_id, ())):
            if q.full():  # slow consumer: drop the oldest event rather than block generation
                with contextlib.suppress(asyncio.QueueEmpty):
                    q.get_nowait()
            q.put_nowait(event)
