"""Job queue abstractions — memory or Redis."""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone
from typing import Protocol

from anime_episode.domain.enums import JobStatus
from anime_episode.domain.models import JobRecord
from anime_episode.storage.artifact_store import ArtifactStore


class JobQueue(Protocol):
    async def enqueue(self, job: JobRecord) -> JobRecord: ...
    async def dequeue(self) -> JobRecord | None: ...
    async def update(self, job: JobRecord) -> None: ...
    async def get(self, job_id: str) -> JobRecord: ...


class MemoryJobQueue:
    """Process-local FIFO queue backed by ArtifactStore for durability of records."""

    def __init__(self, store: ArtifactStore) -> None:
        self.store = store
        self._q: deque[str] = deque()
        self._cv = asyncio.Condition()

    async def enqueue(self, job: JobRecord) -> JobRecord:
        job.status = JobStatus.QUEUED
        self.store.save_job(job)
        async with self._cv:
            self._q.append(job.id)
            self._cv.notify()
        return job

    async def dequeue(self) -> JobRecord | None:
        async with self._cv:
            while not self._q:
                try:
                    await asyncio.wait_for(self._cv.wait(), timeout=1.0)
                except TimeoutError:
                    return None
            job_id = self._q.popleft()
        job = self.store.load_job(job_id)
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(timezone.utc)
        job.attempts += 1
        self.store.save_job(job)
        return job

    async def update(self, job: JobRecord) -> None:
        self.store.save_job(job)

    async def get(self, job_id: str) -> JobRecord:
        return self.store.load_job(job_id)

    async def depth(self) -> int:
        return len(self._q)


def build_queue(store: ArtifactStore, backend: str = "memory", redis_url: str | None = None):
    backend = (backend or "memory").lower()
    if backend == "memory":
        return MemoryJobQueue(store)
    if backend == "redis":
        from anime_episode.config import get_settings
        from anime_episode.queue.redis_queue import RedisJobQueue

        url = redis_url or get_settings().redis_url
        return RedisJobQueue(store, url)
    raise ValueError(f"unknown queue backend: {backend!r} (use memory|redis)")
