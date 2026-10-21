"""Redis-backed job queue for multi-worker shot rendering."""

from __future__ import annotations

from datetime import datetime, timezone

from anime_episode.domain.enums import JobStatus
from anime_episode.domain.models import JobRecord
from anime_episode.storage.artifact_store import ArtifactStore

QUEUE_KEY = "anime:jobs"
JOB_KEY = "anime:job:{job_id}"


class RedisJobQueue:
    """FIFO queue using Redis lists; job payloads mirrored to ArtifactStore."""

    def __init__(
        self,
        store: ArtifactStore,
        redis_url: str,
        *,
        queue_key: str = QUEUE_KEY,
        brpop_timeout_s: int = 1,
    ) -> None:
        self.store = store
        self.redis_url = redis_url
        self.queue_key = queue_key
        self.brpop_timeout_s = brpop_timeout_s
        self._redis = None

    async def _client(self):
        if self._redis is None:
            try:
                import redis.asyncio as redis
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "redis package required for ANIME_QUEUE_BACKEND=redis; "
                    "pip install 'redis>=5'"
                ) from exc
            self._redis = redis.from_url(self.redis_url, decode_responses=True)
        return self._redis

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    async def enqueue(self, job: JobRecord) -> JobRecord:
        job.status = JobStatus.QUEUED
        self.store.save_job(job)
        client = await self._client()
        await client.set(JOB_KEY.format(job_id=job.id), job.model_dump_json())
        await client.lpush(self.queue_key, job.id)
        return job

    async def dequeue(self) -> JobRecord | None:
        client = await self._client()
        item = await client.brpop(self.queue_key, timeout=self.brpop_timeout_s)
        if not item:
            return None
        _key, job_id = item
        job = self.store.load_job(job_id)
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(timezone.utc)
        job.attempts += 1
        self.store.save_job(job)
        await client.set(JOB_KEY.format(job_id=job.id), job.model_dump_json())
        return job

    async def update(self, job: JobRecord) -> None:
        self.store.save_job(job)
        client = await self._client()
        await client.set(JOB_KEY.format(job_id=job.id), job.model_dump_json())

    async def get(self, job_id: str) -> JobRecord:
        return self.store.load_job(job_id)

    async def depth(self) -> int:
        client = await self._client()
        return int(await client.llen(self.queue_key))
