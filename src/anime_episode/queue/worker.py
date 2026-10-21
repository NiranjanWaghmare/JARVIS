"""Background worker that drains the job queue into the orchestrator."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from anime_episode.domain.enums import JobStatus, JobType
from anime_episode.orchestrator.runner import PipelineRunner
from anime_episode.queue.jobs import JobQueue
from anime_episode.storage.artifact_store import ArtifactStore

logger = logging.getLogger(__name__)


class Worker:
    def __init__(
        self,
        queue: JobQueue,
        store: ArtifactStore,
        runner: PipelineRunner | None = None,
    ) -> None:
        self.queue = queue
        self.store = store
        self.runner = runner or PipelineRunner(store=store)
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run_forever(self) -> None:
        logger.info("anime episode worker started")
        while not self._stop.is_set():
            job = await self.queue.dequeue()
            if job is None:
                continue
            try:
                result = await self._dispatch(job.type, job.episode_id, job.shot_id, job.payload)
                job.status = JobStatus.SUCCEEDED
                job.result = result
                job.finished_at = datetime.now(timezone.utc)
            except Exception as exc:  # noqa: BLE001 — worker boundary
                logger.exception("job %s failed", job.id)
                job.status = JobStatus.FAILED
                job.error = str(exc)
                job.finished_at = datetime.now(timezone.utc)
            await self.queue.update(job)

    async def _dispatch(
        self,
        job_type: JobType,
        episode_id: str,
        shot_id: str | None,
        payload: dict,
    ) -> dict:
        if job_type == JobType.PLAN_EPISODE:
            episode = await self.runner.plan(episode_id)
            return {"shot_count": len(episode.shots), "scene_count": len(episode.scenes)}
        if job_type == JobType.BUILD_CHARACTERS:
            episode = await self.runner.build_characters(episode_id)
            return {"character_count": len(episode.characters)}
        if job_type == JobType.RENDER_SHOT:
            if not shot_id:
                raise ValueError("RENDER_SHOT requires shot_id")
            shot = await self.runner.render_shot(episode_id, shot_id)
            return {"shot_id": shot.id, "stage": shot.stage, "artifacts": shot.artifacts}
        if job_type == JobType.ASSEMBLE_EPISODE:
            path = await self.runner.assemble(episode_id)
            return {"final_video_path": path}
        raise ValueError(f"unknown job type: {job_type}")
