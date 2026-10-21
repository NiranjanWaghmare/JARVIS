"""Background worker that drains the job queue into the orchestrator."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from anime_episode.domain.enums import JobStatus, JobType, ShotStage
from anime_episode.domain.models import JobRecord
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
        *,
        chain: bool = True,
        concurrency: int = 1,
    ) -> None:
        self.queue = queue
        self.store = store
        self.runner = runner or PipelineRunner(store=store)
        self.chain = chain
        self.concurrency = max(1, concurrency)
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run_forever(self) -> None:
        logger.info(
            "anime episode worker started (concurrency=%s, chain=%s)",
            self.concurrency,
            self.chain,
        )
        if self.concurrency == 1:
            await self._loop()
            return
        tasks = [asyncio.create_task(self._loop(), name=f"anime-worker-{i}") for i in range(self.concurrency)]
        await asyncio.gather(*tasks)

    async def _loop(self) -> None:
        while not self._stop.is_set():
            job = await self.queue.dequeue()
            if job is None:
                continue
            await self.process_job(job)

    async def process_job(self, job: JobRecord) -> JobRecord:
        try:
            result = await self._dispatch(job.type, job.episode_id, job.shot_id, job.payload)
            job.status = JobStatus.SUCCEEDED
            job.result = result
            job.finished_at = datetime.now(timezone.utc)
            await self.queue.update(job)
            if self.chain and job.payload.get("chain", True):
                await self._enqueue_followups(job)
        except Exception as exc:  # noqa: BLE001 — worker boundary
            logger.exception("job %s failed", job.id)
            job.status = JobStatus.FAILED
            job.error = str(exc)
            job.finished_at = datetime.now(timezone.utc)
            await self.queue.update(job)
        return job

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

    async def _enqueue_followups(self, job: JobRecord) -> None:
        """DAG: plan → characters → render_shot×N → assemble (when all shots done)."""
        if job.type == JobType.PLAN_EPISODE:
            nxt = JobRecord(
                type=JobType.BUILD_CHARACTERS,
                episode_id=job.episode_id,
                payload={"chain": True},
            )
            await self.queue.enqueue(nxt)
            logger.info("chained BUILD_CHARACTERS for %s", job.episode_id)
            return

        if job.type == JobType.BUILD_CHARACTERS:
            episode = self.store.load_episode(job.episode_id)
            for shot in episode.shots:
                nxt = JobRecord(
                    type=JobType.RENDER_SHOT,
                    episode_id=job.episode_id,
                    shot_id=shot.id,
                    payload={"chain": True},
                )
                await self.queue.enqueue(nxt)
            logger.info(
                "chained %s RENDER_SHOT jobs for %s",
                len(episode.shots),
                job.episode_id,
            )
            return

        if job.type == JobType.RENDER_SHOT:
            episode = self.store.load_episode(job.episode_id)
            if any(s.stage == ShotStage.FAILED for s in episode.shots):
                logger.warning("episode %s has failed shots; skipping assemble", job.episode_id)
                return
            if all(s.stage == ShotStage.DONE for s in episode.shots) and episode.shots:
                if not self.store.try_claim_assemble(job.episode_id):
                    return
                nxt = JobRecord(
                    type=JobType.ASSEMBLE_EPISODE,
                    episode_id=job.episode_id,
                    payload={"chain": True},
                )
                await self.queue.enqueue(nxt)
                logger.info("chained ASSEMBLE_EPISODE for %s", job.episode_id)
