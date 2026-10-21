"""Shared API dependencies (singletons for scaffolding)."""

from __future__ import annotations

from functools import lru_cache

from anime_episode.config import get_settings
from anime_episode.orchestrator.runner import PipelineRunner
from anime_episode.queue.jobs import JobQueue, build_queue
from anime_episode.storage.artifact_store import ArtifactStore


@lru_cache
def get_store() -> ArtifactStore:
    return ArtifactStore()


@lru_cache
def get_queue() -> JobQueue:
    settings = get_settings()
    return build_queue(
        get_store(),
        backend=settings.queue_backend,
        redis_url=settings.redis_url,
    )


@lru_cache
def get_runner() -> PipelineRunner:
    return PipelineRunner(store=get_store())
