"""FastAPI routes for episode creation and pipeline control."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from anime_episode.api.deps import get_queue, get_runner, get_store
from anime_episode.api.schemas import (
    CreateEpisodeRequest,
    CreateEpisodeResponse,
    EpisodeStatusResponse,
    JobStatusResponse,
    KickoffRequest,
    KickoffResponse,
    shot_status,
)
from anime_episode.domain.enums import JobType, ShotStage
from anime_episode.domain.models import Episode, JobRecord
from anime_episode.queue.worker import Worker

router = APIRouter(prefix="/v1")


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "anime_episode"}


@router.get("/health/comfyui")
async def comfyui_health() -> dict:
    """Probe configured ComfyUI and list workflow template status."""
    from anime_episode.config import get_settings
    from anime_episode.tools.comfyui_client import ComfyUIClient
    from anime_episode.tools.workflows import WORKFLOWS_DIR, load_workflow

    settings = get_settings()
    client = ComfyUIClient(settings.comfyui_base_url, settings.comfyui_timeout_s)
    reachable = await client.health()
    templates = {
        settings.keyframe_workflow: (WORKFLOWS_DIR / settings.keyframe_workflow).exists(),
        settings.motion_workflow: (WORKFLOWS_DIR / settings.motion_workflow).exists(),
    }
    load_errors: dict[str, str] = {}
    for name, exists in templates.items():
        if not exists:
            continue
        try:
            load_workflow(name)
        except Exception as exc:  # noqa: BLE001
            load_errors[name] = str(exc)
    return {
        "comfyui_base_url": settings.comfyui_base_url,
        "reachable": reachable,
        "dry_run": settings.dry_run,
        "workflows": templates,
        "workflow_load_errors": load_errors,
        "keyframe_ckpt": settings.keyframe_ckpt,
        "motion_unet_high": settings.motion_unet_high,
        "motion_unet_low": settings.motion_unet_low,
    }


@router.post("/episodes", response_model=CreateEpisodeResponse)
async def create_episode(body: CreateEpisodeRequest) -> CreateEpisodeResponse:
    store = get_store()
    episode = Episode(
        title=body.title,
        premise=body.premise,
        style=body.style,
        target_duration_s=body.target_duration_s,
    )
    store.save_episode(episode)
    return CreateEpisodeResponse(id=episode.id, status=episode.status, title=episode.title)


@router.get("/episodes/{episode_id}", response_model=EpisodeStatusResponse)
async def get_episode(episode_id: str) -> EpisodeStatusResponse:
    store = get_store()
    try:
        episode = store.load_episode(episode_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="episode not found") from exc
    return EpisodeStatusResponse(
        id=episode.id,
        title=episode.title,
        status=episode.status,
        target_duration_s=episode.target_duration_s,
        character_count=len(episode.characters),
        scene_count=len(episode.scenes),
        shot_count=len(episode.shots),
        shots_done=sum(1 for s in episode.shots if s.stage == ShotStage.DONE),
        final_video_path=episode.final_video_path,
        error=episode.error,
        characters=episode.characters,
        scenes=episode.scenes,
        shots=[shot_status(s) for s in episode.shots],
    )


@router.post("/episodes/{episode_id}/kickoff", response_model=KickoffResponse)
async def kickoff_episode(episode_id: str, body: KickoffRequest) -> KickoffResponse:
    store = get_store()
    queue = get_queue()
    runner = get_runner()
    try:
        episode = store.load_episode(episode_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="episode not found") from exc

    if body.sync:
        episode = await runner.kickoff_local(episode_id)
        return KickoffResponse(
            episode_id=episode.id,
            mode="sync",
            job_ids=[],
            status=episode.status,
        )

    job_ids: list[str] = []
    plan_job = JobRecord(type=JobType.PLAN_EPISODE, episode_id=episode_id)
    await queue.enqueue(plan_job)
    job_ids.append(plan_job.id)

    # Character + shot + assemble jobs are chained by a coordinator worker in
    # production; for the architecture scaffold we enqueue a bootstrap plan job
    # and expose /kickoff/continue for the remaining DAG after planning.
    return KickoffResponse(
        episode_id=episode_id,
        mode="async",
        job_ids=job_ids,
        status=episode.status,
    )


@router.post("/episodes/{episode_id}/continue", response_model=KickoffResponse)
async def continue_after_plan(episode_id: str) -> KickoffResponse:
    """Enqueue character build, all shot renders, then assemble (post-plan)."""
    store = get_store()
    queue = get_queue()
    try:
        episode = store.load_episode(episode_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="episode not found") from exc
    if not episode.shots:
        raise HTTPException(status_code=400, detail="episode has no shots; run plan first")

    job_ids: list[str] = []
    char_job = JobRecord(type=JobType.BUILD_CHARACTERS, episode_id=episode_id)
    await queue.enqueue(char_job)
    job_ids.append(char_job.id)

    for shot in episode.shots:
        job = JobRecord(type=JobType.RENDER_SHOT, episode_id=episode_id, shot_id=shot.id)
        await queue.enqueue(job)
        job_ids.append(job.id)

    assemble = JobRecord(type=JobType.ASSEMBLE_EPISODE, episode_id=episode_id)
    await queue.enqueue(assemble)
    job_ids.append(assemble.id)

    return KickoffResponse(
        episode_id=episode_id,
        mode="async",
        job_ids=job_ids,
        status=episode.status,
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str) -> JobStatusResponse:
    store = get_store()
    try:
        job = store.load_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc
    return JobStatusResponse(
        id=job.id,
        type=job.type,
        episode_id=job.episode_id,
        shot_id=job.shot_id,
        status=job.status,
        error=job.error,
        result=job.result,
    )


@router.post("/worker/tick")
async def worker_tick(max_jobs: int = 1) -> dict:
    """Process up to N queued jobs in-process (dev helper; use Worker.run_forever in prod)."""
    worker = Worker(queue=get_queue(), store=get_store(), runner=get_runner())
    processed = 0
    for _ in range(max(1, max_jobs)):
        job = await get_queue().dequeue()
        if job is None:
            break
        # Re-enqueue semantics: dequeue already marked running; dispatch via worker internals
        from datetime import datetime, timezone

        from anime_episode.domain.enums import JobStatus

        try:
            result = await worker._dispatch(job.type, job.episode_id, job.shot_id, job.payload)
            job.status = JobStatus.SUCCEEDED
            job.result = result
            job.finished_at = datetime.now(timezone.utc)
        except Exception as exc:  # noqa: BLE001
            job.status = JobStatus.FAILED
            job.error = str(exc)
            job.finished_at = datetime.now(timezone.utc)
        await get_queue().update(job)
        processed += 1
    return {"processed": processed}
