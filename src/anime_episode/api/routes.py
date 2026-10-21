"""FastAPI routes for episode creation and pipeline control."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from anime_episode.api.deps import get_queue, get_runner, get_store
from anime_episode.api.schemas import (
    CharacterVoiceStatusResponse,
    CreateEpisodeRequest,
    CreateEpisodeResponse,
    EpisodeStatusResponse,
    JobStatusResponse,
    KickoffRequest,
    KickoffResponse,
    UpdateCharacterRequest,
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
        settings.animate_workflow: (WORKFLOWS_DIR / settings.animate_workflow).exists(),
        settings.dialogue_workflow: (WORKFLOWS_DIR / settings.dialogue_workflow).exists(),
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
        "animate_unet": settings.animate_unet,
        "dialogue_unet": settings.dialogue_unet,
    }


@router.get("/health/tts")
async def tts_health() -> dict:
    """Probe GPT-SoVITS api_v2 endpoint."""
    from anime_episode.config import get_settings
    from anime_episode.tools.tts_client import GPTSoVITSClient

    settings = get_settings()
    client = GPTSoVITSClient(settings.tts_base_url, settings.tts_timeout_s)
    reachable = await client.health()
    return {
        "tts_base_url": settings.tts_base_url,
        "reachable": reachable,
        "dry_run": settings.dry_run,
        "model": settings.tts_model,
        "default_ref_audio_set": bool(settings.tts_default_ref_audio),
        "text_lang": settings.tts_text_lang,
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


@router.get(
    "/episodes/{episode_id}/characters/voice-status",
    response_model=CharacterVoiceStatusResponse,
)
async def character_voice_status(episode_id: str) -> CharacterVoiceStatusResponse:
    from anime_episode.tools.character_assets import voice_readiness

    store = get_store()
    try:
        episode = store.load_episode(episode_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="episode not found") from exc
    status = voice_readiness(episode)
    return CharacterVoiceStatusResponse(episode_id=episode_id, characters=episode.characters, **status)


@router.patch("/episodes/{episode_id}/characters/{character_id}")
async def update_character(
    episode_id: str,
    character_id: str,
    body: UpdateCharacterRequest,
) -> dict:
    from anime_episode.tools.character_assets import update_character_in_episode

    store = get_store()
    try:
        episode = store.load_episode(episode_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="episode not found") from exc
    if not any(c.id == character_id for c in episode.characters):
        raise HTTPException(status_code=404, detail="character not found")

    fields = body.model_dump(exclude_unset=True)

    def _mutate(ep) -> None:
        update_character_in_episode(ep, character_id, **fields)

    episode = store.mutate_episode(episode_id, _mutate)
    char = next(c for c in episode.characters if c.id == character_id)
    return {"episode_id": episode_id, "character": char.model_dump()}


@router.post("/episodes/{episode_id}/characters/{character_id}/voice")
async def upload_character_voice(
    episode_id: str,
    character_id: str,
    prompt_text: str = "",
    lang: str = "en",
    file_path: str | None = None,
) -> dict:
    """Attach a voice reference WAV already on disk (absolute path on the server)."""
    from pathlib import Path

    from anime_episode.tools.character_assets import install_voice_ref

    if not file_path:
        raise HTTPException(
            status_code=400,
            detail="file_path is required (absolute path to a WAV/MP3 on the server)",
        )
    if not Path(file_path).exists():
        raise HTTPException(status_code=400, detail=f"voice file not found: {file_path}")

    store = get_store()
    try:
        episode = store.load_episode(episode_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="episode not found") from exc
    if not any(c.id == character_id for c in episode.characters):
        raise HTTPException(status_code=404, detail="character not found")

    artifact_dir = str(store.episode_dir(episode_id))

    def _mutate(ep) -> None:
        char = next(c for c in ep.characters if c.id == character_id)
        install_voice_ref(
            artifact_dir=artifact_dir,
            character=char,
            source_path=file_path,
            prompt_text=prompt_text or None,
            lang=lang,
        )

    episode = store.mutate_episode(episode_id, _mutate)
    char = next(c for c in episode.characters if c.id == character_id)
    return {"episode_id": episode_id, "character": char.model_dump()}


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

    # Enqueue plan only; Worker chain auto-enqueues characters → parallel shots → assemble.
    plan_job = JobRecord(
        type=JobType.PLAN_EPISODE,
        episode_id=episode_id,
        payload={"chain": True},
    )
    await queue.enqueue(plan_job)
    return KickoffResponse(
        episode_id=episode_id,
        mode="async",
        job_ids=[plan_job.id],
        status=episode.status,
    )


@router.post("/episodes/{episode_id}/continue", response_model=KickoffResponse)
async def continue_after_plan(episode_id: str) -> KickoffResponse:
    """Manual DAG step: enqueue characters + shots (assemble chains when shots finish)."""
    store = get_store()
    queue = get_queue()
    try:
        episode = store.load_episode(episode_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="episode not found") from exc
    if not episode.shots:
        raise HTTPException(status_code=400, detail="episode has no shots; run plan first")

    job_ids: list[str] = []
    char_job = JobRecord(
        type=JobType.BUILD_CHARACTERS,
        episode_id=episode_id,
        payload={"chain": True},
    )
    await queue.enqueue(char_job)
    job_ids.append(char_job.id)
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
    from anime_episode.config import get_settings

    settings = get_settings()
    worker = Worker(
        queue=get_queue(),
        store=get_store(),
        runner=get_runner(),
        chain=settings.worker_chain,
        concurrency=1,
    )
    processed = 0
    for _ in range(max(1, max_jobs)):
        job = await get_queue().dequeue()
        if job is None:
            break
        await worker.process_job(job)
        processed += 1
    return {"processed": processed}


@router.get("/health/queue")
async def queue_health() -> dict:
    from anime_episode.config import get_settings

    settings = get_settings()
    queue = get_queue()
    depth = await queue.depth() if hasattr(queue, "depth") else None
    return {
        "backend": settings.queue_backend,
        "redis_url": settings.redis_url if settings.queue_backend == "redis" else None,
        "depth": depth,
        "worker_concurrency": settings.worker_concurrency,
        "worker_chain": settings.worker_chain,
    }
