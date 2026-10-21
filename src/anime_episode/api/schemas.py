"""HTTP request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

from anime_episode.domain.enums import EpisodeStatus, JobStatus, JobType, ShotStage
from anime_episode.domain.models import CharacterRef, Scene, Shot


class CreateEpisodeRequest(BaseModel):
    title: str
    premise: str
    style: str = "anime, clean linework, cinematic lighting"
    target_duration_s: int = Field(default=600, ge=30, le=1800)


class CreateEpisodeResponse(BaseModel):
    id: str
    status: EpisodeStatus
    title: str


class KickoffRequest(BaseModel):
    """If sync=true, run the full dry pipeline in-process; else enqueue jobs."""

    sync: bool = False
    enqueue_shots: bool = True


class KickoffResponse(BaseModel):
    episode_id: str
    mode: str
    job_ids: list[str] = Field(default_factory=list)
    status: EpisodeStatus


class ShotStatus(BaseModel):
    id: str
    index: int
    kind: str
    stage: ShotStage
    duration_s: float
    error: str | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)


class EpisodeStatusResponse(BaseModel):
    id: str
    title: str
    status: EpisodeStatus
    target_duration_s: int
    character_count: int
    scene_count: int
    shot_count: int
    shots_done: int
    final_video_path: str | None
    error: str | None
    characters: list[CharacterRef] = Field(default_factory=list)
    scenes: list[Scene] = Field(default_factory=list)
    shots: list[ShotStatus] = Field(default_factory=list)


class JobStatusResponse(BaseModel):
    id: str
    type: JobType
    episode_id: str
    shot_id: str | None
    status: JobStatus
    error: str | None
    result: dict = Field(default_factory=dict)


class UpdateCharacterRequest(BaseModel):
    appearance: str | None = None
    voice_ref_path: str | None = None
    voice_prompt_text: str | None = None
    voice_lang: str | None = None
    lora_path: str | None = None


class CharacterVoiceStatusResponse(BaseModel):
    episode_id: str
    dialogue_shot_count: int
    characters_with_voice: list[str]
    characters_missing_voice: list[str]
    ok_for_dialogue: bool
    characters: list[CharacterRef] = Field(default_factory=list)


def shot_status(shot: Shot) -> ShotStatus:
    return ShotStatus(
        id=shot.id,
        index=shot.index,
        kind=shot.kind.value,
        stage=shot.stage,
        duration_s=shot.duration_s,
        error=shot.error,
        artifacts=shot.artifacts,
    )
