"""Domain models for story → episode generation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from anime_episode.domain.enums import (
    ArtifactKind,
    EpisodeStatus,
    JobStatus,
    JobType,
    ShotKind,
    ShotStage,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class CharacterRef(BaseModel):
    id: str = Field(default_factory=lambda: _id("char"))
    name: str
    appearance: str
    voice_ref_path: str | None = None
    sheet_paths: list[str] = Field(default_factory=list)
    lora_path: str | None = None
    ip_adapter_refs: list[str] = Field(default_factory=list)


class CameraNotes(BaseModel):
    framing: str = "medium"
    movement: str = "static"
    angle: str = "eye_level"


class Shot(BaseModel):
    id: str = Field(default_factory=lambda: _id("shot"))
    index: int
    kind: ShotKind = ShotKind.ACTION
    duration_s: float = 5.0
    scene_id: str
    description: str
    visual_prompt: str
    negative_prompt: str = "low quality, blurry, deformed hands"
    dialogue: str | None = None
    speaker_character_id: str | None = None
    character_ids: list[str] = Field(default_factory=list)
    camera: CameraNotes = Field(default_factory=CameraNotes)
    stage: ShotStage = ShotStage.PENDING
    needs_animate: bool = True
    artifacts: dict[str, str] = Field(default_factory=dict)
    error: str | None = None


class Scene(BaseModel):
    id: str = Field(default_factory=lambda: _id("scene"))
    index: int
    title: str
    location: str
    summary: str
    shot_ids: list[str] = Field(default_factory=list)


class Artifact(BaseModel):
    id: str = Field(default_factory=lambda: _id("art"))
    kind: ArtifactKind
    path: str
    shot_id: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_now)


class Episode(BaseModel):
    id: str = Field(default_factory=lambda: _id("ep"))
    title: str
    premise: str
    style: str = "anime, clean linework, cinematic lighting"
    target_duration_s: int = 600
    status: EpisodeStatus = EpisodeStatus.DRAFT
    characters: list[CharacterRef] = Field(default_factory=list)
    scenes: list[Scene] = Field(default_factory=list)
    shots: list[Shot] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    final_video_path: str | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    error: str | None = None

    def shot_by_id(self, shot_id: str) -> Shot:
        for shot in self.shots:
            if shot.id == shot_id:
                return shot
        raise KeyError(shot_id)

    def touch(self) -> None:
        self.updated_at = _now()


class JobRecord(BaseModel):
    id: str = Field(default_factory=lambda: _id("job"))
    type: JobType
    episode_id: str
    shot_id: str | None = None
    status: JobStatus = JobStatus.QUEUED
    attempts: int = 0
    max_attempts: int = 3
    payload: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    created_at: datetime = Field(default_factory=_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
