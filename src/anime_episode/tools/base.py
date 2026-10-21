"""Tool protocol — every pipeline stage is a Tool with typed I/O."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

from anime_episode.domain.models import Episode, Shot

InT = TypeVar("InT", bound=BaseModel)
OutT = TypeVar("OutT", bound=BaseModel)


class ToolContext(BaseModel):
    """Shared runtime context passed into every tool call."""

    episode_id: str
    dry_run: bool = True
    artifact_dir: str
    extra: dict[str, Any] = Field(default_factory=dict)


class Tool(ABC, Generic[InT, OutT]):
    name: str
    description: str

    @abstractmethod
    async def run(self, inp: InT, ctx: ToolContext) -> OutT:
        raise NotImplementedError


class EpisodeAwareInput(BaseModel):
    episode: Episode


class ShotAwareInput(BaseModel):
    episode: Episode
    shot: Shot
