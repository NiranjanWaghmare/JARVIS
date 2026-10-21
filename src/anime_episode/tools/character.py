"""Character bible tool — sheets / LoRA hooks before rendering shots."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from anime_episode.config import get_settings
from anime_episode.domain.enums import ArtifactKind
from anime_episode.domain.models import Artifact, Episode
from anime_episode.tools.base import Tool, ToolContext
from anime_episode.tools.keyframe import _placeholder_png


class CharacterBuildInput(BaseModel):
    episode: Episode


class CharacterBuildOutput(BaseModel):
    episode: Episode
    artifacts: list[Artifact] = Field(default_factory=list)


class CharacterBuildTool(Tool[CharacterBuildInput, CharacterBuildOutput]):
    name = "character_build"
    description = "Create character reference sheets used for identity locking."

    async def run(self, inp: CharacterBuildInput, ctx: ToolContext) -> CharacterBuildOutput:
        settings = get_settings()
        episode = inp.episode.model_copy(deep=True)
        out_dir = Path(ctx.artifact_dir) / "characters"
        out_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[Artifact] = []

        for char in episode.characters:
            sheet = out_dir / f"{char.id}_sheet.png"
            if ctx.dry_run or settings.dry_run:
                sheet.write_bytes(_placeholder_png())
            else:
                # Hook: multi-view sheet via Animagine + IP-Adapter / CharForge.
                sheet.write_bytes(_placeholder_png())
            char.sheet_paths = [str(sheet)]
            artifacts.append(
                Artifact(
                    kind=ArtifactKind.CHARACTER_SHEET,
                    path=str(sheet),
                    meta={"character_id": char.id, "name": char.name},
                )
            )

        episode.artifacts.extend(artifacts)
        return CharacterBuildOutput(episode=episode, artifacts=artifacts)
