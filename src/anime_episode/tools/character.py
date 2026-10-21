"""Character bible tool — sheets / voice layout before rendering shots."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from anime_episode.config import get_settings
from anime_episode.domain.enums import ArtifactKind
from anime_episode.domain.models import Artifact, CharacterRef, Episode
from anime_episode.tools.base import Tool, ToolContext
from anime_episode.tools.character_assets import character_dir, voice_readiness
from anime_episode.tools.comfyui_client import ComfyUIClient
from anime_episode.tools.keyframe import _placeholder_png
from anime_episode.tools.workflows import (
    first_image,
    load_workflow,
    prepare_keyframe_workflow,
)


class CharacterBuildInput(BaseModel):
    episode: Episode


class CharacterBuildOutput(BaseModel):
    episode: Episode
    artifacts: list[Artifact] = Field(default_factory=list)
    voice_status: dict = Field(default_factory=dict)


class CharacterBuildTool(Tool[CharacterBuildInput, CharacterBuildOutput]):
    name = "character_build"
    description = "Create character reference sheets used for identity locking."

    def __init__(self, comfy: ComfyUIClient | None = None) -> None:
        settings = get_settings()
        self.comfy = comfy or ComfyUIClient(settings.comfyui_base_url, settings.comfyui_timeout_s)

    async def run(self, inp: CharacterBuildInput, ctx: ToolContext) -> CharacterBuildOutput:
        settings = get_settings()
        episode = inp.episode.model_copy(deep=True)
        artifacts: list[Artifact] = []

        for char in episode.characters:
            cdir = character_dir(ctx.artifact_dir, char.id)
            sheet = cdir / "sheet_front.png"
            prompt = (
                f"character design sheet, full body, front view, white background, "
                f"{char.name}, {char.appearance}, {episode.style}, masterpiece, best quality"
            )
            if ctx.dry_run or settings.dry_run:
                sheet.write_bytes(_placeholder_png())
            else:
                await self._render_sheet(sheet, prompt, ctx.episode_id)
            char.sheet_paths = [str(sheet)]
            # Preserve existing voice refs; ensure folder exists for uploads.
            if char.voice_ref_path and not Path(char.voice_ref_path).exists():
                char.voice_ref_path = None
            artifacts.append(
                Artifact(
                    kind=ArtifactKind.CHARACTER_SHEET,
                    path=str(sheet),
                    meta={"character_id": char.id, "name": char.name},
                )
            )

        episode.artifacts.extend(artifacts)
        status = voice_readiness(episode)
        return CharacterBuildOutput(episode=episode, artifacts=artifacts, voice_status=status)

    async def _render_sheet(self, out_path: Path, prompt: str, client_id: str) -> None:
        settings = get_settings()
        template = load_workflow(settings.keyframe_workflow)
        workflow = prepare_keyframe_workflow(
            template,
            positive=prompt,
            negative=settings.default_negative_prompt + ", multiple views collage",
            ckpt_name=settings.keyframe_ckpt,
            width=settings.keyframe_width,
            height=settings.keyframe_height,
        )
        history = await self.comfy.run_workflow(workflow, client_id=client_id)
        image = first_image(history)
        data = await self.comfy.download_file(
            image["filename"],
            subfolder=image.get("subfolder", ""),
            folder_type=image.get("type", "output"),
        )
        out_path.write_bytes(data)


def ensure_character_folders(episode: Episode, artifact_dir: str | Path) -> list[CharacterRef]:
    for char in episode.characters:
        character_dir(artifact_dir, char.id)
    return episode.characters
