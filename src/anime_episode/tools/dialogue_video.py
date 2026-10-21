"""Dialogue video tool — Wan2.2-S2V audio-driven talking shots."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from anime_episode.config import get_settings
from anime_episode.domain.enums import ShotKind
from anime_episode.domain.models import Episode, Shot
from anime_episode.tools.base import Tool, ToolContext


class DialogueVideoInput(BaseModel):
    episode: Episode
    shot: Shot
    keyframe_path: str
    audio_path: str | None
    fallback_clip_path: str


class DialogueVideoOutput(BaseModel):
    path: str
    skipped: bool = False
    model: str


class DialogueVideoTool(Tool[DialogueVideoInput, DialogueVideoOutput]):
    name = "dialogue_video"
    description = "Generate lip-synced talking video from keyframe + TTS audio."

    async def run(self, inp: DialogueVideoInput, ctx: ToolContext) -> DialogueVideoOutput:
        settings = get_settings()
        if inp.shot.kind != ShotKind.DIALOGUE or not inp.audio_path:
            return DialogueVideoOutput(
                path=inp.fallback_clip_path,
                skipped=True,
                model=settings.dialogue_model,
            )

        out_dir = Path(ctx.artifact_dir) / "dialogue"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{inp.shot.id}.mp4"

        if ctx.dry_run or settings.dry_run:
            out_path.write_text(
                f"dry-run s2v image={inp.keyframe_path} audio={inp.audio_path}\n"
            )
            return DialogueVideoOutput(path=str(out_path), model=settings.dialogue_model)

        raise NotImplementedError(
            f"Connect Wan-S2V backend for {settings.dialogue_model}"
        )
