"""Dialogue video tool — Wan2.2-S2V via ComfyUI."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from anime_episode.config import get_settings
from anime_episode.domain.enums import ShotKind
from anime_episode.domain.models import Episode, Shot
from anime_episode.tools.base import Tool, ToolContext
from anime_episode.tools.comfyui_client import ComfyUIClient
from anime_episode.tools.workflows import (
    first_video,
    load_workflow,
    prepare_s2v_workflow,
)


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
    workflow: str | None = None


class DialogueVideoTool(Tool[DialogueVideoInput, DialogueVideoOutput]):
    name = "dialogue_video"
    description = "Generate lip-synced talking video from keyframe + TTS audio."

    def __init__(self, comfy: ComfyUIClient | None = None) -> None:
        settings = get_settings()
        self.comfy = comfy or ComfyUIClient(settings.comfyui_base_url, settings.comfyui_timeout_s)

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

        prompt = _s2v_prompt(inp.episode, inp.shot)
        negative = inp.shot.negative_prompt or settings.default_negative_prompt

        if ctx.dry_run or settings.dry_run:
            out_path.write_text(
                f"dry-run s2v image={inp.keyframe_path} audio={inp.audio_path}\n"
            )
            return DialogueVideoOutput(
                path=str(out_path),
                model=settings.dialogue_model,
                workflow="dry_run",
            )

        image_name = await self.comfy.upload_image(inp.keyframe_path)
        audio_name = await self.comfy.upload_audio(inp.audio_path)
        template = load_workflow(settings.dialogue_workflow)
        length = _s2v_length(inp.shot.duration_s, settings.dialogue_length)
        workflow = prepare_s2v_workflow(
            template,
            positive=prompt,
            negative=negative,
            image_filename=image_name,
            audio_filename=audio_name,
            width=settings.dialogue_width,
            height=settings.dialogue_height,
            length=length,
            unet_name=settings.dialogue_unet,
        )
        history = await self.comfy.run_workflow(workflow, client_id=ctx.episode_id)
        video = first_video(history)
        data = await self.comfy.download_file(
            video["filename"],
            subfolder=video.get("subfolder", ""),
            folder_type=video.get("type", "output"),
        )
        out_path.write_bytes(data)
        return DialogueVideoOutput(
            path=str(out_path),
            model=settings.dialogue_model,
            workflow=settings.dialogue_workflow,
        )


def _s2v_prompt(episode: Episode, shot: Shot) -> str:
    line = shot.dialogue or ""
    return (
        f"Character speaking: {line}. {shot.description}. "
        f"{shot.visual_prompt}. {episode.style}, natural lip sync, expressive face"
    )


def _s2v_length(duration_s: float, default_length: int) -> int:
    # S2V templates often use chunks of 77 frames (~4.8s at 16fps).
    target = int(round(duration_s * 16))
    if target < 17:
        return default_length
    return max(default_length, target)
