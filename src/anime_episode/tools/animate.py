"""Identity lock tool — Wan2.2-Animate via ComfyUI."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from anime_episode.config import get_settings
from anime_episode.domain.models import Episode, Shot
from anime_episode.tools.base import Tool, ToolContext
from anime_episode.tools.comfyui_client import ComfyUIClient
from anime_episode.tools.workflows import (
    first_video,
    load_workflow,
    prepare_animate_workflow,
)


class AnimateInput(BaseModel):
    episode: Episode
    shot: Shot
    keyframe_path: str
    driving_clip_path: str


class AnimateOutput(BaseModel):
    path: str
    model: str
    skipped: bool = False
    workflow: str | None = None


class AnimateTool(Tool[AnimateInput, AnimateOutput]):
    name = "animate"
    description = "Lock character identity onto a driving motion clip."

    def __init__(self, comfy: ComfyUIClient | None = None) -> None:
        settings = get_settings()
        self.comfy = comfy or ComfyUIClient(settings.comfyui_base_url, settings.comfyui_timeout_s)

    async def run(self, inp: AnimateInput, ctx: ToolContext) -> AnimateOutput:
        settings = get_settings()
        out_dir = Path(ctx.artifact_dir) / "animate"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{inp.shot.id}.mp4"

        if not inp.shot.needs_animate:
            return AnimateOutput(
                path=inp.driving_clip_path,
                model=settings.animate_model,
                skipped=True,
            )

        prompt = _animate_prompt(inp.episode, inp.shot)
        negative = inp.shot.negative_prompt or settings.default_negative_prompt

        if ctx.dry_run or settings.dry_run:
            out_path.write_text(
                f"dry-run animate ref={inp.keyframe_path} drive={inp.driving_clip_path}\n"
            )
            return AnimateOutput(
                path=str(out_path),
                model=settings.animate_model,
                workflow="dry_run",
            )

        image_name = await self.comfy.upload_image(inp.keyframe_path)
        video_name = await self.comfy.upload_video(inp.driving_clip_path)
        template = load_workflow(settings.animate_workflow)
        length = _wan_length(inp.shot.duration_s, settings.animate_length)
        workflow = prepare_animate_workflow(
            template,
            positive=prompt,
            negative=negative,
            image_filename=image_name,
            video_filename=video_name,
            width=settings.animate_width,
            height=settings.animate_height,
            length=length,
            unet_name=settings.animate_unet,
        )
        history = await self.comfy.run_workflow(workflow, client_id=ctx.episode_id)
        video = first_video(history)
        data = await self.comfy.download_file(
            video["filename"],
            subfolder=video.get("subfolder", ""),
            folder_type=video.get("type", "output"),
        )
        out_path.write_bytes(data)
        return AnimateOutput(
            path=str(out_path),
            model=settings.animate_model,
            workflow=settings.animate_workflow,
        )


def _animate_prompt(episode: Episode, shot: Shot) -> str:
    return (
        f"{shot.description}. Preserve exact character identity from reference. "
        f"{shot.visual_prompt}. {episode.style}, natural anime motion"
    )


def _wan_length(duration_s: float, default_length: int) -> int:
    target = int(round(duration_s * 16))
    if target < 5:
        target = default_length
    n = max(1, round((target - 1) / 4))
    return 4 * n + 1
