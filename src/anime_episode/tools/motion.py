"""Motion tool — image-to-video (AniSora / AnimeGen / Wan2.2 I2V via ComfyUI)."""

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
    prepare_motion_workflow,
)


class MotionInput(BaseModel):
    episode: Episode
    shot: Shot
    keyframe_path: str


class MotionOutput(BaseModel):
    path: str
    model: str
    workflow: str | None = None
    uploaded_image: str | None = None


class MotionTool(Tool[MotionInput, MotionOutput]):
    name = "motion"
    description = "Animate a keyframe into a short clip via anime I2V."

    def __init__(self, comfy: ComfyUIClient | None = None) -> None:
        settings = get_settings()
        self.comfy = comfy or ComfyUIClient(settings.comfyui_base_url, settings.comfyui_timeout_s)

    async def run(self, inp: MotionInput, ctx: ToolContext) -> MotionOutput:
        settings = get_settings()
        out_dir = Path(ctx.artifact_dir) / "motion"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{inp.shot.id}.mp4"

        motion_prompt = _motion_prompt(inp.episode, inp.shot)
        negative = inp.shot.negative_prompt or settings.default_negative_prompt

        if ctx.dry_run or settings.dry_run:
            out_path.write_text(f"dry-run motion from {inp.keyframe_path}\n")
            return MotionOutput(
                path=str(out_path),
                model=settings.motion_model,
                workflow="dry_run",
            )

        uploaded = await self.comfy.upload_image(inp.keyframe_path)
        template = load_workflow(settings.motion_workflow)
        # length ≈ duration_s * 16fps, snapped to Wan-friendly 4n+1
        length = _wan_length(inp.shot.duration_s, settings.motion_length)
        workflow = prepare_motion_workflow(
            template,
            positive=motion_prompt,
            negative=negative,
            image_filename=uploaded,
            width=settings.motion_width,
            height=settings.motion_height,
            length=length,
            high_noise=settings.motion_unet_high,
            low_noise=settings.motion_unet_low,
        )
        history = await self.comfy.run_workflow(workflow, client_id=ctx.episode_id)
        video = first_video(history)
        data = await self.comfy.download_file(
            video["filename"],
            subfolder=video.get("subfolder", ""),
            folder_type=video.get("type", "output"),
        )
        out_path.write_bytes(data)
        return MotionOutput(
            path=str(out_path),
            model=settings.motion_model,
            workflow=settings.motion_workflow,
            uploaded_image=uploaded,
        )


def _motion_prompt(episode: Episode, shot: Shot) -> str:
    cam = shot.camera
    return (
        f"{shot.description}. {shot.visual_prompt}. "
        f"camera {cam.framing}, {cam.movement}, {cam.angle}. "
        f"{episode.style}, smooth anime motion, consistent character design"
    )


def _wan_length(duration_s: float, default_length: int) -> int:
    """Wan I2V prefers frame counts of the form 4n+1."""
    target = int(round(duration_s * 16))
    if target < 5:
        target = default_length
    # snap down/up to 4n+1
    n = max(1, round((target - 1) / 4))
    return 4 * n + 1
