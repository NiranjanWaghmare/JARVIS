"""Motion tool — image-to-video (AniSora / AnimeGen / Wan2.2 I2V)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from anime_episode.config import get_settings
from anime_episode.domain.models import Episode, Shot
from anime_episode.tools.base import Tool, ToolContext
from anime_episode.tools.comfyui_client import ComfyUIClient


class MotionInput(BaseModel):
    episode: Episode
    shot: Shot
    keyframe_path: str


class MotionOutput(BaseModel):
    path: str
    model: str


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

        if ctx.dry_run or settings.dry_run:
            out_path.write_text(f"dry-run motion from {inp.keyframe_path}\n")
            return MotionOutput(path=str(out_path), model=settings.motion_model)

        # Wire AniSora / Wan I2V Comfy workflow here.
        _ = self.comfy
        raise NotImplementedError(
            f"Connect ComfyUI workflow for {settings.motion_model}; "
            "dry_run=false path not yet wired."
        )
