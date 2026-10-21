"""Identity lock tool — Wan2.2-Animate face/character consistency pass."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from anime_episode.config import get_settings
from anime_episode.domain.models import Episode, Shot
from anime_episode.tools.base import Tool, ToolContext


class AnimateInput(BaseModel):
    episode: Episode
    shot: Shot
    keyframe_path: str
    driving_clip_path: str


class AnimateOutput(BaseModel):
    path: str
    model: str
    skipped: bool = False


class AnimateTool(Tool[AnimateInput, AnimateOutput]):
    name = "animate"
    description = "Lock character identity onto a driving motion clip."

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

        if ctx.dry_run or settings.dry_run:
            out_path.write_text(
                f"dry-run animate ref={inp.keyframe_path} drive={inp.driving_clip_path}\n"
            )
            return AnimateOutput(path=str(out_path), model=settings.animate_model)

        raise NotImplementedError(
            f"Connect Wan-Animate backend for {settings.animate_model}"
        )
