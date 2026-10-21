"""Assemble tool — upscale, interpolate, stitch shots into final episode."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from pydantic import BaseModel, Field

from anime_episode.config import get_settings
from anime_episode.domain.models import Episode
from anime_episode.tools.base import Tool, ToolContext


class AssembleInput(BaseModel):
    episode: Episode
    shot_clip_paths: list[str] = Field(default_factory=list)


class AssembleOutput(BaseModel):
    path: str
    shot_count: int


class AssembleTool(Tool[AssembleInput, AssembleOutput]):
    name = "assemble"
    description = "Concatenate per-shot clips into the final episode MP4."

    async def run(self, inp: AssembleInput, ctx: ToolContext) -> AssembleOutput:
        settings = get_settings()
        out_dir = Path(ctx.artifact_dir) / "final"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{inp.episode.id}.mp4"

        paths = inp.shot_clip_paths or [
            shot.artifacts.get("final_shot")
            for shot in sorted(inp.episode.shots, key=lambda s: s.index)
            if shot.artifacts.get("final_shot")
        ]
        paths = [p for p in paths if p]
        if not paths:
            raise RuntimeError("No shot clips available to assemble")

        if ctx.dry_run or settings.dry_run:
            out_path.write_text("\n".join(paths) + "\n")
            return AssembleOutput(path=str(out_path), shot_count=len(paths))

        list_file = out_dir / "concat.txt"
        list_file.write_text("".join(f"file '{Path(p).resolve()}'\n" for p in paths))
        cmd = [
            settings.ffmpeg_bin,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            str(out_path),
        ]
        if not shutil.which(settings.ffmpeg_bin):
            raise RuntimeError(f"ffmpeg not found: {settings.ffmpeg_bin}")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(stderr.decode()[-2000:])
        return AssembleOutput(path=str(out_path), shot_count=len(paths))
