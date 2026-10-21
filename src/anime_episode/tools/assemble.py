"""Assemble tool — mux dialogue audio, then stitch shots into final episode."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from pydantic import BaseModel, Field

from anime_episode.config import get_settings
from anime_episode.domain.models import Episode, Shot
from anime_episode.tools.base import Tool, ToolContext


class AssembleInput(BaseModel):
    episode: Episode
    shot_clip_paths: list[str] = Field(default_factory=list)


class AssembleOutput(BaseModel):
    path: str
    shot_count: int
    muxed_shots: int = 0


class AssembleTool(Tool[AssembleInput, AssembleOutput]):
    name = "assemble"
    description = "Mux TTS audio into dialogue clips and concatenate into the final MP4."

    async def run(self, inp: AssembleInput, ctx: ToolContext) -> AssembleOutput:
        settings = get_settings()
        out_dir = Path(ctx.artifact_dir) / "final"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{inp.episode.id}.mp4"
        mux_dir = Path(ctx.artifact_dir) / "muxed"
        mux_dir.mkdir(parents=True, exist_ok=True)

        shots = sorted(inp.episode.shots, key=lambda s: s.index)
        muxed_count = 0
        prepared: list[str] = []

        if inp.shot_clip_paths:
            prepared = list(inp.shot_clip_paths)
        else:
            for shot in shots:
                clip = shot.artifacts.get("final_shot")
                if not clip:
                    continue
                voice = shot.artifacts.get("voice")
                if voice and settings.mux_dialogue_audio:
                    muxed_path = mux_dir / f"{shot.id}.mp4"
                    if ctx.dry_run or settings.dry_run:
                        muxed_path.write_text(f"mux {clip} + {voice}\n")
                        prepared.append(str(muxed_path))
                        muxed_count += 1
                    else:
                        await _mux_audio(clip, voice, str(muxed_path), settings.ffmpeg_bin)
                        prepared.append(str(muxed_path))
                        muxed_count += 1
                else:
                    prepared.append(clip)

        if not prepared:
            raise RuntimeError("No shot clips available to assemble")

        if ctx.dry_run or settings.dry_run:
            out_path.write_text("\n".join(prepared) + "\n")
            return AssembleOutput(path=str(out_path), shot_count=len(prepared), muxed_shots=muxed_count)

        list_file = out_dir / "concat.txt"
        list_file.write_text("".join(f"file '{Path(p).resolve()}'\n" for p in prepared))
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
        return AssembleOutput(path=str(out_path), shot_count=len(prepared), muxed_shots=muxed_count)


async def _mux_audio(video_path: str, audio_path: str, out_path: str, ffmpeg_bin: str) -> None:
    """Replace/add audio track; keep video stream, shortest duration wins."""
    if not shutil.which(ffmpeg_bin):
        raise RuntimeError(f"ffmpeg not found: {ffmpeg_bin}")
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        out_path,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed: {stderr.decode()[-2000:]}")


def shot_needs_mux(shot: Shot) -> bool:
    return bool(shot.artifacts.get("voice") and shot.artifacts.get("final_shot"))
