"""TTS tool — GPT-SoVITS (or Fish Speech) dialogue audio."""

from __future__ import annotations

from pathlib import Path

import httpx
from pydantic import BaseModel

from anime_episode.config import get_settings
from anime_episode.domain.models import CharacterRef, Episode, Shot
from anime_episode.tools.base import Tool, ToolContext


class TTSInput(BaseModel):
    episode: Episode
    shot: Shot


class TTSOutput(BaseModel):
    path: str | None
    skipped: bool = False
    model: str


class TTSTool(Tool[TTSInput, TTSOutput]):
    name = "tts"
    description = "Synthesize dialogue audio for a shot."

    async def run(self, inp: TTSInput, ctx: ToolContext) -> TTSOutput:
        settings = get_settings()
        if not inp.shot.dialogue:
            return TTSOutput(path=None, skipped=True, model=settings.tts_model)

        out_dir = Path(ctx.artifact_dir) / "voice"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{inp.shot.id}.wav"

        if ctx.dry_run or settings.dry_run:
            out_path.write_bytes(b"RIFF$$$$WAVEdry")
            return TTSOutput(path=str(out_path), model=settings.tts_model)

        speaker = _speaker(inp.episode, inp.shot.speaker_character_id)
        async with httpx.AsyncClient(timeout=120) as client:
            # GPT-SoVITS typical API shape; adapt to your server.
            payload = {
                "text": inp.shot.dialogue,
                "text_language": "en",
                "ref_audio_path": speaker.voice_ref_path if speaker else None,
            }
            r = await client.post(f"{settings.tts_base_url}/tts", json=payload)
            r.raise_for_status()
            out_path.write_bytes(r.content)
        return TTSOutput(path=str(out_path), model=settings.tts_model)


def _speaker(episode: Episode, character_id: str | None) -> CharacterRef | None:
    if not character_id:
        return None
    for c in episode.characters:
        if c.id == character_id:
            return c
    return None
