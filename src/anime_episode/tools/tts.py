"""TTS tool — GPT-SoVITS dialogue audio."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from anime_episode.config import get_settings
from anime_episode.domain.models import CharacterRef, Episode, Shot
from anime_episode.tools.base import Tool, ToolContext
from anime_episode.tools.tts_client import GPTSoVITSClient


class TTSInput(BaseModel):
    episode: Episode
    shot: Shot


class TTSOutput(BaseModel):
    path: str | None
    skipped: bool = False
    model: str
    speaker: str | None = None


class TTSTool(Tool[TTSInput, TTSOutput]):
    name = "tts"
    description = "Synthesize dialogue audio for a shot."

    def __init__(self, client: GPTSoVITSClient | None = None) -> None:
        settings = get_settings()
        self.client = client or GPTSoVITSClient(settings.tts_base_url, settings.tts_timeout_s)

    async def run(self, inp: TTSInput, ctx: ToolContext) -> TTSOutput:
        settings = get_settings()
        if not inp.shot.dialogue:
            return TTSOutput(path=None, skipped=True, model=settings.tts_model)

        out_dir = Path(ctx.artifact_dir) / "voice"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{inp.shot.id}.wav"

        speaker = _speaker(inp.episode, inp.shot.speaker_character_id)

        if ctx.dry_run or settings.dry_run:
            out_path.write_bytes(b"RIFF$$$$WAVEdry")
            return TTSOutput(
                path=str(out_path),
                model=settings.tts_model,
                speaker=speaker.name if speaker else None,
            )

        ref_path, prompt_text, text_lang, prompt_lang = _voice_params(speaker, settings)
        audio = await self.client.synthesize(
            text=inp.shot.dialogue,
            text_lang=text_lang,
            ref_audio_path=ref_path,
            prompt_lang=prompt_lang,
            prompt_text=prompt_text,
            media_type="wav",
            speed_factor=settings.tts_speed,
        )
        out_path.write_bytes(audio)
        return TTSOutput(
            path=str(out_path),
            model=settings.tts_model,
            speaker=speaker.name if speaker else None,
        )


def _speaker(episode: Episode, character_id: str | None) -> CharacterRef | None:
    if not character_id:
        return None
    for c in episode.characters:
        if c.id == character_id:
            return c
    return None


def _voice_params(
    speaker: CharacterRef | None,
    settings,
) -> tuple[str, str, str, str]:
    ref = (speaker.voice_ref_path if speaker and speaker.voice_ref_path else None) or settings.tts_default_ref_audio
    if not ref:
        raise RuntimeError(
            "No voice reference: set character.voice_ref_path or ANIME_TTS_DEFAULT_REF_AUDIO"
        )
    prompt_text = (
        (speaker.voice_prompt_text if speaker and speaker.voice_prompt_text else None)
        or settings.tts_default_prompt_text
    )
    text_lang = (
        (speaker.voice_lang if speaker and speaker.voice_lang else None)
        or settings.tts_text_lang
    )
    prompt_lang = (
        (speaker.voice_lang if speaker and speaker.voice_lang else None)
        or settings.tts_prompt_lang
    )
    return ref, prompt_text, text_lang, prompt_lang
