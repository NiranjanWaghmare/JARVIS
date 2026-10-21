"""Helpers for character voice refs and sheet directories under an episode."""

from __future__ import annotations

import shutil
from pathlib import Path

from anime_episode.domain.models import CharacterRef, Episode


def character_dir(artifact_dir: str | Path, character_id: str) -> Path:
    path = Path(artifact_dir) / "characters" / character_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def install_voice_ref(
    *,
    artifact_dir: str | Path,
    character: CharacterRef,
    source_path: str | Path,
    prompt_text: str | None = None,
    lang: str | None = None,
) -> CharacterRef:
    """Copy a reference WAV into the episode character folder and update the ref."""
    src = Path(source_path)
    if not src.exists():
        raise FileNotFoundError(f"voice ref not found: {src}")
    dest_dir = character_dir(artifact_dir, character.id)
    dest = dest_dir / f"voice_ref{src.suffix.lower() or '.wav'}"
    shutil.copy2(src, dest)
    character.voice_ref_path = str(dest)
    if prompt_text is not None:
        character.voice_prompt_text = prompt_text
    if lang is not None:
        character.voice_lang = lang
    return character


def voice_readiness(episode: Episode) -> dict:
    """Report which characters still need voice refs for dialogue shots."""
    speakers = {
        s.speaker_character_id
        for s in episode.shots
        if s.dialogue and s.speaker_character_id
    }
    ready: list[str] = []
    missing: list[str] = []
    for char in episode.characters:
        if char.id not in speakers and not speakers:
            # No dialogue speakers yet — still report voice status.
            pass
        has_voice = bool(char.voice_ref_path and Path(char.voice_ref_path).exists())
        if has_voice:
            ready.append(char.name)
        else:
            missing.append(char.name)
    dialogue_shots = sum(1 for s in episode.shots if s.dialogue)
    return {
        "dialogue_shot_count": dialogue_shots,
        "characters_with_voice": ready,
        "characters_missing_voice": missing,
        "ok_for_dialogue": dialogue_shots == 0 or (bool(speakers) and not missing),
    }


def update_character_in_episode(episode: Episode, character_id: str, **fields) -> CharacterRef:
    for i, char in enumerate(episode.characters):
        if char.id != character_id:
            continue
        data = char.model_dump()
        data.update({k: v for k, v in fields.items() if v is not None})
        updated = CharacterRef.model_validate(data)
        episode.characters[i] = updated
        return updated
    raise KeyError(character_id)
