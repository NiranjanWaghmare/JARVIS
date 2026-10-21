"""Character voice asset helpers + API."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from anime_episode.api.deps import get_queue, get_runner, get_store
from anime_episode.config import get_settings
from anime_episode.domain.models import CharacterRef, Episode, Shot
from anime_episode.domain.enums import ShotKind
from anime_episode.main import app
from anime_episode.storage.artifact_store import ArtifactStore
from anime_episode.tools.character_assets import install_voice_ref, voice_readiness
from anime_episode.tools.workflow_validate import validate_all
from anime_episode.smoke_check import run_checks
import asyncio


def test_workflow_markers_valid() -> None:
    results = validate_all()
    assert all(r.ok for r in results), results


def test_install_voice_and_readiness(tmp_path: Path) -> None:
    wav = tmp_path / "aiko.wav"
    wav.write_bytes(b"RIFF")
    char = CharacterRef(name="Aiko", appearance="black hair")
    episode = Episode(title="t", premise="p", characters=[char])
    episode.shots = [
        Shot(
            index=0,
            kind=ShotKind.DIALOGUE,
            scene_id="s",
            description="talk",
            visual_prompt="face",
            dialogue="hi",
            speaker_character_id=char.id,
        )
    ]
    status = voice_readiness(episode)
    assert "Aiko" in status["characters_missing_voice"]

    install_voice_ref(
        artifact_dir=tmp_path / episode.id,
        character=char,
        source_path=wav,
        prompt_text="hello there",
        lang="en",
    )
    episode.characters[0] = char
    status = voice_readiness(episode)
    assert status["ok_for_dialogue"]
    assert Path(char.voice_ref_path).exists()


def test_character_patch_api(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ANIME_DRY_RUN", "true")
    monkeypatch.setenv("ANIME_ARTIFACT_ROOT", str(tmp_path))
    get_settings.cache_clear()
    get_store.cache_clear()
    get_queue.cache_clear()
    get_runner.cache_clear()

    store = ArtifactStore(root=tmp_path)
    char = CharacterRef(name="Ren", appearance="silver hair")
    episode = Episode(title="t", premise="p", characters=[char])
    store.save_episode(episode)

    client = TestClient(app)
    patched = client.patch(
        f"/v1/episodes/{episode.id}/characters/{char.id}",
        json={"voice_prompt_text": "sample line", "voice_lang": "en"},
    )
    assert patched.status_code == 200
    assert patched.json()["character"]["voice_prompt_text"] == "sample line"

    wav = tmp_path / "ren.wav"
    wav.write_bytes(b"RIFF")
    uploaded = client.post(
        f"/v1/episodes/{episode.id}/characters/{char.id}/voice",
        params={"file_path": str(wav), "prompt_text": "sample line", "lang": "en"},
    )
    assert uploaded.status_code == 200
    assert uploaded.json()["character"]["voice_ref_path"]

    status = client.get(f"/v1/episodes/{episode.id}/characters/voice-status")
    assert status.status_code == 200
    assert "Ren" in status.json()["characters_with_voice"]

    get_settings.cache_clear()
    get_store.cache_clear()
    get_queue.cache_clear()
    get_runner.cache_clear()


def test_smoke_check_offline() -> None:
    report = asyncio.run(run_checks(live=False))
    assert report["checks"]["workflows"]["ok"]
    assert report["code_location"].startswith("GitHub")
    assert "Hugging Face" in report["weights_location"]
