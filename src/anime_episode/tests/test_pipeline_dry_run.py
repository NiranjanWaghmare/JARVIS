"""Architecture smoke tests — dry-run full episode without GPUs."""

from __future__ import annotations

import asyncio
from pathlib import Path

from anime_episode.config import get_settings
from anime_episode.domain.enums import EpisodeStatus, ShotStage
from anime_episode.domain.models import Episode
from anime_episode.orchestrator.pipeline import stages_for_shot
from anime_episode.orchestrator.runner import PipelineRunner
from anime_episode.domain.enums import ShotKind
from anime_episode.storage.artifact_store import ArtifactStore


def test_dialogue_stages_include_tts() -> None:
    stages = [e.stage for e in stages_for_shot(ShotKind.DIALOGUE, needs_animate=True)]
    assert ShotStage.TTS in stages
    assert ShotStage.DIALOGUE_VIDEO in stages


def test_action_stages_skip_tts() -> None:
    stages = [e.stage for e in stages_for_shot(ShotKind.ACTION, needs_animate=True)]
    assert ShotStage.TTS not in stages
    assert ShotStage.MOTION in stages


def test_dry_run_kickoff(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ANIME_DRY_RUN", "true")
    monkeypatch.setenv("ANIME_ARTIFACT_ROOT", str(tmp_path))
    get_settings.cache_clear()

    store = ArtifactStore(root=tmp_path)
    episode = Episode(
        title="Test Arc",
        premise="A radio whispers tomorrow's weather to a rooftop girl.",
        target_duration_s=120,
    )
    store.save_episode(episode)
    runner = PipelineRunner(store=store)
    result = asyncio.run(runner.kickoff_local(episode.id))

    assert result.status == EpisodeStatus.COMPLETED
    assert result.final_video_path
    assert Path(result.final_video_path).exists()
    assert len(result.shots) >= 4
    assert all(s.stage == ShotStage.DONE for s in result.shots)
    assert all("final_shot" in s.artifacts for s in result.shots)

    get_settings.cache_clear()
