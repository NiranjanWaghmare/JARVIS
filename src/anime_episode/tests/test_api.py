"""API smoke tests for create + sync kickoff."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from anime_episode.api.deps import get_queue, get_runner, get_store
from anime_episode.config import get_settings
from anime_episode.main import app


def test_create_and_sync_kickoff(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ANIME_DRY_RUN", "true")
    monkeypatch.setenv("ANIME_ARTIFACT_ROOT", str(tmp_path))
    get_settings.cache_clear()
    get_store.cache_clear()
    get_queue.cache_clear()
    get_runner.cache_clear()

    client = TestClient(app)
    created = client.post(
        "/v1/episodes",
        json={
            "title": "Neon Rain",
            "premise": "Two students chase a glitch through the city.",
            "target_duration_s": 120,
        },
    )
    assert created.status_code == 200
    episode_id = created.json()["id"]

    kicked = client.post(
        f"/v1/episodes/{episode_id}/kickoff",
        json={"sync": True},
    )
    assert kicked.status_code == 200
    body = kicked.json()
    assert body["status"] == "completed"

    status = client.get(f"/v1/episodes/{episode_id}")
    assert status.status_code == 200
    data = status.json()
    assert data["shot_count"] >= 4
    assert data["shots_done"] == data["shot_count"]
    assert data["final_video_path"]

    get_settings.cache_clear()
    get_store.cache_clear()
    get_queue.cache_clear()
    get_runner.cache_clear()
