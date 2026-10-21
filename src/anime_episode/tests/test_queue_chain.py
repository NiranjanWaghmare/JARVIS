"""Tests for DAG chaining, parallel-safe shot updates, and audio mux dry-run."""

from __future__ import annotations

import asyncio
from pathlib import Path

from anime_episode.config import get_settings
from anime_episode.domain.enums import JobStatus, JobType, ShotStage
from anime_episode.domain.models import Episode, JobRecord, Shot
from anime_episode.domain.enums import ShotKind
from anime_episode.orchestrator.runner import PipelineRunner
from anime_episode.queue.jobs import MemoryJobQueue
from anime_episode.queue.worker import Worker
from anime_episode.storage.artifact_store import ArtifactStore
from anime_episode.tools.assemble import AssembleInput, AssembleTool
from anime_episode.tools.base import ToolContext


def test_worker_chains_plan_to_assemble(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ANIME_DRY_RUN", "true")
    monkeypatch.setenv("ANIME_ARTIFACT_ROOT", str(tmp_path))
    get_settings.cache_clear()

    store = ArtifactStore(root=tmp_path)
    queue = MemoryJobQueue(store)
    runner = PipelineRunner(store=store)
    worker = Worker(queue=queue, store=store, runner=runner, chain=True, concurrency=1)

    episode = Episode(
        title="Chain Test",
        premise="A rooftop chase under neon rain.",
        target_duration_s=120,
    )
    store.save_episode(episode)

    plan = JobRecord(type=JobType.PLAN_EPISODE, episode_id=episode.id, payload={"chain": True})
    asyncio.run(queue.enqueue(plan))

    # Drain until assemble succeeds or safety limit.
    for _ in range(64):
        job = asyncio.run(queue.dequeue())
        if job is None:
            break
        asyncio.run(worker.process_job(job))

    final = store.load_episode(episode.id)
    assert final.status.value == "completed"
    assert final.final_video_path
    assert all(s.stage == ShotStage.DONE for s in final.shots)

    jobs = store.list_jobs(episode.id)
    assert any(j.type == JobType.ASSEMBLE_EPISODE and j.status == JobStatus.SUCCEEDED for j in jobs)
    # Only one assemble claim / job succeeded
    assemble_jobs = [j for j in jobs if j.type == JobType.ASSEMBLE_EPISODE]
    assert len(assemble_jobs) == 1

    get_settings.cache_clear()


def test_parallel_shot_updates_do_not_lose_shots(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ANIME_DRY_RUN", "true")
    monkeypatch.setenv("ANIME_ARTIFACT_ROOT", str(tmp_path))
    get_settings.cache_clear()

    store = ArtifactStore(root=tmp_path)
    episode = Episode(title="Lock", premise="parallel shots", target_duration_s=60)
    episode.shots = [
        Shot(index=0, kind=ShotKind.ACTION, scene_id="s", description="a", visual_prompt="a"),
        Shot(index=1, kind=ShotKind.ACTION, scene_id="s", description="b", visual_prompt="b"),
    ]
    store.save_episode(episode)
    runner = PipelineRunner(store=store)

    async def _both() -> None:
        await asyncio.gather(
            runner.render_shot(episode.id, episode.shots[0].id),
            runner.render_shot(episode.id, episode.shots[1].id),
        )

    asyncio.run(_both())
    loaded = store.load_episode(episode.id)
    assert all(s.stage == ShotStage.DONE for s in loaded.shots)
    assert all("final_shot" in s.artifacts for s in loaded.shots)

    get_settings.cache_clear()


def test_assemble_muxes_voice_in_dry_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ANIME_DRY_RUN", "true")
    monkeypatch.setenv("ANIME_MUX_DIALOGUE_AUDIO", "true")
    get_settings.cache_clear()

    episode = Episode(title="Mux", premise="talk")
    shot = Shot(
        index=0,
        kind=ShotKind.DIALOGUE,
        scene_id="s",
        description="line",
        visual_prompt="face",
        dialogue="hello",
        artifacts={
            "final_shot": str(tmp_path / "clip.mp4"),
            "voice": str(tmp_path / "line.wav"),
        },
    )
    episode.shots = [shot]
    (tmp_path / "clip.mp4").write_text("v")
    (tmp_path / "line.wav").write_text("a")

    tool = AssembleTool()
    ctx = ToolContext(episode_id=episode.id, dry_run=True, artifact_dir=str(tmp_path))
    out = asyncio.run(tool.run(AssembleInput(episode=episode), ctx))
    assert out.muxed_shots == 1
    assert Path(out.path).exists()
    mux_note = (tmp_path / "muxed" / f"{shot.id}.mp4").read_text()
    assert "mux" in mux_note

    get_settings.cache_clear()
