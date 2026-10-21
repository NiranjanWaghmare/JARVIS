"""Episode + artifact persistence (filesystem JSON for scaffolding)."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

from anime_episode.config import get_settings
from anime_episode.domain.models import Episode, JobRecord


class ArtifactStore:
    """Simple durable store. Swap for Postgres/S3 later without changing callers."""

    def __init__(self, root: Path | None = None) -> None:
        settings = get_settings()
        self.root = Path(root or settings.artifact_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def episode_dir(self, episode_id: str) -> Path:
        path = self.root / episode_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_episode(self, episode: Episode) -> None:
        path = self.episode_dir(episode.id) / "episode.json"
        with self._lock:
            path.write_text(episode.model_dump_json(indent=2))

    def load_episode(self, episode_id: str) -> Episode:
        path = self.episode_dir(episode_id) / "episode.json"
        if not path.exists():
            raise FileNotFoundError(episode_id)
        return Episode.model_validate_json(path.read_text())

    def list_episodes(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def save_job(self, job: JobRecord) -> None:
        path = self.root / "_jobs" / f"{job.id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            path.write_text(job.model_dump_json(indent=2))

    def load_job(self, job_id: str) -> JobRecord:
        path = self.root / "_jobs" / f"{job_id}.json"
        if not path.exists():
            raise FileNotFoundError(job_id)
        return JobRecord.model_validate_json(path.read_text())

    def list_jobs(self, episode_id: str | None = None) -> list[JobRecord]:
        jobs_dir = self.root / "_jobs"
        if not jobs_dir.exists():
            return []
        jobs: list[JobRecord] = []
        for path in sorted(jobs_dir.glob("*.json")):
            job = JobRecord.model_validate_json(path.read_text())
            if episode_id is None or job.episode_id == episode_id:
                jobs.append(job)
        return jobs

    def dump_meta(self, episode_id: str, name: str, payload: dict) -> Path:
        path = self.episode_dir(episode_id) / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2))
        return path
