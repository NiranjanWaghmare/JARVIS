"""Episode + artifact persistence (filesystem JSON for scaffolding)."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Callable, Iterator

from anime_episode.config import get_settings
from anime_episode.domain.models import Episode, JobRecord, Shot


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

    @contextmanager
    def _episode_file_lock(self, episode_id: str) -> Iterator[None]:
        """Cross-process lock so parallel shot workers don't clobber episode.json.

        Uses flock only (not self._lock) to avoid deadlocking with job writes.
        """
        lock_path = self.episode_dir(episode_id) / ".episode.lock"
        lock_path.touch(exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as handle:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except ImportError:
                # Non-POSIX: process-local lock only.
                with self._lock:
                    yield

    def save_episode(self, episode: Episode) -> None:
        path = self.episode_dir(episode.id) / "episode.json"
        with self._episode_file_lock(episode.id):
            path.write_text(episode.model_dump_json(indent=2))

    def load_episode(self, episode_id: str) -> Episode:
        path = self.episode_dir(episode_id) / "episode.json"
        if not path.exists():
            raise FileNotFoundError(episode_id)
        with self._episode_file_lock(episode_id):
            return Episode.model_validate_json(path.read_text())

    def mutate_episode(self, episode_id: str, mutator: Callable[[Episode], None]) -> Episode:
        """Load → mutate → save under a single lock (safe for parallel shot updates)."""
        path = self.episode_dir(episode_id) / "episode.json"
        if not path.exists():
            raise FileNotFoundError(episode_id)
        with self._episode_file_lock(episode_id):
            episode = Episode.model_validate_json(path.read_text())
            mutator(episode)
            episode.touch()
            path.write_text(episode.model_dump_json(indent=2))
            return episode

    def update_shot(self, episode_id: str, shot: Shot) -> Episode:
        def _apply(episode: Episode) -> None:
            for i, existing in enumerate(episode.shots):
                if existing.id == shot.id:
                    episode.shots[i] = shot
                    return
            raise KeyError(shot.id)

        return self.mutate_episode(episode_id, _apply)

    def list_episodes(self) -> list[str]:
        return sorted(
            p.name
            for p in self.root.iterdir()
            if p.is_dir() and not p.name.startswith("_")
        )

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

    def try_claim_assemble(self, episode_id: str) -> bool:
        """Atomically claim the assemble step so parallel workers enqueue it once."""
        claim = self.episode_dir(episode_id) / ".assemble.claimed"
        with self._episode_file_lock(episode_id):
            if claim.exists():
                return False
            claim.write_text("1")
            return True

