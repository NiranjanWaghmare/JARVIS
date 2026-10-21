"""Preflight checks for anime episode pipeline (no GPU generation required).

Usage:
  PYTHONPATH=src python -m anime_episode.smoke_check
  PYTHONPATH=src python -m anime_episode.smoke_check --live
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from anime_episode.config import get_settings
from anime_episode.tools.comfyui_client import ComfyUIClient
from anime_episode.tools.tts_client import GPTSoVITSClient
from anime_episode.tools.workflow_validate import validate_all


async def run_checks(*, live: bool) -> dict:
    settings = get_settings()
    report: dict = {
        "dry_run": settings.dry_run,
        "queue_backend": settings.queue_backend,
        "code_location": "GitHub NiranjanWaghmare/JARVIS src/anime_episode/",
        "weights_location": "Hugging Face → local ComfyUI/GPT-SoVITS (see MODELS.md)",
        "checks": {},
    }

    wf_results = validate_all()
    report["checks"]["workflows"] = {
        "ok": all(r.ok for r in wf_results),
        "details": [
            {"name": r.name, "ok": r.ok, "missing": r.missing, "error": r.error}
            for r in wf_results
        ],
    }

    report["checks"]["artifact_root"] = {
        "path": str(settings.artifact_root),
        "ok": True,
    }
    Path(settings.artifact_root).mkdir(parents=True, exist_ok=True)

    comfy = ComfyUIClient(settings.comfyui_base_url, settings.comfyui_timeout_s)
    comfy_ok = await comfy.health()
    report["checks"]["comfyui"] = {
        "url": settings.comfyui_base_url,
        "reachable": comfy_ok,
        "required_for_live": True,
        "ok": comfy_ok if live or not settings.dry_run else True,
    }

    tts = GPTSoVITSClient(settings.tts_base_url, settings.tts_timeout_s)
    tts_ok = await tts.health()
    report["checks"]["tts"] = {
        "url": settings.tts_base_url,
        "reachable": tts_ok,
        "default_ref_set": bool(settings.tts_default_ref_audio),
        "ok": tts_ok if live else True,
    }

    if settings.queue_backend == "redis":
        try:
            from anime_episode.queue.redis_queue import RedisJobQueue
            from anime_episode.storage.artifact_store import ArtifactStore

            q = RedisJobQueue(ArtifactStore(), settings.redis_url)
            depth = await q.depth()
            await q.close()
            report["checks"]["redis"] = {"ok": True, "depth": depth, "url": settings.redis_url}
        except Exception as exc:  # noqa: BLE001
            report["checks"]["redis"] = {"ok": not live, "error": str(exc)}
    else:
        report["checks"]["redis"] = {"ok": True, "skipped": "memory backend"}

    report["ok"] = all(
        c.get("ok", True) for c in report["checks"].values() if isinstance(c, dict)
    )
    report["next_steps"] = []
    if not report["checks"]["workflows"]["ok"]:
        report["next_steps"].append("Fix workflow marker titles (see workflows/README.md)")
    if live and not comfy_ok:
        report["next_steps"].append("Start ComfyUI and install weights from MODELS.md")
    if live and not tts_ok:
        report["next_steps"].append("Start GPT-SoVITS api_v2 on ANIME_TTS_BASE_URL")
    if live and settings.dry_run:
        report["next_steps"].append("Set ANIME_DRY_RUN=false for real generation")
    if not report["next_steps"]:
        report["next_steps"].append(
            "Create episode + PATCH character voices, then kickoff sync or async"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Anime episode pipeline smoke check")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Fail if ComfyUI/TTS are unreachable (GPU machine preflight)",
    )
    args = parser.parse_args(argv)
    report = asyncio.run(run_checks(live=args.live))
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
