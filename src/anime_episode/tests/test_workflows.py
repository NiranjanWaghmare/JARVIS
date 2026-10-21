"""Unit tests for ComfyUI workflow injection + mocked keyframe/motion runs."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from anime_episode.config import get_settings
from anime_episode.domain.enums import ShotKind
from anime_episode.domain.models import CameraNotes, Episode, Shot
from anime_episode.tools.base import ToolContext
from anime_episode.tools.keyframe import KeyframeInput, KeyframeTool
from anime_episode.tools.motion import MotionInput, MotionTool, _wan_length
from anime_episode.tools.workflows import (
    MARKER_LOAD_IMAGE,
    MARKER_POSITIVE,
    WorkflowError,
    find_nodes_by_title,
    first_image,
    first_video,
    load_workflow,
    prepare_keyframe_workflow,
    prepare_motion_workflow,
)


def test_load_bundled_workflows() -> None:
    kf = load_workflow("keyframe_animagine.json")
    assert find_nodes_by_title(kf, MARKER_POSITIVE)
    motion = load_workflow("motion_wan_i2v.json")
    assert find_nodes_by_title(motion, MARKER_LOAD_IMAGE)


def test_prepare_keyframe_injects_prompt_and_size() -> None:
    template = load_workflow("keyframe_animagine.json")
    wf = prepare_keyframe_workflow(
        template,
        positive="a girl on a rooftop, anime",
        negative="blurry",
        ckpt_name="custom.safetensors",
        width=1024,
        height=1024,
        seed=42,
    )
    pos = find_nodes_by_title(wf, MARKER_POSITIVE)[0][1]
    assert pos["inputs"]["text"] == "a girl on a rooftop, anime"
    ckpt = find_nodes_by_title(wf, "CHECKPOINT")[0][1]
    assert ckpt["inputs"]["ckpt_name"] == "custom.safetensors"
    latent = find_nodes_by_title(wf, "EMPTY_LATENT")[0][1]
    assert latent["inputs"]["width"] == 1024
    sampler = find_nodes_by_title(wf, "SAMPLER")[0][1]
    assert sampler["inputs"]["seed"] == 42


def test_prepare_motion_injects_image_and_length() -> None:
    template = load_workflow("motion_wan_i2v.json")
    wf = prepare_motion_workflow(
        template,
        positive="slow pan, wind in hair",
        negative="static",
        image_filename="shot_001.png",
        width=640,
        height=640,
        length=81,
        high_noise="high.safetensors",
        low_noise="low.safetensors",
        seed=7,
    )
    img = find_nodes_by_title(wf, MARKER_LOAD_IMAGE)[0][1]
    assert img["inputs"]["image"] == "shot_001.png"
    latent = find_nodes_by_title(wf, "VIDEO_LATENT")[0][1]
    assert latent["inputs"]["length"] == 81
    high = find_nodes_by_title(wf, "UNET_HIGH")[0][1]
    assert high["inputs"]["unet_name"] == "high.safetensors"


def test_wan_length_snaps_to_4n_plus_1() -> None:
    assert _wan_length(5.0, 81) == 81
    assert _wan_length(2.0, 81) == 33  # 32→33


def test_history_helpers() -> None:
    history = {
        "outputs": {
            "7": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]},
            "10": {"videos": [{"filename": "b.mp4", "subfolder": "video", "type": "output"}]},
        }
    }
    assert first_image(history)["filename"] == "a.png"
    assert first_video(history)["filename"] == "b.mp4"
    with pytest.raises(WorkflowError):
        first_video({"outputs": {}})


class _FakeComfy:
    def __init__(self) -> None:
        self.last_workflow: dict[str, Any] | None = None
        self.uploaded: list[str] = []

    async def run_workflow(self, workflow: dict[str, Any], *, client_id: str) -> dict[str, Any]:
        self.last_workflow = workflow
        # Detect keyframe vs motion by presence of LoadImage marker content
        has_load = any(
            isinstance(n, dict) and n.get("class_type") == "LoadImage" for n in workflow.values()
        )
        if has_load:
            return {
                "outputs": {
                    "10": {
                        "videos": [
                            {"filename": "motion.mp4", "subfolder": "", "type": "output"}
                        ]
                    }
                }
            }
        return {
            "outputs": {
                "7": {"images": [{"filename": "kf.png", "subfolder": "", "type": "output"}]}
            }
        }

    async def upload_image(self, path: str | Path, **kwargs: Any) -> str:
        self.uploaded.append(str(path))
        return Path(path).name

    async def download_file(self, filename: str, **kwargs: Any) -> bytes:
        if filename.endswith(".mp4"):
            return b"\x00\x00\x00\x18ftypmp42"
        return b"\x89PNG\r\n\x1a\n"


def test_keyframe_and_motion_live_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ANIME_DRY_RUN", "false")
    monkeypatch.setenv("ANIME_ARTIFACT_ROOT", str(tmp_path))
    get_settings.cache_clear()

    fake = _FakeComfy()
    episode = Episode(title="t", premise="p")
    shot = Shot(
        index=0,
        kind=ShotKind.ACTION,
        scene_id="scene_1",
        description="Aiko turns toward the city lights",
        visual_prompt="anime girl, rooftop, dusk",
        camera=CameraNotes(movement="slow_pan"),
    )
    ctx = ToolContext(episode_id=episode.id, dry_run=False, artifact_dir=str(tmp_path / episode.id))

    kf_tool = KeyframeTool(comfy=fake)  # type: ignore[arg-type]
    kf = asyncio.run(kf_tool.run(KeyframeInput(episode=episode, shot=shot), ctx))
    assert Path(kf.path).exists()
    assert fake.last_workflow is not None
    pos = find_nodes_by_title(fake.last_workflow, MARKER_POSITIVE)[0][1]
    assert "rooftop" in pos["inputs"]["text"]

    keyframe_path = Path(kf.path)
    motion_tool = MotionTool(comfy=fake)  # type: ignore[arg-type]
    motion = asyncio.run(
        motion_tool.run(
            MotionInput(episode=episode, shot=shot, keyframe_path=str(keyframe_path)),
            ctx,
        )
    )
    assert Path(motion.path).read_bytes().startswith(b"\x00\x00")
    assert fake.uploaded
    assert find_nodes_by_title(fake.last_workflow, MARKER_LOAD_IMAGE)[0][1]["inputs"]["image"] == keyframe_path.name

    get_settings.cache_clear()
