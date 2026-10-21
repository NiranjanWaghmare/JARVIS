"""Tests for Wan-Animate, GPT-SoVITS, and Wan-S2V live wiring."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from anime_episode.config import get_settings
from anime_episode.domain.enums import ShotKind
from anime_episode.domain.models import CharacterRef, Episode, Shot
from anime_episode.tools.animate import AnimateInput, AnimateTool
from anime_episode.tools.base import ToolContext
from anime_episode.tools.dialogue_video import DialogueVideoInput, DialogueVideoTool
from anime_episode.tools.tts import TTSInput, TTSTool
from anime_episode.tools.workflows import (
    MARKER_LOAD_AUDIO,
    MARKER_LOAD_IMAGE,
    MARKER_LOAD_VIDEO,
    MARKER_POSITIVE,
    MARKER_UNET,
    find_nodes_by_title,
    load_workflow,
    prepare_animate_workflow,
    prepare_s2v_workflow,
)


def test_prepare_animate_and_s2v_workflows() -> None:
    anim = prepare_animate_workflow(
        load_workflow("animate_wan.json"),
        positive="lock identity",
        negative="drift",
        image_filename="ref.png",
        video_filename="drive.mp4",
        width=640,
        height=640,
        length=81,
        unet_name="animate.safetensors",
        seed=1,
    )
    assert find_nodes_by_title(anim, MARKER_LOAD_IMAGE)[0][1]["inputs"]["image"] == "ref.png"
    assert find_nodes_by_title(anim, MARKER_LOAD_VIDEO)[0][1]["inputs"]["video"] == "drive.mp4"
    assert find_nodes_by_title(anim, MARKER_UNET)[0][1]["inputs"]["unet_name"] == "animate.safetensors"

    s2v = prepare_s2v_workflow(
        load_workflow("dialogue_s2v.json"),
        positive="speaking",
        negative="static mouth",
        image_filename="face.png",
        audio_filename="line.wav",
        width=640,
        height=640,
        length=77,
        unet_name="s2v.safetensors",
        seed=2,
    )
    assert find_nodes_by_title(s2v, MARKER_LOAD_AUDIO)[0][1]["inputs"]["audio"] == "line.wav"
    assert find_nodes_by_title(s2v, MARKER_POSITIVE)[0][1]["inputs"]["text"] == "speaking"


class _FakeComfy:
    def __init__(self) -> None:
        self.last_workflow: dict[str, Any] | None = None
        self.uploads: list[str] = []

    async def upload_image(self, path: str | Path, **kwargs: Any) -> str:
        self.uploads.append(f"img:{path}")
        return Path(path).name

    async def upload_video(self, path: str | Path, **kwargs: Any) -> str:
        self.uploads.append(f"vid:{path}")
        return Path(path).name

    async def upload_audio(self, path: str | Path, **kwargs: Any) -> str:
        self.uploads.append(f"aud:{path}")
        return Path(path).name

    async def run_workflow(self, workflow: dict[str, Any], *, client_id: str) -> dict[str, Any]:
        self.last_workflow = workflow
        return {
            "outputs": {
                "out": {"videos": [{"filename": "out.mp4", "subfolder": "", "type": "output"}]}
            }
        }

    async def download_file(self, filename: str, **kwargs: Any) -> bytes:
        return b"\x00\x00\x00\x18ftypmp42"


class _FakeTTS:
    def __init__(self) -> None:
        self.last: dict[str, Any] | None = None

    async def synthesize(self, **kwargs: Any) -> bytes:
        self.last = kwargs
        return b"RIFF....WAVEdatafake"


def test_animate_tts_s2v_live_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ANIME_DRY_RUN", "false")
    monkeypatch.setenv("ANIME_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("ANIME_TTS_DEFAULT_REF_AUDIO", str(tmp_path / "ref.wav"))
    get_settings.cache_clear()

    (tmp_path / "ref.wav").write_bytes(b"RIFF")
    keyframe = tmp_path / "kf.png"
    drive = tmp_path / "drive.mp4"
    keyframe.write_bytes(b"\x89PNG")
    drive.write_bytes(b"mp4")

    episode = Episode(title="t", premise="p")
    speaker = CharacterRef(
        name="Aiko",
        appearance="black hair",
        voice_ref_path=str(tmp_path / "ref.wav"),
        voice_prompt_text="hello",
        voice_lang="en",
    )
    episode.characters = [speaker]
    shot = Shot(
        index=0,
        kind=ShotKind.DIALOGUE,
        scene_id="s1",
        description="Aiko speaks",
        visual_prompt="anime girl talking",
        dialogue="We keep moving.",
        speaker_character_id=speaker.id,
        needs_animate=True,
    )
    ctx = ToolContext(episode_id=episode.id, dry_run=False, artifact_dir=str(tmp_path / episode.id))

    fake_comfy = _FakeComfy()
    anim = asyncio.run(
        AnimateTool(comfy=fake_comfy).run(  # type: ignore[arg-type]
            AnimateInput(
                episode=episode,
                shot=shot,
                keyframe_path=str(keyframe),
                driving_clip_path=str(drive),
            ),
            ctx,
        )
    )
    assert Path(anim.path).exists()
    assert find_nodes_by_title(fake_comfy.last_workflow, MARKER_LOAD_VIDEO)

    fake_tts = _FakeTTS()
    voice = asyncio.run(
        TTSTool(client=fake_tts).run(TTSInput(episode=episode, shot=shot), ctx)  # type: ignore[arg-type]
    )
    assert voice.path and Path(voice.path).exists()
    assert fake_tts.last is not None
    assert fake_tts.last["text"] == "We keep moving."

    dial = asyncio.run(
        DialogueVideoTool(comfy=fake_comfy).run(  # type: ignore[arg-type]
            DialogueVideoInput(
                episode=episode,
                shot=shot,
                keyframe_path=str(keyframe),
                audio_path=voice.path,
                fallback_clip_path=str(drive),
            ),
            ctx,
        )
    )
    assert Path(dial.path).exists()
    assert find_nodes_by_title(fake_comfy.last_workflow, MARKER_LOAD_AUDIO)

    get_settings.cache_clear()
