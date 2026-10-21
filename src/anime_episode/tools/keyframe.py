"""Keyframe tool — anime still per shot via ComfyUI / Animagine."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from anime_episode.config import get_settings
from anime_episode.domain.models import Episode, Shot
from anime_episode.tools.base import Tool, ToolContext
from anime_episode.tools.comfyui_client import ComfyUIClient
from anime_episode.tools.workflows import (
    first_image,
    load_workflow,
    prepare_keyframe_workflow,
)


class KeyframeInput(BaseModel):
    episode: Episode
    shot: Shot


class KeyframeOutput(BaseModel):
    path: str
    prompt_used: str
    workflow: str | None = None
    prompt_id_meta: str | None = None


class KeyframeTool(Tool[KeyframeInput, KeyframeOutput]):
    name = "keyframe"
    description = "Generate an identity-aware anime keyframe for a shot."

    def __init__(self, comfy: ComfyUIClient | None = None) -> None:
        settings = get_settings()
        self.comfy = comfy or ComfyUIClient(settings.comfyui_base_url, settings.comfyui_timeout_s)

    async def run(self, inp: KeyframeInput, ctx: ToolContext) -> KeyframeOutput:
        settings = get_settings()
        shot = inp.shot
        chars = [c for c in inp.episode.characters if c.id in shot.character_ids]
        char_bits = ", ".join(f"{c.name}: {c.appearance}" for c in chars)
        prompt = f"{shot.visual_prompt}, {char_bits}, {inp.episode.style}".strip(", ")
        negative = shot.negative_prompt or settings.default_negative_prompt

        out_dir = Path(ctx.artifact_dir) / "keyframes"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{shot.id}.png"

        if ctx.dry_run or settings.dry_run:
            out_path.write_bytes(_placeholder_png())
            return KeyframeOutput(path=str(out_path), prompt_used=prompt, workflow="dry_run")

        template = load_workflow(settings.keyframe_workflow)
        workflow = prepare_keyframe_workflow(
            template,
            positive=prompt,
            negative=negative,
            ckpt_name=settings.keyframe_ckpt,
            width=settings.keyframe_width,
            height=settings.keyframe_height,
        )
        history = await self.comfy.run_workflow(workflow, client_id=ctx.episode_id)
        image = first_image(history)
        data = await self.comfy.download_file(
            image["filename"],
            subfolder=image.get("subfolder", ""),
            folder_type=image.get("type", "output"),
        )
        out_path.write_bytes(data)
        return KeyframeOutput(
            path=str(out_path),
            prompt_used=prompt,
            workflow=settings.keyframe_workflow,
        )


def _placeholder_png() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
        b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
