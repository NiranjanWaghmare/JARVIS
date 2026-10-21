"""Keyframe tool — anime still per shot via ComfyUI / Animagine."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from anime_episode.config import get_settings
from anime_episode.domain.models import Episode, Shot
from anime_episode.tools.base import Tool, ToolContext
from anime_episode.tools.comfyui_client import ComfyUIClient


class KeyframeInput(BaseModel):
    episode: Episode
    shot: Shot


class KeyframeOutput(BaseModel):
    path: str
    prompt_used: str


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

        out_dir = Path(ctx.artifact_dir) / "keyframes"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{shot.id}.png"

        if ctx.dry_run or settings.dry_run:
            out_path.write_bytes(_placeholder_png())
            return KeyframeOutput(path=str(out_path), prompt_used=prompt)

        # Real path: load workflow JSON, inject prompt / refs, run ComfyUI.
        workflow = _load_workflow("keyframe_animagine.json")
        _inject_text(workflow, prompt)
        prompt_id = await self.comfy.queue_prompt(workflow, client_id=ctx.episode_id)
        history = await self.comfy.wait_for_prompt(prompt_id)
        filename = _first_image_filename(history)
        data = await self.comfy.download_image(filename)
        out_path.write_bytes(data)
        return KeyframeOutput(path=str(out_path), prompt_used=prompt)


def _placeholder_png() -> bytes:
    # Minimal 1x1 PNG.
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
        b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def _load_workflow(name: str) -> dict:
    import json

    path = Path(__file__).resolve().parent.parent / "workflows" / name
    if not path.exists():
        return {"3": {"inputs": {"text": ""}, "class_type": "CLIPTextEncode"}}
    return json.loads(path.read_text())


def _inject_text(workflow: dict, prompt: str) -> None:
    for node in workflow.values():
        if isinstance(node, dict) and node.get("class_type") == "CLIPTextEncode":
            node.setdefault("inputs", {})["text"] = prompt
            return


def _first_image_filename(history: dict) -> str:
    outputs = history.get("outputs", {})
    for node_out in outputs.values():
        images = node_out.get("images") or []
        if images:
            return images[0]["filename"]
    raise RuntimeError("ComfyUI history contained no images")
