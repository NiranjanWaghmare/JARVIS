"""ComfyUI API-format workflow load + marker-based injection."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

WORKFLOWS_DIR = Path(__file__).resolve().parent.parent / "workflows"

# Nodes are targeted by `_meta.title` so users can swap exported ComfyUI graphs
# without changing Python — keep these titles on the relevant nodes.
MARKER_POSITIVE = "POSITIVE_PROMPT"
MARKER_NEGATIVE = "NEGATIVE_PROMPT"
MARKER_LOAD_IMAGE = "LOAD_IMAGE"
MARKER_LOAD_VIDEO = "LOAD_VIDEO"
MARKER_LOAD_AUDIO = "LOAD_AUDIO"
MARKER_CHECKPOINT = "CHECKPOINT"
MARKER_UNET = "UNET"
MARKER_UNET_HIGH = "UNET_HIGH"
MARKER_UNET_LOW = "UNET_LOW"
MARKER_CLIP = "CLIP_LOADER"
MARKER_VAE = "VAE_LOADER"
MARKER_EMPTY_LATENT = "EMPTY_LATENT"
MARKER_VIDEO_LATENT = "VIDEO_LATENT"
MARKER_SAMPLER = "SAMPLER"
MARKER_SAVE_IMAGE = "SAVE_IMAGE"
MARKER_SAVE_VIDEO = "SAVE_VIDEO"


class WorkflowError(RuntimeError):
    pass


def load_workflow(name: str, *, workflows_dir: Path | None = None) -> dict[str, Any]:
    root = workflows_dir or WORKFLOWS_DIR
    path = root / name
    if not path.exists():
        raise WorkflowError(f"workflow not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not data:
        raise WorkflowError(f"workflow must be a non-empty API-format object: {path}")
    return data


def find_nodes_by_title(workflow: dict[str, Any], title: str) -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        meta = node.get("_meta") or {}
        if meta.get("title") == title:
            found.append((node_id, node))
    return found


def require_node(workflow: dict[str, Any], title: str) -> tuple[str, dict[str, Any]]:
    matches = find_nodes_by_title(workflow, title)
    if not matches:
        raise WorkflowError(f"missing node with _meta.title={title!r}")
    return matches[0]


def set_input(node: dict[str, Any], key: str, value: Any) -> None:
    node.setdefault("inputs", {})[key] = value


def inject_text_prompts(
    workflow: dict[str, Any],
    *,
    positive: str,
    negative: str | None = None,
) -> dict[str, Any]:
    wf = copy.deepcopy(workflow)
    _, pos = require_node(wf, MARKER_POSITIVE)
    set_input(pos, "text", positive)
    if negative is not None:
        matches = find_nodes_by_title(wf, MARKER_NEGATIVE)
        if matches:
            set_input(matches[0][1], "text", negative)
    return wf


def inject_load_image(workflow: dict[str, Any], filename: str) -> dict[str, Any]:
    wf = copy.deepcopy(workflow)
    _, node = require_node(wf, MARKER_LOAD_IMAGE)
    set_input(node, "image", filename)
    return wf


def inject_load_video(workflow: dict[str, Any], filename: str) -> dict[str, Any]:
    wf = copy.deepcopy(workflow)
    _, node = require_node(wf, MARKER_LOAD_VIDEO)
    # Native LoadVideo uses `file` or `video` depending on ComfyUI version.
    inputs = node.setdefault("inputs", {})
    if "file" in inputs:
        inputs["file"] = filename
    else:
        inputs["video"] = filename
    return wf


def inject_load_audio(workflow: dict[str, Any], filename: str) -> dict[str, Any]:
    wf = copy.deepcopy(workflow)
    _, node = require_node(wf, MARKER_LOAD_AUDIO)
    set_input(node, "audio", filename)
    return wf


def inject_checkpoint(workflow: dict[str, Any], ckpt_name: str) -> dict[str, Any]:
    wf = copy.deepcopy(workflow)
    matches = find_nodes_by_title(wf, MARKER_CHECKPOINT)
    if matches:
        set_input(matches[0][1], "ckpt_name", ckpt_name)
    return wf


def inject_unet(workflow: dict[str, Any], unet_name: str) -> dict[str, Any]:
    wf = copy.deepcopy(workflow)
    matches = find_nodes_by_title(wf, MARKER_UNET)
    if matches:
        set_input(matches[0][1], "unet_name", unet_name)
    return wf


def inject_diffusion_pair(
    workflow: dict[str, Any],
    *,
    high_noise: str | None = None,
    low_noise: str | None = None,
) -> dict[str, Any]:
    wf = copy.deepcopy(workflow)
    if high_noise:
        matches = find_nodes_by_title(wf, MARKER_UNET_HIGH)
        if matches:
            set_input(matches[0][1], "unet_name", high_noise)
    if low_noise:
        matches = find_nodes_by_title(wf, MARKER_UNET_LOW)
        if matches:
            set_input(matches[0][1], "unet_name", low_noise)
    return wf


def inject_latent_size(
    workflow: dict[str, Any],
    *,
    width: int | None = None,
    height: int | None = None,
    length: int | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    wf = copy.deepcopy(workflow)
    for title in (MARKER_EMPTY_LATENT, MARKER_VIDEO_LATENT):
        for _, node in find_nodes_by_title(wf, title):
            if width is not None:
                set_input(node, "width", width)
            if height is not None:
                set_input(node, "height", height)
            if length is not None and "length" in node.get("inputs", {}):
                set_input(node, "length", length)
            if batch_size is not None and "batch_size" in node.get("inputs", {}):
                set_input(node, "batch_size", batch_size)
    return wf


def inject_seed(workflow: dict[str, Any], seed: int | None = None) -> dict[str, Any]:
    wf = copy.deepcopy(workflow)
    value = random.randint(0, 2**63 - 1) if seed is None else seed
    for node in wf.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        if "seed" in inputs and not isinstance(inputs["seed"], list):
            inputs["seed"] = value
        if "noise_seed" in inputs and not isinstance(inputs["noise_seed"], list):
            inputs["noise_seed"] = value
    return wf


def prepare_keyframe_workflow(
    workflow: dict[str, Any],
    *,
    positive: str,
    negative: str,
    ckpt_name: str,
    width: int,
    height: int,
    seed: int | None = None,
) -> dict[str, Any]:
    wf = inject_text_prompts(workflow, positive=positive, negative=negative)
    wf = inject_checkpoint(wf, ckpt_name)
    wf = inject_latent_size(wf, width=width, height=height, batch_size=1)
    return inject_seed(wf, seed)


def prepare_motion_workflow(
    workflow: dict[str, Any],
    *,
    positive: str,
    negative: str,
    image_filename: str,
    width: int,
    height: int,
    length: int,
    high_noise: str | None = None,
    low_noise: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    wf = inject_text_prompts(workflow, positive=positive, negative=negative)
    wf = inject_load_image(wf, image_filename)
    wf = inject_diffusion_pair(wf, high_noise=high_noise, low_noise=low_noise)
    wf = inject_latent_size(wf, width=width, height=height, length=length)
    return inject_seed(wf, seed)


def prepare_animate_workflow(
    workflow: dict[str, Any],
    *,
    positive: str,
    negative: str,
    image_filename: str,
    video_filename: str,
    width: int,
    height: int,
    length: int,
    unet_name: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Wan2.2-Animate Move mode: reference image + driving video → identity-locked clip."""
    wf = inject_text_prompts(workflow, positive=positive, negative=negative)
    wf = inject_load_image(wf, image_filename)
    wf = inject_load_video(wf, video_filename)
    if unet_name:
        wf = inject_unet(wf, unet_name)
    wf = inject_latent_size(wf, width=width, height=height, length=length)
    return inject_seed(wf, seed)


def prepare_s2v_workflow(
    workflow: dict[str, Any],
    *,
    positive: str,
    negative: str,
    image_filename: str,
    audio_filename: str,
    width: int,
    height: int,
    length: int | None = None,
    unet_name: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Wan2.2-S2V: reference image + dialogue audio → talking clip."""
    wf = inject_text_prompts(workflow, positive=positive, negative=negative)
    wf = inject_load_image(wf, image_filename)
    wf = inject_load_audio(wf, audio_filename)
    if unet_name:
        wf = inject_unet(wf, unet_name)
    if length is not None:
        wf = inject_latent_size(wf, width=width, height=height, length=length)
    else:
        wf = inject_latent_size(wf, width=width, height=height)
    return inject_seed(wf, seed)


def iter_history_files(history: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten ComfyUI history outputs into [{filename, subfolder, type, kind}]."""
    files: list[dict[str, Any]] = []
    outputs = history.get("outputs") or {}
    for node_out in outputs.values():
        if not isinstance(node_out, dict):
            continue
        for kind in ("images", "gifs", "videos"):
            for item in node_out.get(kind) or []:
                files.append(
                    {
                        "filename": item["filename"],
                        "subfolder": item.get("subfolder", ""),
                        "type": item.get("type", "output"),
                        "kind": kind,
                    }
                )
    return files


def first_image(history: dict[str, Any]) -> dict[str, Any]:
    for item in iter_history_files(history):
        if item["kind"] == "images":
            return item
    raise WorkflowError("ComfyUI history contained no images")


def first_video(history: dict[str, Any]) -> dict[str, Any]:
    for item in iter_history_files(history):
        if item["kind"] in {"videos", "gifs"}:
            return item
    # Some SaveVideo nodes still report under images with .mp4
    for item in iter_history_files(history):
        name = item["filename"].lower()
        if name.endswith((".mp4", ".webm", ".gif")):
            return item
    raise WorkflowError("ComfyUI history contained no video outputs")
