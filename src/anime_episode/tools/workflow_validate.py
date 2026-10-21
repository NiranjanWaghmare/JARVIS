"""Validate bundled ComfyUI workflows have required injection markers."""

from __future__ import annotations

from dataclasses import dataclass

from anime_episode.tools.workflows import (
    MARKER_LOAD_AUDIO,
    MARKER_LOAD_IMAGE,
    MARKER_LOAD_VIDEO,
    MARKER_POSITIVE,
    MARKER_UNET,
    MARKER_UNET_HIGH,
    MARKER_UNET_LOW,
    find_nodes_by_title,
    load_workflow,
)

REQUIRED: dict[str, list[str]] = {
    "keyframe_animagine.json": [MARKER_POSITIVE, "NEGATIVE_PROMPT", "CHECKPOINT", "EMPTY_LATENT"],
    "motion_wan_i2v.json": [
        MARKER_POSITIVE,
        "NEGATIVE_PROMPT",
        MARKER_LOAD_IMAGE,
        MARKER_UNET_HIGH,
        MARKER_UNET_LOW,
        "VIDEO_LATENT",
    ],
    "animate_wan.json": [
        MARKER_POSITIVE,
        MARKER_LOAD_IMAGE,
        MARKER_LOAD_VIDEO,
        MARKER_UNET,
        "VIDEO_LATENT",
    ],
    "dialogue_s2v.json": [
        MARKER_POSITIVE,
        MARKER_LOAD_IMAGE,
        MARKER_LOAD_AUDIO,
        MARKER_UNET,
        "VIDEO_LATENT",
    ],
}


@dataclass
class WorkflowCheck:
    name: str
    ok: bool
    missing: list[str]
    error: str | None = None


def validate_all() -> list[WorkflowCheck]:
    results: list[WorkflowCheck] = []
    for name, markers in REQUIRED.items():
        try:
            wf = load_workflow(name)
            missing = [m for m in markers if not find_nodes_by_title(wf, m)]
            results.append(WorkflowCheck(name=name, ok=not missing, missing=missing))
        except Exception as exc:  # noqa: BLE001
            results.append(WorkflowCheck(name=name, ok=False, missing=list(markers), error=str(exc)))
    return results
