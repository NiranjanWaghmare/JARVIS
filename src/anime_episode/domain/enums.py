"""Shared enums for episode pipeline state."""

from __future__ import annotations

from enum import StrEnum


class EpisodeStatus(StrEnum):
    DRAFT = "draft"
    PLANNING = "planning"
    RENDERING = "rendering"
    ASSEMBLING = "assembling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ShotStage(StrEnum):
    """Ordered processing stages for a single shot."""

    PENDING = "pending"
    KEYFRAME = "keyframe"
    MOTION = "motion"
    ANIMATE = "animate"  # optional identity lock
    TTS = "tts"  # dialogue shots only
    DIALOGUE_VIDEO = "dialogue_video"  # S2V; dialogue shots only
    POST = "post"  # per-shot upscale / interpolate
    DONE = "done"
    FAILED = "failed"


class ShotKind(StrEnum):
    ACTION = "action"
    DIALOGUE = "dialogue"
    ESTABLISHING = "establishing"
    TRANSITION = "transition"


class JobType(StrEnum):
    PLAN_EPISODE = "plan_episode"
    BUILD_CHARACTERS = "build_characters"
    RENDER_SHOT = "render_shot"
    ASSEMBLE_EPISODE = "assemble_episode"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ArtifactKind(StrEnum):
    SCRIPT = "script"
    CHARACTER_SHEET = "character_sheet"
    KEYFRAME = "keyframe"
    MOTION_CLIP = "motion_clip"
    ANIMATED_CLIP = "animated_clip"
    VOICE = "voice"
    DIALOGUE_CLIP = "dialogue_clip"
    FINAL_SHOT = "final_shot"
    FINAL_EPISODE = "final_episode"
    LOG = "log"
