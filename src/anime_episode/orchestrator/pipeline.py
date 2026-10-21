"""Shot stage graph — which tools run, in what order, with which skip rules."""

from __future__ import annotations

from dataclasses import dataclass

from anime_episode.domain.enums import ShotKind, ShotStage


@dataclass(frozen=True)
class StageEdge:
    stage: ShotStage
    tool: str
    optional: bool = False


# Ordered stages for a single shot. Dialogue shots include TTS + S2V;
# action shots skip those and keep the animate/motion result as final.
SHOT_STAGES: list[StageEdge] = [
    StageEdge(ShotStage.KEYFRAME, "keyframe"),
    StageEdge(ShotStage.MOTION, "motion"),
    StageEdge(ShotStage.ANIMATE, "animate", optional=True),
    StageEdge(ShotStage.TTS, "tts", optional=True),
    StageEdge(ShotStage.DIALOGUE_VIDEO, "dialogue_video", optional=True),
    StageEdge(ShotStage.POST, "post", optional=True),
]


def stages_for_shot(kind: ShotKind, needs_animate: bool) -> list[StageEdge]:
    stages: list[StageEdge] = []
    for edge in SHOT_STAGES:
        if edge.stage == ShotStage.ANIMATE and not needs_animate:
            continue
        if edge.stage in {ShotStage.TTS, ShotStage.DIALOGUE_VIDEO} and kind != ShotKind.DIALOGUE:
            continue
        if edge.stage == ShotStage.POST:
            # Post is reserved for Real-ESRGAN/RIFE; optional until wired.
            continue
        stages.append(edge)
    return stages


# Episode-level DAG (jobs), not per-frame:
#   plan_episode → build_characters → [render_shot × N parallel] → assemble_episode
EPISODE_JOB_ORDER = [
    "plan_episode",
    "build_characters",
    "render_shot",
    "assemble_episode",
]
