"""Pipeline runner — executes episode-level and shot-level stages via tools."""

from __future__ import annotations

import logging
from pathlib import Path

from anime_episode.config import get_settings
from anime_episode.domain.enums import ArtifactKind, EpisodeStatus, ShotStage
from anime_episode.domain.models import Artifact, Episode, Shot
from anime_episode.orchestrator.pipeline import stages_for_shot
from anime_episode.storage.artifact_store import ArtifactStore
from anime_episode.tools.animate import AnimateInput, AnimateTool
from anime_episode.tools.assemble import AssembleInput, AssembleTool
from anime_episode.tools.base import ToolContext
from anime_episode.tools.character import CharacterBuildInput, CharacterBuildTool
from anime_episode.tools.dialogue_video import DialogueVideoInput, DialogueVideoTool
from anime_episode.tools.keyframe import KeyframeInput, KeyframeTool
from anime_episode.tools.motion import MotionInput, MotionTool
from anime_episode.tools.planner import PlanInput, StoryPlannerTool
from anime_episode.tools.tts import TTSInput, TTSTool

logger = logging.getLogger(__name__)


class PipelineRunner:
    def __init__(
        self,
        store: ArtifactStore | None = None,
        planner: StoryPlannerTool | None = None,
        characters: CharacterBuildTool | None = None,
        keyframe: KeyframeTool | None = None,
        motion: MotionTool | None = None,
        animate: AnimateTool | None = None,
        tts: TTSTool | None = None,
        dialogue: DialogueVideoTool | None = None,
        assemble: AssembleTool | None = None,
    ) -> None:
        self.store = store or ArtifactStore()
        self.planner = planner or StoryPlannerTool()
        self.characters = characters or CharacterBuildTool()
        self.keyframe = keyframe or KeyframeTool()
        self.motion = motion or MotionTool()
        self.animate = animate or AnimateTool()
        self.tts = tts or TTSTool()
        self.dialogue = dialogue or DialogueVideoTool()
        self.assemble_tool = assemble or AssembleTool()

    def _ctx(self, episode_id: str) -> ToolContext:
        settings = get_settings()
        artifact_dir = str(self.store.episode_dir(episode_id))
        return ToolContext(
            episode_id=episode_id,
            dry_run=settings.dry_run,
            artifact_dir=artifact_dir,
        )

    async def plan(self, episode_id: str) -> Episode:
        episode = self.store.load_episode(episode_id)
        episode.status = EpisodeStatus.PLANNING
        self.store.save_episode(episode)

        out = await self.planner.run(PlanInput(episode=episode), self._ctx(episode_id))
        episode = out.episode
        episode.status = EpisodeStatus.DRAFT
        episode.touch()
        self.store.save_episode(episode)
        self.store.dump_meta(episode_id, "plan", out.raw_plan)
        return episode

    async def build_characters(self, episode_id: str) -> Episode:
        episode = self.store.load_episode(episode_id)
        out = await self.characters.run(
            CharacterBuildInput(episode=episode),
            self._ctx(episode_id),
        )
        episode = out.episode
        episode.touch()
        self.store.save_episode(episode)
        return episode

    async def render_shot(self, episode_id: str, shot_id: str) -> Shot:
        episode = self.store.load_episode(episode_id)
        shot = episode.shot_by_id(shot_id).model_copy(deep=True)
        ctx = self._ctx(episode_id)
        keyframe_artifact: Artifact | None = None

        self.store.mutate_episode(
            episode_id,
            lambda ep: setattr(ep, "status", EpisodeStatus.RENDERING),
        )

        try:
            for edge in stages_for_shot(shot.kind, shot.needs_animate):
                shot.stage = edge.stage
                if edge.stage == ShotStage.KEYFRAME:
                    kf = await self.keyframe.run(
                        KeyframeInput(episode=episode, shot=shot),
                        ctx,
                    )
                    shot.artifacts["keyframe"] = kf.path
                    keyframe_artifact = Artifact(
                        kind=ArtifactKind.KEYFRAME, path=kf.path, shot_id=shot.id
                    )
                elif edge.stage == ShotStage.MOTION:
                    motion = await self.motion.run(
                        MotionInput(
                            episode=episode,
                            shot=shot,
                            keyframe_path=shot.artifacts["keyframe"],
                        ),
                        ctx,
                    )
                    shot.artifacts["motion_clip"] = motion.path
                elif edge.stage == ShotStage.ANIMATE:
                    anim = await self.animate.run(
                        AnimateInput(
                            episode=episode,
                            shot=shot,
                            keyframe_path=shot.artifacts["keyframe"],
                            driving_clip_path=shot.artifacts["motion_clip"],
                        ),
                        ctx,
                    )
                    shot.artifacts["animated_clip"] = anim.path
                elif edge.stage == ShotStage.TTS:
                    voice = await self.tts.run(TTSInput(episode=episode, shot=shot), ctx)
                    if voice.path:
                        shot.artifacts["voice"] = voice.path
                elif edge.stage == ShotStage.DIALOGUE_VIDEO:
                    fallback = shot.artifacts.get("animated_clip") or shot.artifacts["motion_clip"]
                    dial = await self.dialogue.run(
                        DialogueVideoInput(
                            episode=episode,
                            shot=shot,
                            keyframe_path=shot.artifacts["keyframe"],
                            audio_path=shot.artifacts.get("voice"),
                            fallback_clip_path=fallback,
                        ),
                        ctx,
                    )
                    shot.artifacts["dialogue_clip"] = dial.path

            shot.artifacts["final_shot"] = (
                shot.artifacts.get("dialogue_clip")
                or shot.artifacts.get("animated_clip")
                or shot.artifacts["motion_clip"]
            )
            shot.stage = ShotStage.DONE
            shot.error = None
        except Exception as exc:  # noqa: BLE001
            shot.stage = ShotStage.FAILED
            shot.error = str(exc)
            logger.exception("shot %s failed", shot_id)

            def _fail(ep: Episode) -> None:
                for i, existing in enumerate(ep.shots):
                    if existing.id == shot.id:
                        ep.shots[i] = shot
                        break
                ep.error = f"shot {shot_id}: {exc}"
                ep.status = EpisodeStatus.FAILED

            self.store.mutate_episode(episode_id, _fail)
            raise

        def _ok(ep: Episode) -> None:
            for i, existing in enumerate(ep.shots):
                if existing.id == shot.id:
                    ep.shots[i] = shot
                    break
            if keyframe_artifact is not None:
                ep.artifacts.append(keyframe_artifact)

        self.store.mutate_episode(episode_id, _ok)
        return shot

    async def assemble(self, episode_id: str) -> str:
        episode = self.store.load_episode(episode_id)
        unfinished = [s.id for s in episode.shots if s.stage != ShotStage.DONE]
        if unfinished:
            raise RuntimeError(f"shots not done: {unfinished[:8]}")

        self.store.mutate_episode(
            episode_id,
            lambda ep: setattr(ep, "status", EpisodeStatus.ASSEMBLING),
        )
        episode = self.store.load_episode(episode_id)

        out = await self.assemble_tool.run(
            AssembleInput(episode=episode),
            self._ctx(episode_id),
        )

        def _complete(ep: Episode) -> None:
            ep.final_video_path = out.path
            ep.artifacts.append(Artifact(kind=ArtifactKind.FINAL_EPISODE, path=out.path))
            ep.status = EpisodeStatus.COMPLETED

        self.store.mutate_episode(episode_id, _complete)
        return out.path

    async def kickoff_local(self, episode_id: str) -> Episode:
        """Run the full DAG in-process (useful for dry-run demos)."""
        await self.plan(episode_id)
        await self.build_characters(episode_id)
        episode = self.store.load_episode(episode_id)
        for shot in sorted(episode.shots, key=lambda s: s.index):
            await self.render_shot(episode_id, shot.id)
        await self.assemble(episode_id)
        return self.store.load_episode(episode_id)


def ensure_workspace(episode_id: str, store: ArtifactStore | None = None) -> Path:
    store = store or ArtifactStore()
    return store.episode_dir(episode_id)
