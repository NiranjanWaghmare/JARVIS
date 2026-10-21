"""Story planner tool — rough premise → scenes + shots."""

from __future__ import annotations

import json
import math
from typing import Any

from pydantic import BaseModel, Field

from anime_episode.config import get_settings
from anime_episode.domain.enums import ShotKind
from anime_episode.domain.models import CameraNotes, CharacterRef, Episode, Scene, Shot
from anime_episode.tools.base import Tool, ToolContext


class PlanInput(BaseModel):
    episode: Episode


class PlanOutput(BaseModel):
    episode: Episode
    raw_plan: dict[str, Any] = Field(default_factory=dict)


class StoryPlannerTool(Tool[PlanInput, PlanOutput]):
    name = "story_planner"
    description = "Decompose a rough story into scenes, characters, and timed shots."

    async def run(self, inp: PlanInput, ctx: ToolContext) -> PlanOutput:
        settings = get_settings()
        episode = inp.episode.model_copy(deep=True)

        if ctx.dry_run or settings.dry_run:
            plan = _stub_plan(episode)
        else:
            plan = await _llm_plan(episode)

        episode.characters = [
            CharacterRef(name=c["name"], appearance=c["appearance"])
            for c in plan["characters"]
        ]
        name_to_id = {c.name: c.id for c in episode.characters}

        scenes: list[Scene] = []
        shots: list[Shot] = []
        for s_idx, scene_data in enumerate(plan["scenes"]):
            scene = Scene(
                index=s_idx,
                title=scene_data["title"],
                location=scene_data["location"],
                summary=scene_data["summary"],
            )
            for sh_idx, shot_data in enumerate(scene_data["shots"]):
                kind = ShotKind(shot_data.get("kind", "action"))
                speaker = shot_data.get("speaker")
                shot = Shot(
                    index=len(shots),
                    kind=kind,
                    duration_s=float(shot_data.get("duration_s", settings.default_shot_duration_s)),
                    scene_id=scene.id,
                    description=shot_data["description"],
                    visual_prompt=shot_data["visual_prompt"],
                    dialogue=shot_data.get("dialogue"),
                    speaker_character_id=name_to_id.get(speaker) if speaker else None,
                    character_ids=[
                        name_to_id[n]
                        for n in shot_data.get("characters", [])
                        if n in name_to_id
                    ],
                    camera=CameraNotes(**shot_data.get("camera", {})),
                    needs_animate=bool(shot_data.get("needs_animate", True)),
                )
                scene.shot_ids.append(shot.id)
                shots.append(shot)
            scenes.append(scene)

        episode.scenes = scenes
        episode.shots = shots
        return PlanOutput(episode=episode, raw_plan=plan)


def _stub_plan(episode: Episode) -> dict[str, Any]:
    """Deterministic plan for dry-run / architecture smoke tests."""
    settings = get_settings()
    n_shots = max(4, min(24, math.ceil(episode.target_duration_s / settings.default_shot_duration_s / 10)))
    # Keep dry-run small: ~4–24 shots representing the full episode structure.
    characters = [
        {
            "name": "Aiko",
            "appearance": "teenage girl, short black hair, red hoodie, determined eyes",
        },
        {
            "name": "Ren",
            "appearance": "teenage boy, silver hair, blue jacket, calm expression",
        },
    ]
    scenes = []
    remaining = n_shots
    scene_templates = [
        ("Opening", "city rooftop at dusk", "Introduce world and lead character"),
        ("Conflict", "neon alley", "Obstacle appears; tension rises"),
        ("Climax", "abandoned train station", "Confrontation and choice"),
        ("Resolution", "quiet park at night", "Emotional landing and hook"),
    ]
    shot_i = 0
    for title, location, summary in scene_templates:
        count = max(1, remaining // (5 - len(scenes))) if remaining else 1
        count = min(count, remaining) if remaining else 1
        scene_shots = []
        for j in range(count):
            kind = "dialogue" if j % 3 == 1 else "action"
            if j == 0:
                kind = "establishing"
            speaker = "Aiko" if kind == "dialogue" else None
            scene_shots.append(
                {
                    "kind": kind,
                    "duration_s": settings.default_shot_duration_s,
                    "description": f"{title} beat {j + 1} for: {episode.premise[:120]}",
                    "visual_prompt": (
                        f"{episode.style}, {location}, cinematic anime still, "
                        f"beat {shot_i + 1}, {summary}"
                    ),
                    "dialogue": "We keep moving." if kind == "dialogue" else None,
                    "speaker": speaker,
                    "characters": ["Aiko", "Ren"] if kind != "establishing" else ["Aiko"],
                    "camera": {
                        "framing": "wide" if kind == "establishing" else "medium",
                        "movement": "slow_pan" if kind == "action" else "static",
                        "angle": "eye_level",
                    },
                    "needs_animate": kind != "establishing",
                }
            )
            shot_i += 1
        scenes.append(
            {
                "title": title,
                "location": location,
                "summary": summary,
                "shots": scene_shots,
            }
        )
        remaining -= count
        if remaining <= 0:
            break
    return {"characters": characters, "scenes": scenes}


async def _llm_plan(episode: Episode) -> dict[str, Any]:
    """LLM-backed planner. Requires OPENAI-compatible credentials when dry_run=False."""
    settings = get_settings()
    try:
        from litellm import acompletion
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("litellm is required for non-dry-run planning") from exc

    target_shots = max(
        8,
        math.ceil(episode.target_duration_s / settings.default_shot_duration_s),
    )
    system = (
        "You are an anime episode director. Return ONLY valid JSON with keys "
        "`characters` and `scenes`. Each scene has title, location, summary, shots. "
        "Each shot has kind (action|dialogue|establishing|transition), duration_s, "
        "description, visual_prompt, optional dialogue/speaker, characters[], "
        "camera{framing,movement,angle}, needs_animate."
    )
    user = (
        f"Title: {episode.title}\n"
        f"Premise: {episode.premise}\n"
        f"Style: {episode.style}\n"
        f"Target duration seconds: {episode.target_duration_s}\n"
        f"Target shot count ~{target_shots} at ~{settings.default_shot_duration_s}s each.\n"
        "Cover a full episode arc: opening, conflict, climax, resolution."
    )
    resp = await acompletion(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        api_base=settings.llm_base_url,
        api_key=settings.openai_api_key,
    )
    content = resp.choices[0].message.content
    return json.loads(content)
