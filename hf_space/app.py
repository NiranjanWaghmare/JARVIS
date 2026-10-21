"""Hugging Face Space — Anime Episode Generator (dry-run demo)."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Package is vendored next to app.py as ./anime_episode
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("ANIME_DRY_RUN", "true")
os.environ.setdefault("ANIME_QUEUE_BACKEND", "memory")

import gradio as gr

from anime_episode.config import get_settings
from anime_episode.domain.models import Episode
from anime_episode.orchestrator.runner import PipelineRunner
from anime_episode.storage.artifact_store import ArtifactStore
from anime_episode.tools.character_assets import voice_readiness
from anime_episode.tools.workflow_validate import validate_all

ARTIFACT_ROOT = Path(tempfile.gettempdir()) / "anime_episode_space"
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["ANIME_ARTIFACT_ROOT"] = str(ARTIFACT_ROOT)
get_settings.cache_clear()


def _fmt_shots(episode: Episode) -> str:
    lines = []
    for shot in sorted(episode.shots, key=lambda s: s.index):
        dial = f' | dialogue: "{shot.dialogue}"' if shot.dialogue else ""
        lines.append(
            f"{shot.index:02d}. [{shot.kind.value}] {shot.duration_s:.0f}s — "
            f"{shot.description}{dial}"
        )
    return "\n".join(lines) if lines else "(no shots)"


def _fmt_characters(episode: Episode) -> str:
    if not episode.characters:
        return "(none yet — appear after plan)"
    lines = []
    for c in episode.characters:
        voice = "voice✓" if c.voice_ref_path else "voice✗"
        lines.append(f"- **{c.name}** ({voice}): {c.appearance}")
    return "\n".join(lines)


def generate_episode(
    title: str,
    premise: str,
    style: str,
    duration_s: int,
) -> tuple[str, str, str, str, str]:
    title = (title or "").strip() or "Untitled Episode"
    premise = (premise or "").strip()
    if not premise:
        raise gr.Error("Please enter a rough story / premise.")
    style = (style or "").strip() or "anime, clean linework, cinematic lighting"
    duration_s = int(duration_s or 120)

    get_settings.cache_clear()
    store = ArtifactStore(root=ARTIFACT_ROOT)
    runner = PipelineRunner(store=store)
    episode = Episode(
        title=title,
        premise=premise,
        style=style,
        target_duration_s=max(30, min(600, duration_s)),
    )
    store.save_episode(episode)
    result = __import__("asyncio").run(runner.kickoff_local(episode.id))

    summary = (
        f"**{result.title}** (`{result.id}`)\n\n"
        f"Status: **{result.status.value}**  \n"
        f"Scenes: {len(result.scenes)} · Shots: {len(result.shots)}  \n"
        f"Final artifact: `{result.final_video_path}`  \n"
        f"Mode: dry-run (CPU demo — no GPU video)"
    )
    scenes = "\n".join(
        f"{s.index + 1}. **{s.title}** @ {s.location} — {s.summary}"
        for s in result.scenes
    )
    voice = voice_readiness(result)
    voice_md = (
        f"Dialogue shots: {voice['dialogue_shot_count']}  \n"
        f"Voices ready: {', '.join(voice['characters_with_voice']) or '—'}  \n"
        f"Missing: {', '.join(voice['characters_missing_voice']) or '—'}"
    )
    meta = {
        "episode_id": result.id,
        "status": result.status.value,
        "shot_count": len(result.shots),
        "final_video_path": result.final_video_path,
        "dry_run": True,
    }
    return summary, _fmt_characters(result), scenes, _fmt_shots(result), json.dumps(meta, indent=2)


def preflight() -> str:
    get_settings.cache_clear()
    settings = get_settings()
    checks = validate_all()
    lines = [
        f"dry_run={settings.dry_run}",
        f"artifact_root={settings.artifact_root}",
        "",
        "Workflows:",
    ]
    for c in checks:
        mark = "✓" if c.ok else "✗"
        extra = f" missing={c.missing}" if c.missing else ""
        lines.append(f"  {mark} {c.name}{extra}")
    lines.append("")
    lines.append(
        "Full GPU render: clone GitHub repo, install ComfyUI weights "
        "(MODELS.md), set ANIME_DRY_RUN=false."
    )
    return "\n".join(lines)


def clear_artifacts() -> str:
    if ARTIFACT_ROOT.exists():
        shutil.rmtree(ARTIFACT_ROOT, ignore_errors=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    return f"Cleared {ARTIFACT_ROOT}"


with gr.Blocks(title="Anime Episode Generator") as demo:
    gr.Markdown(
        """
# Anime Episode Generator
Open-source **story → episode** pipeline demo (plan + dry-run assemble).

Code on **GitHub** · Weights on **Hugging Face** · This Space is **CPU dry-run**.
"""
    )
    with gr.Row():
        with gr.Column(scale=1):
            title_in = gr.Textbox(label="Title", value="Rooftop Signal")
            premise_in = gr.Textbox(
                label="Rough story / premise",
                lines=5,
                value=(
                    "Aiko finds a radio on a rainy rooftop that speaks from tomorrow. "
                    "With Ren, she races through neon alleys to stop a glitch that "
                    "could erase their city from the timeline."
                ),
            )
            style_in = gr.Textbox(
                label="Visual style",
                value="anime, clean linework, cinematic lighting, dusk neon",
            )
            duration_in = gr.Slider(
                60,
                600,
                value=120,
                step=30,
                label="Target duration (seconds)",
            )
            run_btn = gr.Button("Generate episode plan", variant="primary")
            with gr.Row():
                preflight_btn = gr.Button("Preflight")
                clear_btn = gr.Button("Clear cache")
        with gr.Column(scale=1):
            summary_out = gr.Markdown(label="Summary")
            chars_out = gr.Markdown(label="Characters")
            scenes_out = gr.Markdown(label="Scenes")
            shots_out = gr.Textbox(label="Shots", lines=16)
            meta_out = gr.Code(label="Meta JSON", language="json")
            util_out = gr.Textbox(label="Utilities", lines=8)

    run_btn.click(
        generate_episode,
        inputs=[title_in, premise_in, style_in, duration_in],
        outputs=[summary_out, chars_out, scenes_out, shots_out, meta_out],
    )
    preflight_btn.click(preflight, outputs=util_out)
    clear_btn.click(clear_artifacts, outputs=util_out)

    gr.Markdown(
        """
### Links
- GitHub package: [src/anime_episode](https://github.com/NiranjanWaghmare/JARVIS/tree/cursor/anime-episode-architecture-aa56/src/anime_episode)
- Models guide: [MODELS.md](https://github.com/NiranjanWaghmare/JARVIS/blob/cursor/anime-episode-architecture-aa56/src/anime_episode/MODELS.md)
"""
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch()
