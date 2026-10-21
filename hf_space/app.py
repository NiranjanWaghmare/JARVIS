"""Hugging Face Space — Anime Episode Generator (plan + ZeroGPU Wan clips).

IMPORTANT: `spaces` must be imported before torch/diffusers so ZeroGPU can
hijack CUDA for module-level `.to("cuda")` packing.
"""

from __future__ import annotations

import spaces  # noqa: F401 — must be first among CUDA-touching imports

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import gradio as gr
import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video
from PIL import Image

# Package is vendored next to app.py as ./anime_episode
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("ANIME_DRY_RUN", "true")
os.environ.setdefault("ANIME_QUEUE_BACKEND", "memory")

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

# --- Wan2.2 short-clip pipeline (ZeroGPU) ---------------------------------
# Module-level load + .to("cuda") is required for ZeroGPU pack/unpack.
MODEL_ID = os.environ.get("WAN_MODEL_ID", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
DTYPE = torch.bfloat16

print(f"Loading {MODEL_ID} for ZeroGPU…")
_vae = AutoencoderKLWan.from_pretrained(
    MODEL_ID,
    subfolder="vae",
    torch_dtype=torch.float32,
)
pipe = WanPipeline.from_pretrained(
    MODEL_ID,
    vae=_vae,
    torch_dtype=DTYPE,
)
pipe.to("cuda")
print("Wan pipeline ready (CUDA-emulated until @spaces.GPU).")


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
        f"Mode: dry-run plan (full ~10 min episode needs local ComfyUI GPU)"
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
        "voice": voice,
    }
    _ = voice_md  # kept in meta
    return summary, _fmt_characters(result), scenes, _fmt_shots(result), json.dumps(meta, indent=2)


def _gpu_duration(prompt: str, image, width: int, height: int, num_frames: int, steps: int, *args, **kwargs) -> int:
    """Estimate ZeroGPU reservation from frames × steps (fit free-tier caps)."""
    base = 45
    per = max(1, int(num_frames) // 24) * max(1, int(steps) // 5) * 4
    return int(min(180, max(60, base + per)))


@spaces.GPU(duration=_gpu_duration)
def generate_clip(
    prompt: str,
    image: Image.Image | None,
    width: int,
    height: int,
    num_frames: int,
    steps: int,
    guidance: float,
    seed: int,
) -> tuple[str | None, str]:
    """Generate a short anime clip on ZeroGPU (not a full episode)."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Enter a prompt for the clip.")

    width = int(width)
    height = int(height)
    num_frames = int(num_frames)
    steps = int(steps)
    guidance = float(guidance)
    seed = int(seed)

    if seed < 0:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    generator = torch.Generator(device="cuda").manual_seed(seed)

    gen_params: dict = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "guidance_scale": guidance,
        "num_inference_steps": steps,
        "generator": generator,
    }
    if image is not None:
        gen_params["image"] = image.convert("RGB")

    print(
        f"Wan clip: {width}x{height} frames={num_frames} steps={steps} seed={seed}"
    )
    frames = pipe(**gen_params).frames[0]

    out_dir = ARTIFACT_ROOT / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"clip_{seed}.mp4"
    # frames may be list[np.ndarray] or np.ndarray
    export_to_video(frames, str(out_path), fps=24)
    secs = max(1, (num_frames - 1) / 24)
    status = (
        f"OK — ~{secs:.1f}s clip @ {width}x{height}, {num_frames} frames, "
        f"{steps} steps, seed={seed}. "
        f"ZeroGPU quota applies; full episodes need a dedicated GPU host."
    )
    return str(out_path), status


def preflight() -> str:
    get_settings.cache_clear()
    settings = get_settings()
    checks = validate_all()
    lines = [
        f"dry_run={settings.dry_run}",
        f"artifact_root={settings.artifact_root}",
        f"wan_model={MODEL_ID}",
        f"torch_cuda_available={torch.cuda.is_available()}",
        "",
        "Workflows:",
    ]
    for c in checks:
        mark = "✓" if c.ok else "✗"
        extra = f" missing={c.missing}" if c.missing else ""
        lines.append(f"  {mark} {c.name}{extra}")
    lines.append("")
    lines.append(
        "Short clips: ZeroGPU Wan2.2 TI2V (~2–4s). "
        "Full ~10 min episode: local ComfyUI + MODELS.md weights."
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
Open-source **story → episode plan** plus **ZeroGPU Wan short clips**.

Code on **GitHub** · Weights on **Hugging Face Hub** · Hardware: **ZeroGPU**
"""
    )
    with gr.Tabs():
        with gr.Tab("Episode plan (dry-run)"):
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

        with gr.Tab("Short clip (ZeroGPU Wan)"):
            gr.Markdown(
                """
Generate a **few-second** anime-style clip with Wan2.2 TI2V-5B on ZeroGPU.
This is **not** a 10-minute episode — daily GPU quota is limited (~minutes).
Defaults fit free-tier ZeroGPU; raise frames/steps only if you have PRO quota.
"""
            )
            with gr.Row():
                with gr.Column():
                    clip_prompt = gr.Textbox(
                        label="Prompt",
                        lines=3,
                        value=(
                            "anime, cinematic dusk neon alley, rain on pavement, "
                            "girl with short black hair holding a glowing radio, "
                            "clean linework, dynamic camera push-in"
                        ),
                    )
                    clip_image = gr.Image(
                        label="Optional start image (I2V)",
                        type="pil",
                    )
                    with gr.Accordion("Advanced", open=False):
                        with gr.Row():
                            clip_w = gr.Slider(512, 1280, value=832, step=16, label="Width")
                            clip_h = gr.Slider(288, 720, value=480, step=16, label="Height")
                        clip_frames = gr.Slider(
                            25,
                            97,
                            value=49,
                            step=8,
                            label="Frames (~24fps; 49 ≈ 2s)",
                        )
                        clip_steps = gr.Slider(15, 50, value=25, step=1, label="Steps")
                        clip_guidance = gr.Slider(1.0, 10.0, value=5.0, step=0.5, label="Guidance")
                        clip_seed = gr.Number(value=-1, precision=0, label="Seed (−1 = random)")
                    clip_btn = gr.Button("Generate clip on ZeroGPU", variant="primary")
                with gr.Column():
                    clip_video = gr.Video(label="Clip", autoplay=True)
                    clip_status = gr.Textbox(label="Status", lines=3)

            clip_btn.click(
                generate_clip,
                inputs=[
                    clip_prompt,
                    clip_image,
                    clip_w,
                    clip_h,
                    clip_frames,
                    clip_steps,
                    clip_guidance,
                    clip_seed,
                ],
                outputs=[clip_video, clip_status],
            )

    gr.Markdown(
        """
### Links
- GitHub package: [src/anime_episode](https://github.com/NiranjanWaghmare/JARVIS/tree/cursor/anime-episode-architecture-aa56/src/anime_episode)
- Models guide: [MODELS.md](https://github.com/NiranjanWaghmare/JARVIS/blob/cursor/anime-episode-architecture-aa56/src/anime_episode/MODELS.md)
"""
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
