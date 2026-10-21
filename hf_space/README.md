---
title: Anime Episode Generator
emoji: 🎬
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
python_version: "3.12"
hardware: zero-a10g
pinned: false
license: apache-2.0
short_description: Story → anime plan + ZeroGPU Wan short clips
tags:
  - anime
  - video
  - wan
  - zerogpu
  - storytelling
---

# Anime Episode Generator

Turn a **rough story** into a structured anime episode plan, and generate
**short Wan2.2 clips** on Hugging Face **ZeroGPU**.

| Mode | What you get | Hardware |
|------|----------------|----------|
| Episode plan | Scenes / shots / dry-run assemble | CPU (always) |
| Short clip | ~2–4s Wan2.2 TI2V video | ZeroGPU (`@spaces.GPU`) |

> **Full ~10 minute episodes are not practical on Spaces ZeroGPU** (daily
> quota is measured in minutes). For the full ComfyUI shot pipeline, clone
> the GitHub repo and follow
> [`MODELS.md`](https://github.com/NiranjanWaghmare/JARVIS/blob/cursor/anime-episode-architecture-aa56/src/anime_episode/MODELS.md).

## Source

- **App code (GitHub):** [NiranjanWaghmare/JARVIS](https://github.com/NiranjanWaghmare/JARVIS) → `src/anime_episode/`
- **Weights (Hugging Face):** `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (downloaded by the Space at build)

## Pipeline

`plan → character sheets → keyframes → motion → animate → TTS → S2V → assemble`
