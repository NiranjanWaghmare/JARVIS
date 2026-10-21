---
title: Anime Episode Generator
emoji: 🎬
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
python_version: "3.12"
pinned: false
license: apache-2.0
short_description: Story → anime episode plan (open-source pipeline demo)
tags:
  - anime
  - video
  - comfyui
  - wan
  - storytelling
---

# Anime Episode Generator

Turn a **rough story** into a structured anime episode plan and dry-run render
using the open-source stack (AniSora / Wan2.2 / GPT-SoVITS adapters).

> **This Space runs in dry-run mode by default** (CPU-friendly).  
> Full GPU video generation needs ComfyUI + model weights on a GPU host — see
> [`MODELS.md`](https://github.com/NiranjanWaghmare/JARVIS/blob/cursor/anime-episode-architecture-aa56/src/anime_episode/MODELS.md).

## Source

- **App code (GitHub):** [NiranjanWaghmare/JARVIS](https://github.com/NiranjanWaghmare/JARVIS) → `src/anime_episode/`
- **Weights (Hugging Face):** Wan2.2 / Animagine checkpoints (not bundled in this Space)

## Pipeline

`plan → character sheets → keyframes → motion → animate → TTS → S2V → assemble`
