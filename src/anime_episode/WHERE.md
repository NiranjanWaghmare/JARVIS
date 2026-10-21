# Where is the code vs the models?

| What | Where | Link |
|---|---|---|
| **Application code** (this pipeline) | **GitHub** — `NiranjanWaghmare/JARVIS` | Branch `cursor/anime-episode-architecture-aa56`, PR #1 |
| **Package path** | Repo: `src/anime_episode/` | Not published to PyPI |
| **Demo UI (HF Space)** | **Hugging Face Spaces** (Gradio dry-run) | Created via `scripts/deploy_hf_space.py` from `hf_space/` |
| **Model weights** | **Hugging Face Hub** (download into ComfyUI) | See [MODELS.md](./MODELS.md) |
| **Artifacts** (episodes, clips) | Local disk `artifacts/anime_episodes/` | gitignored |

**Code source of truth = GitHub.** Hugging Face Spaces hosts a Gradio demo that vendors
this package at deploy time. HF Hub is also where open weights (Animagine, Wan2.2, …) live.
