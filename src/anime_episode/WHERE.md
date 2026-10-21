# Where is the code vs the models?

| What | Where | Link |
|---|---|---|
| **Application code** (this pipeline) | **GitHub** — `NiranjanWaghmare/JARVIS` | Branch `cursor/anime-episode-architecture-aa56`, PR #1 |
| **Package path** | Repo: `src/anime_episode/` | Not published to PyPI |
| **Demo UI (HF Space)** | **Hugging Face Spaces** (ZeroGPU + dry-run plan) | https://huggingface.co/spaces/NiranjanWaghmare/anime-episode-generator |
| **Model weights** | **Hugging Face Hub** (Space pulls Wan TI2V; ComfyUI for full eps) | See [MODELS.md](./MODELS.md) |
| **Artifacts** (episodes, clips) | Local disk `artifacts/anime_episodes/` | gitignored |

**Code source of truth = GitHub.** The HF Space vendors this package and runs **ZeroGPU**
for short Wan2.2 clips (not full ~10 min episodes). HF Hub also hosts ComfyUI weights.
