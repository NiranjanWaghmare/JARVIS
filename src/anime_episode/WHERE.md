# Where is the code vs the models?

| What | Where | Link |
|---|---|---|
| **Application code** (this pipeline) | **GitHub** — `NiranjanWaghmare/JARVIS` | Branch `cursor/anime-episode-architecture-aa56`, PR #1 |
| **Package path** | Repo: `src/anime_episode/` | Not published to PyPI / HF |
| **Model weights** | **Hugging Face** (download into ComfyUI) | See [MODELS.md](./MODELS.md) |
| **Artifacts** (episodes, clips) | Local disk `artifacts/anime_episodes/` | gitignored |

Nothing from this app is hosted on Hugging Face. HF is only the source for open weights
(Animagine, Wan2.2, AniSora, etc.) that ComfyUI / GPT-SoVITS load at runtime.
