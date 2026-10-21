# Anime Episode Generator — Tool Architecture

Turn a rough story into a ~10-minute anime episode by orchestrating open-source
models as **tools** behind a job queue. No single model generates the full
episode; the system plans shots, renders them independently, then assembles.

## Stack (defaults)

| Stage | Tool | Model / backend |
|---|---|---|
| Plan | `story_planner` | LLM (LiteLLM) |
| Characters | `character_build` | Animagine XL sheets |
| Keyframe | `keyframe` | Animagine XL via ComfyUI |
| Motion | `motion` | Index-AniSora V3.2 / AnimeGen / Wan2.2 I2V |
| Identity | `animate` | Wan2.2-Animate-14B |
| Voice | `tts` | GPT-SoVITS |
| Dialogue video | `dialogue_video` | Wan2.2-S2V-14B |
| Assemble | `assemble` | FFmpeg (+ Real-ESRGAN / RIFE later) |

`ANIME_DRY_RUN=true` (default) stubs GPU tools so API + queue work without VRAM.

## Component diagram

```
                 ┌────────────┐
  Client ───────►│  FastAPI   │
                 │  /v1/*     │
                 └─────┬──────┘
                       │ enqueue jobs
                       ▼
                 ┌────────────┐       ┌─────────────────┐
                 │ Job Queue  │──────►│ Worker(s)       │
                 │ memory/Redis│       │ PipelineRunner  │
                 └────────────┘       └────────┬────────┘
                                               │ calls tools
                       ┌───────────────────────┼───────────────────────┐
                       ▼                       ▼                       ▼
                 ┌──────────┐           ┌──────────┐            ┌──────────┐
                 │ LLM      │           │ ComfyUI  │            │ TTS srv  │
                 │ planner  │           │ AniSora/ │            │ GPT-SoVITS│
                 └──────────┘           │ Wan2.2   │            └──────────┘
                                        └──────────┘
                                               │
                                               ▼
                                        ┌──────────────┐
                                        │ ArtifactStore│  artifacts/anime_episodes/<ep_id>/
                                        │ episode.json │
                                        │ keyframes/   │
                                        │ motion/ …    │
                                        └──────────────┘
```

## Episode DAG

```
plan_episode
    → build_characters
        → render_shot (× N, parallelizable)
            → assemble_episode
```

### Per-shot stages

```
keyframe → motion → [animate?] → [tts → dialogue_video?] → final_shot
```

- **Action / establishing**: keyframe → motion → optional animate  
- **Dialogue**: above + TTS + Wan-S2V (uses keyframe + audio)

## Packages

```
src/anime_episode/
  api/            FastAPI routes + schemas
  domain/         Episode, Shot, Job, enums
  orchestrator/   stage graph + PipelineRunner
  queue/          JobQueue + Worker
  tools/          one tool per model stage
  storage/        ArtifactStore (JSON/files; swap to S3/Postgres later)
  workflows/      ComfyUI workflow JSON placeholders
```

## API (v1)

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/episodes` | Create episode from premise |
| GET | `/v1/episodes/{id}` | Status, shots, artifacts |
| POST | `/v1/episodes/{id}/kickoff` | `sync=true` full dry run, or enqueue plan |
| POST | `/v1/episodes/{id}/continue` | After plan: enqueue characters + shots + assemble |
| GET | `/v1/jobs/{id}` | Job status |
| POST | `/v1/worker/tick` | Dev: process N jobs in-process |

## Run locally (dry-run)

```bash
# from repo root, with package on PYTHONPATH / uv
uv sync --extra anime
uv run uvicorn anime_episode.main:app --app-dir src --port 8088

# sync dry pipeline
curl -s -X POST localhost:8088/v1/episodes \
  -H 'content-type: application/json' \
  -d '{"title":"Rooftop Signal","premise":"Aiko finds a radio that speaks from tomorrow."}'

curl -s -X POST localhost:8088/v1/episodes/<id>/kickoff \
  -H 'content-type: application/json' \
  -d '{"sync":true}'
```

Worker process (async mode):

```bash
uv run python -m anime_episode.worker_main
```

## ComfyUI wiring (keyframes + motion)

Bundled API-format templates:

- `workflows/keyframe_animagine.json` — Animagine XL stills
- `workflows/motion_wan_i2v.json` — Wan2.2 I2V (swap UNET names for AniSora)

Injection uses `_meta.title` markers (`POSITIVE_PROMPT`, `LOAD_IMAGE`, …) so you can
replace the JSON with your own ComfyUI **Save (API Format)** export without code changes.

```bash
export ANIME_DRY_RUN=false
export ANIME_COMFYUI_BASE_URL=http://127.0.0.1:8188
# optional overrides
export ANIME_KEYFRAME_CKPT=animagine-xl-4.0.safetensors
export ANIME_MOTION_UNET_HIGH=wan2.2_i2v_high_noise_14B_fp16.safetensors
export ANIME_MOTION_UNET_LOW=wan2.2_i2v_low_noise_14B_fp16.safetensors
curl -s localhost:8088/v1/health/comfyui
```

Motion path: upload keyframe → inject workflow → `/prompt` → poll history → download mp4.

## Extending further

1. Run GPT-SoVITS; set `ANIME_TTS_BASE_URL`
2. Add `animate_wan.json` + wire `AnimateTool` the same way as motion
3. Switch `ANIME_QUEUE_BACKEND=redis` once Redis adapter is implemented
4. Scale `render_shot` workers horizontally (one GPU each)

## Design rules

1. **Tools are pure stage adapters** — no HTTP or queue logic inside tools.
2. **Orchestrator owns stage order** — skip rules live in `pipeline.py`.
3. **Artifacts are paths on disk** — episode JSON is the source of truth for status.
4. **Jobs are retryable** — shot failures do not wipe the whole episode.
5. **Dry-run is first-class** — CI and local demos never need GPUs.
