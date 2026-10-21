# Anime Episode Generator

Open-source **story → ~10 min anime episode** pipeline architecture.

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the full tool/queue design.

## Quick start (dry-run, no GPU)

```bash
pip install -e ".[anime,anime-dev]"
export PYTHONPATH=src
export ANIME_DRY_RUN=true

uvicorn anime_episode.main:app --port 8088
```

```bash
curl -s -X POST localhost:8088/v1/episodes \
  -H 'content-type: application/json' \
  -d '{"title":"Rooftop Signal","premise":"Aiko finds a radio that speaks from tomorrow.","target_duration_s":120}'

# sync end-to-end dry pipeline
curl -s -X POST localhost:8088/v1/episodes/<EPISODE_ID>/kickoff \
  -H 'content-type: application/json' \
  -d '{"sync":true}'
```

## ComfyUI + TTS (live path)

```bash
export ANIME_DRY_RUN=false
export ANIME_COMFYUI_BASE_URL=http://127.0.0.1:8188
export ANIME_TTS_BASE_URL=http://127.0.0.1:9880
export ANIME_TTS_DEFAULT_REF_AUDIO=/path/to/ref.wav
curl -s localhost:8088/v1/health/comfyui
curl -s localhost:8088/v1/health/tts
```

Templates: keyframe, motion I2V, Wan-Animate, Wan-S2V — see `workflows/README.md`.

## Tests

```bash
PYTHONPATH=src pytest src/anime_episode/tests -q
```
