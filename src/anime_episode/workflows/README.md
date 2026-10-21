# ComfyUI workflows (API format)

These JSON files are **ComfyUI API-format** graphs (node id → `{class_type, inputs, _meta}`),
not the UI “full” workflow export.

## Marker convention

Keep these `_meta.title` values on the nodes you want the Python injector to edit:

| Title | Used for |
|---|---|
| `POSITIVE_PROMPT` | Positive CLIP text |
| `NEGATIVE_PROMPT` | Negative CLIP text |
| `CHECKPOINT` | SDXL checkpoint name (keyframes) |
| `EMPTY_LATENT` | Width / height / batch |
| `LOAD_IMAGE` | Uploaded keyframe / reference image |
| `LOAD_VIDEO` | Driving clip for Wan-Animate |
| `LOAD_AUDIO` | Dialogue audio for Wan-S2V |
| `UNET` | Single diffusion model (Animate / S2V) |
| `UNET_HIGH` / `UNET_LOW` | Wan2.2 / AniSora MoE weights |
| `VIDEO_LATENT` | Width / height / length (frames) |
| `SAVE_IMAGE` / `SAVE_VIDEO` | Output nodes (not rewritten) |

## Bundled templates

| File | Stage | Default models |
|---|---|---|
| `keyframe_animagine.json` | Keyframe | `animagine-xl-4.0.safetensors` |
| `motion_wan_i2v.json` | Motion I2V | Wan2.2 I2V A14B high/low noise |
| `animate_wan.json` | Identity lock | `wan2.2_animate_14B_bf16.safetensors` |
| `dialogue_s2v.json` | Talking / lip-sync | `wan2.2_s2v_14B_fp8_scaled.safetensors` |

AniSora V3.2 / AnimeGen: install their ComfyUI weights, then set env:

```bash
export ANIME_MOTION_UNET_HIGH=anisora_v3.2_high.safetensors
export ANIME_MOTION_UNET_LOW=anisora_v3.2_low.safetensors
# or replace motion_wan_i2v.json with your exported API graph
```

## Swap your own export

1. In ComfyUI, build/load the graph, **Save (API Format)**.
2. Rename node titles (right-click → Title) to the markers above.
3. Drop the file into this folder and set `ANIME_KEYFRAME_WORKFLOW` / `ANIME_MOTION_WORKFLOW`.

## Required ComfyUI models

```
ComfyUI/models/
  diffusion_models/
    wan2.2_i2v_high_noise_14B_fp16.safetensors
    wan2.2_i2v_low_noise_14B_fp16.safetensors
    wan2.2_animate_14B_bf16.safetensors
    wan2.2_s2v_14B_fp8_scaled.safetensors
  text_encoders/
    umt5_xxl_fp8_e4m3fn_scaled.safetensors
  audio_encoders/
    wav2vec2_large_english_fp16.safetensors
  vae/
    wan_2.1_vae.safetensors
  checkpoints/
    animagine-xl-4.0.safetensors
```

Packaged from [Comfy-Org/Wan_2.2_ComfyUI_Repackaged](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged).

## GPT-SoVITS

Run `api_v2.py` (port 9880). Set per-character `voice_ref_path` / `voice_prompt_text`, or:

```bash
export ANIME_TTS_DEFAULT_REF_AUDIO=/path/to/ref.wav
export ANIME_TTS_DEFAULT_PROMPT_TEXT="reference transcript"
export ANIME_TTS_TEXT_LANG=en
curl -s localhost:8088/v1/health/tts
```

Bundled Animate/S2V graphs are **injection skeletons**. For production quality, export the
official ComfyUI template (API format), retitle marker nodes, and point
`ANIME_ANIMATE_WORKFLOW` / `ANIME_DIALOGUE_WORKFLOW` at your file.
