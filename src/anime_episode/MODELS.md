# Model weights (Hugging Face → ComfyUI / GPT-SoVITS)

Code stays on **GitHub**. Download weights to your GPU machine (not into this git repo).

## Quick install (huggingface-cli)

```bash
pip install -U "huggingface_hub[cli]"

# Animagine XL 4.0 (keyframes) → ComfyUI/models/checkpoints/
huggingface-cli download cagliostrolab/animagine-xl-4.0 \
  --include "*.safetensors" \
  --local-dir ComfyUI/models/checkpoints/

# Wan 2.2 ComfyUI repack (I2V / Animate / S2V pieces live under diffusion_models, text_encoders, vae)
huggingface-cli download Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
  --local-dir /tmp/wan22_repack
# Then copy the files listed below into the matching ComfyUI/models/* folders.
```

## Required layout

```
ComfyUI/models/
  checkpoints/
    animagine-xl-4.0.safetensors          # or Opt / Opt Clear variant
  diffusion_models/
    wan2.2_i2v_high_noise_14B_fp16.safetensors
    wan2.2_i2v_low_noise_14B_fp16.safetensors
    wan2.2_animate_14B_bf16.safetensors   # or fp8 KJ scaled
    wan2.2_s2v_14B_fp8_scaled.safetensors
  text_encoders/
    umt5_xxl_fp8_e4m3fn_scaled.safetensors
  audio_encoders/
    wav2vec2_large_english_fp16.safetensors
  vae/
    wan_2.1_vae.safetensors
```

### Optional anime I2V swap

| Model | HF / source | Env override |
|---|---|---|
| Index-AniSora V3.2 | [bilibili/Index-anisora](https://github.com/bilibili/Index-anisora) / ModelScope | `ANIME_MOTION_UNET_*` |
| AnimeGen | [aidealab/AnimeGen-T2V](https://huggingface.co/aidealab/AnimeGen-T2V) | same |

## GPT-SoVITS

Clone and run API v2 locally (not stored in this repo):

```bash
git clone https://github.com/RVC-Boss/GPT-SoVITS
cd GPT-SoVITS && python api_v2.py -a 127.0.0.1 -p 9880
export ANIME_TTS_BASE_URL=http://127.0.0.1:9880
```

Put 5–30s reference WAVs per character under your episode artifacts or absolute paths,
then `PATCH /v1/episodes/{id}/characters/{char_id}` with `voice_ref_path` + `voice_prompt_text`.

## Preflight

```bash
PYTHONPATH=src python -m anime_episode.smoke_check
# or
curl -s localhost:8088/v1/health/comfyui
curl -s localhost:8088/v1/health/tts
```
