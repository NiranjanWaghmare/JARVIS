# Deploy this Gradio app to Hugging Face Spaces (ZeroGPU)

```bash
# 1) Token with write access: https://huggingface.co/settings/tokens
export HF_TOKEN=hf_...

# 2) Optional: choose Space id (default: <your-username>/anime-episode-generator)
export HF_SPACE_ID=YourUsername/anime-episode-generator

# 3) From repo root
pip install 'huggingface_hub>=0.26'
python3 scripts/deploy_hf_space.py
```

The script:
1. Copies `src/anime_episode` into `hf_space/anime_episode`
2. Creates the Space (Gradio SDK) if needed
3. Sets hardware to **`zero-a10g`** (ZeroGPU)
4. Uploads `hf_space/` to `https://huggingface.co/spaces/<HF_SPACE_ID>`

### Notes

- **Short clips** use Wan2.2 TI2V on ZeroGPU (`@spaces.GPU`, ~60–180s reservation).
- **Full ~10 min episodes** need a dedicated GPU host + ComfyUI weights (`MODELS.md`).
- Free accounts in good standing can host a limited number of ZeroGPU Spaces; PRO raises host + quota limits.
- First build downloads Wan weights (~tens of GB) and can take a while before the Space is READY.
- Creating Gradio Spaces may return HTTP 402 if the account cannot host that hardware — enable ZeroGPU eligibility / PRO and retry.
