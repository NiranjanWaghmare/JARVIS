# Deploy this Gradio app to Hugging Face Spaces

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
3. Uploads `hf_space/` to `https://huggingface.co/spaces/<HF_SPACE_ID>`

### Notes

- Space runs **dry-run** (CPU). Full GPU video needs ComfyUI + weights on a GPU host.
- Creating Gradio Spaces may require a HF PRO plan or ZeroGPU — if `create_repo` returns HTTP 402, enable billing / ZeroGPU and retry.
- After deploy, open the Space URL printed by the script (also written to `hf_space/SPACE_URL.txt`).
