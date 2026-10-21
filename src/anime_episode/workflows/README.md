# ComfyUI workflows

Place exported ComfyUI API-format JSON here:

- `keyframe_animagine.json` — Animagine XL (+ IP-Adapter / character LoRA)
- `motion_anisora.json` — Index-AniSora V3.2 or AnimeGen I2V
- `animate_wan.json` — Wan2.2-Animate-14B
- `dialogue_s2v.json` — Wan2.2-S2V-14B

Tools load these by name and inject prompts / image paths at runtime.
Until files exist, dry-run mode stubs outputs.
