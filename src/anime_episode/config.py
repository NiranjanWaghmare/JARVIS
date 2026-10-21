"""Runtime settings for the anime episode pipeline."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ANIME_",
        env_file=".env",
        extra="ignore",
    )

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8088
    artifact_root: Path = Field(default=Path("artifacts/anime_episodes"))

    # Queue
    redis_url: str = "redis://localhost:6379/0"
    queue_backend: str = "memory"  # memory | redis
    worker_concurrency: int = 1

    # Model / tool endpoints
    llm_model: str = "openai/gpt-4o-mini"
    llm_base_url: str | None = None
    openai_api_key: str | None = None

    comfyui_base_url: str = "http://127.0.0.1:8188"
    comfyui_timeout_s: int = 600

    tts_base_url: str = "http://127.0.0.1:9880"
    ffmpeg_bin: str = "ffmpeg"

    # Pipeline defaults (open-source stack)
    keyframe_model: str = "animagine-xl-4.0"
    motion_model: str = "anisora-v3.2"
    animate_model: str = "wan2.2-animate-14b"
    dialogue_model: str = "wan2.2-s2v-14b"
    tts_model: str = "gpt-sovits-v3"

    # ComfyUI workflow files under workflows/
    keyframe_workflow: str = "keyframe_animagine.json"
    motion_workflow: str = "motion_wan_i2v.json"
    keyframe_ckpt: str = "animagine-xl-4.0.safetensors"
    # Wan2.2 I2V MoE pair — swap for AniSora / AnimeGen weights when installed
    motion_unet_high: str = "wan2.2_i2v_high_noise_14B_fp16.safetensors"
    motion_unet_low: str = "wan2.2_i2v_low_noise_14B_fp16.safetensors"

    keyframe_width: int = 832
    keyframe_height: int = 1216
    motion_width: int = 640
    motion_height: int = 640
    motion_length: int = 81  # Wan frames (4n+1); ~5s at 16fps
    default_negative_prompt: str = (
        "lowres, bad anatomy, bad hands, text, error, missing fingers, "
        "extra digit, fewer digits, cropped, worst quality, low quality, blurry"
    )

    # Episode geometry
    target_duration_s: int = 600
    default_shot_duration_s: float = 5.0
    dry_run: bool = True  # stub tools until GPUs/backends are wired



@lru_cache
def get_settings() -> Settings:
    return Settings()
