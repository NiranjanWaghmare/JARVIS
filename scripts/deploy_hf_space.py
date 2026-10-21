#!/usr/bin/env python3
"""Create/update the Hugging Face Space for the anime episode generator.

Usage:
  export HF_TOKEN=hf_...
  export HF_SPACE_ID=YourUser/anime-episode-generator   # optional
  PYTHONPATH=src python3 scripts/deploy_hf_space.py

Requires: pip install 'huggingface_hub>=0.26'
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SPACE_DIR = REPO_ROOT / "hf_space"
PKG_SRC = REPO_ROOT / "src" / "anime_episode"
DEFAULT_SPACE = os.environ.get("HF_SPACE_ID", "anime-episode-generator")


def _sync_package() -> None:
    dest = SPACE_DIR / "anime_episode"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        PKG_SRC,
        dest,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests", ".pytest_cache"),
    )
    # Drop heavy/dev-only bits if any
    print(f"Synced package → {dest}")


def main() -> int:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print(
            "ERROR: Set HF_TOKEN (https://huggingface.co/settings/tokens) then re-run.\n"
            "  export HF_TOKEN=hf_...\n"
            "  export HF_SPACE_ID=YourUsername/anime-episode-generator\n"
            "  python3 scripts/deploy_hf_space.py",
            file=sys.stderr,
        )
        return 2

    try:
        from huggingface_hub import HfApi, whoami
    except ImportError:
        print("ERROR: pip install 'huggingface_hub>=0.26'", file=sys.stderr)
        return 2

    api = HfApi(token=token)
    user = whoami(token=token)
    username = user["name"]
    space_id = DEFAULT_SPACE
    if "/" not in space_id:
        space_id = f"{username}/{space_id}"

    print(f"Authenticated as {username}")
    print(f"Target Space: https://huggingface.co/spaces/{space_id}")

    _sync_package()

    try:
        api.create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="gradio",
            private=False,
            exist_ok=True,
        )
    except Exception as exc:  # noqa: BLE001
        # Free accounts may need ZeroGPU / PRO for Gradio Spaces (HTTP 402).
        print(f"create_repo note: {exc}")

    # Prefer ZeroGPU for short Wan clips (full episodes still need a dedicated GPU).
    hardware = os.environ.get("HF_SPACE_HARDWARE", "zero-a10g")
    try:
        from huggingface_hub import SpaceHardware

        hw = SpaceHardware(hardware)
        api.request_space_hardware(repo_id=space_id, hardware=hw)
        print(f"Requested hardware: {hardware}")
    except Exception as exc:  # noqa: BLE001
        print(f"request_space_hardware note: {exc}")

    api.upload_folder(
        folder_path=str(SPACE_DIR),
        repo_id=space_id,
        repo_type="space",
        commit_message="Deploy ZeroGPU Wan short-clip Space",
        ignore_patterns=["**/__pycache__/**", "**/*.pyc", "SPACE_URL.txt"],
    )
    url = f"https://huggingface.co/spaces/{space_id}"
    print(f"Deployed: {url}")
    (REPO_ROOT / "hf_space" / "SPACE_URL.txt").write_text(url + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
