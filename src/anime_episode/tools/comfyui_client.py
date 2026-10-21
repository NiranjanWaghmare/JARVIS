"""Thin ComfyUI HTTP client used by keyframe / motion / animate / S2V tools."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx


class ComfyUIClient:
    def __init__(self, base_url: str, timeout_s: int = 600) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{self.base_url}/system_stats")
                return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def queue_prompt(self, workflow: dict[str, Any], client_id: str) -> str:
        payload = {"prompt": workflow, "client_id": client_id}
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(f"{self.base_url}/prompt", json=payload)
            r.raise_for_status()
            data = r.json()
            return str(data["prompt_id"])

    async def wait_for_prompt(
        self,
        prompt_id: str,
        poll_interval_s: float = 1.5,
    ) -> dict[str, Any]:
        deadline = asyncio.get_event_loop().time() + self.timeout_s
        async with httpx.AsyncClient(timeout=30) as client:
            while asyncio.get_event_loop().time() < deadline:
                hist = await client.get(f"{self.base_url}/history/{prompt_id}")
                hist.raise_for_status()
                body = hist.json()
                if prompt_id in body:
                    return body[prompt_id]
                await asyncio.sleep(poll_interval_s)
        raise TimeoutError(f"ComfyUI prompt {prompt_id} timed out after {self.timeout_s}s")

    async def download_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.get(f"{self.base_url}/view", params=params)
            r.raise_for_status()
            return r.content
