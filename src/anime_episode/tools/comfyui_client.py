"""Thin ComfyUI HTTP client used by keyframe / motion / animate / S2V tools."""

from __future__ import annotations

import asyncio
import mimetypes
from pathlib import Path
from typing import Any

import httpx


class ComfyUIError(RuntimeError):
    pass


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

    async def ensure_healthy(self) -> None:
        if not await self.health():
            raise ComfyUIError(
                f"ComfyUI not reachable at {self.base_url}. "
                "Start ComfyUI or set ANIME_DRY_RUN=true."
            )

    async def upload_image(
        self,
        path: str | Path,
        *,
        overwrite: bool = True,
        image_type: str = "input",
        subfolder: str = "",
    ) -> str:
        """Upload a local image; returns the filename ComfyUI stored under input/."""
        return await self._upload(path, field="image", overwrite=overwrite, folder_type=image_type, subfolder=subfolder)

    async def upload_video(
        self,
        path: str | Path,
        *,
        overwrite: bool = True,
        subfolder: str = "",
    ) -> str:
        """Upload a driving video into ComfyUI's input folder."""
        return await self._upload(path, field="image", overwrite=overwrite, folder_type="input", subfolder=subfolder)

    async def upload_audio(
        self,
        path: str | Path,
        *,
        overwrite: bool = True,
        subfolder: str = "",
    ) -> str:
        """Upload dialogue audio into ComfyUI's input folder for LoadAudio."""
        return await self._upload(path, field="image", overwrite=overwrite, folder_type="input", subfolder=subfolder)

    async def _upload(
        self,
        path: str | Path,
        *,
        field: str,
        overwrite: bool,
        folder_type: str,
        subfolder: str,
    ) -> str:
        path = Path(path)
        if not path.exists():
            raise ComfyUIError(f"file not found: {path}")
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = {
            "overwrite": "true" if overwrite else "false",
            "type": folder_type,
            "subfolder": subfolder,
        }
        files = {field: (path.name, path.read_bytes(), mime)}
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(f"{self.base_url}/upload/image", data=data, files=files)
            r.raise_for_status()
            body = r.json()
        name = body.get("name") or path.name
        return str(name)

    async def queue_prompt(self, workflow: dict[str, Any], client_id: str) -> str:
        payload = {"prompt": workflow, "client_id": client_id}
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(f"{self.base_url}/prompt", json=payload)
            if r.status_code >= 400:
                detail = r.text[:2000]
                raise ComfyUIError(f"ComfyUI /prompt failed ({r.status_code}): {detail}")
            data = r.json()
            if "error" in data:
                raise ComfyUIError(f"ComfyUI rejected prompt: {data['error']}")
            node_errors = data.get("node_errors") or {}
            if node_errors:
                raise ComfyUIError(f"ComfyUI node_errors: {node_errors}")
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
                    entry = body[prompt_id]
                    status = (entry.get("status") or {}).get("status_str")
                    if status == "error":
                        raise ComfyUIError(f"ComfyUI prompt {prompt_id} failed: {entry.get('status')}")
                    return entry
                await asyncio.sleep(poll_interval_s)
        raise TimeoutError(f"ComfyUI prompt {prompt_id} timed out after {self.timeout_s}s")

    async def download_file(
        self,
        filename: str,
        *,
        subfolder: str = "",
        folder_type: str = "output",
    ) -> bytes:
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.get(f"{self.base_url}/view", params=params)
            r.raise_for_status()
            return r.content

    async def download_image(
        self,
        filename: str,
        subfolder: str = "",
        folder_type: str = "output",
    ) -> bytes:
        return await self.download_file(filename, subfolder=subfolder, folder_type=folder_type)

    async def run_workflow(
        self,
        workflow: dict[str, Any],
        *,
        client_id: str,
    ) -> dict[str, Any]:
        await self.ensure_healthy()
        prompt_id = await self.queue_prompt(workflow, client_id=client_id)
        return await self.wait_for_prompt(prompt_id)
