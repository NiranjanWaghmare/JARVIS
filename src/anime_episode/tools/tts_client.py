"""GPT-SoVITS HTTP client (api_v2 /tts)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx


class TTSError(RuntimeError):
    pass


class GPTSoVITSClient:
    """Client for GPT-SoVITS `api_v2.py` (default port 9880)."""

    def __init__(self, base_url: str, timeout_s: int = 180) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    async def health(self) -> bool:
        # api_v2 has no dedicated health route; probe /docs or /openapi.json.
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                for path in ("/docs", "/openapi.json", "/"):
                    r = await client.get(f"{self.base_url}{path}")
                    if r.status_code < 500:
                        return True
        except httpx.HTTPError:
            return False
        return False

    async def synthesize(
        self,
        *,
        text: str,
        text_lang: str,
        ref_audio_path: str,
        prompt_lang: str,
        prompt_text: str = "",
        media_type: str = "wav",
        speed_factor: float = 1.0,
        extra: dict[str, Any] | None = None,
    ) -> bytes:
        if not text.strip():
            raise TTSError("text is empty")
        if not Path(ref_audio_path).exists():
            raise TTSError(f"ref audio not found: {ref_audio_path}")

        payload: dict[str, Any] = {
            "text": text,
            "text_lang": text_lang,
            "ref_audio_path": str(Path(ref_audio_path).resolve()),
            "prompt_lang": prompt_lang,
            "prompt_text": prompt_text,
            "media_type": media_type,
            "streaming_mode": False,
            "speed_factor": speed_factor,
        }
        if extra:
            payload.update(extra)

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(f"{self.base_url}/tts", json=payload)
            if r.status_code >= 400:
                raise TTSError(f"GPT-SoVITS /tts failed ({r.status_code}): {r.text[:1500]}")
            content_type = r.headers.get("content-type", "")
            if "application/json" in content_type:
                # Error payloads sometimes return JSON with 200.
                raise TTSError(f"GPT-SoVITS returned JSON instead of audio: {r.text[:1500]}")
            return r.content
