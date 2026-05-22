"""
Speech-to-Text engine — faster-whisper with CoreML/Metal on macOS.

faster-whisper uses ctranslate2 which on Apple Silicon selects Metal automatically
when compute_type="auto". The whisper.cpp CoreML path is also supported via the
`whisper-cpp-python` drop-in if installed.
"""

import logging
import time
from typing import Optional

import numpy as np

log = logging.getLogger("jarvis.stt")


class STTEngine:
    def __init__(self, cfg: dict):
        stt_cfg = cfg.get("stt", {})
        self.model_size: str = stt_cfg.get("model_size", "small")
        self.language: str   = stt_cfg.get("language", "en")
        self.device: str     = stt_cfg.get("device", "auto")
        self.compute_type: str = stt_cfg.get("compute_type", "auto")
        self.beam_size: int  = stt_cfg.get("beam_size", 3)
        self.vad_filter: bool = stt_cfg.get("vad_filter", True)
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        t0 = time.time()
        try:
            from faster_whisper import WhisperModel
            device = self._resolve_device()
            compute = self._resolve_compute(device)
            log.info("Loading whisper/%s on %s (%s)…", self.model_size, device, compute)
            self._model = WhisperModel(
                self.model_size,
                device=device,
                compute_type=compute,
            )
            log.info("Whisper model loaded in %.1fs.", time.time() - t0)
        except ImportError:
            raise ImportError(
                "faster-whisper not installed. Run: pip install faster-whisper"
            )

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import platform
            if platform.system() == "Darwin":
                return "cpu"   # faster-whisper uses CoreML via cpu on macOS
        except Exception:
            pass
        return "cpu"

    def _resolve_compute(self, device: str) -> str:
        if self.compute_type != "auto":
            return self.compute_type
        if device == "cuda":
            return "float16"
        return "int8"  # best speed on CPU / CoreML

    def transcribe(self, audio: np.ndarray) -> Optional[str]:
        """Transcribe float32 audio (16 kHz mono). Returns cleaned text or None."""
        self._load()
        t0 = time.time()
        try:
            segments, _ = self._model.transcribe(
                audio,
                language=self.language,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as exc:
            log.error("Transcription error: %s", exc)
            return None

        elapsed = time.time() - t0
        log.debug("STT: %s (%.0fms)", repr(text[:80]), elapsed * 1000)

        if not text:
            return None
        return text
