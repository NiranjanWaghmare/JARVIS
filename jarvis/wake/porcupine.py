"""
Porcupine wake word listener — always-on at ~1% CPU.

Requires:
  pip install pvporcupine
  Free access key at https://console.picovoice.ai/

If no access key is configured, this listener silently disables itself
and logs a warning — other wake triggers (triple-tap, hotkey) still work.
"""

import logging
import os
import struct
from typing import Callable, Optional

log = logging.getLogger("jarvis.wake.porcupine")

try:
    import pvporcupine
    _PORCUPINE_PKG = True
except ImportError:
    _PORCUPINE_PKG = False

try:
    import sounddevice as sd
    _SD_OK = True
except ImportError:
    _SD_OK = False


class PorcupineWake:
    def __init__(self, cfg: dict, on_wake: Callable[[], None]):
        wake_cfg = cfg.get("wake", {})
        self._access_key: str = (
            wake_cfg.get("porcupine_access_key", "")
            or os.getenv("PORCUPINE_ACCESS_KEY", "")
        )
        self._keyword: str      = wake_cfg.get("wake_word", "jarvis")
        self._sensitivity: float = float(wake_cfg.get("sensitivity", 0.5))
        self._on_wake = on_wake
        self._running = False

    @property
    def available(self) -> bool:
        return bool(_PORCUPINE_PKG and _SD_OK and self._access_key)

    def run(self):
        """Blocking loop — run this in a daemon thread."""
        if not _PORCUPINE_PKG:
            log.warning("pvporcupine not installed: pip install pvporcupine")
            return
        if not _SD_OK:
            log.warning("sounddevice not installed — Porcupine unavailable.")
            return
        if not self._access_key:
            log.warning(
                "No PORCUPINE_ACCESS_KEY configured. Get a free key at "
                "https://console.picovoice.ai/ and set it in config.yaml → wake.porcupine_access_key"
            )
            return

        try:
            porcupine = pvporcupine.create(
                access_key=self._access_key,
                keywords=[self._keyword],
                sensitivities=[self._sensitivity],
            )
        except Exception as exc:
            log.error("Porcupine init failed: %s", exc)
            return

        frame_length = porcupine.frame_length
        sample_rate  = porcupine.sample_rate
        self._running = True
        log.info("Porcupine listening for '%s' (sensitivity=%.2f).", self._keyword, self._sensitivity)

        try:
            with sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
                blocksize=frame_length,
            ) as stream:
                while self._running:
                    pcm, _ = stream.read(frame_length)
                    pcm_int16 = pcm.flatten().astype("int16").tolist()
                    result = porcupine.process(pcm_int16)
                    if result >= 0:
                        log.info("Wake word '%s' detected.", self._keyword)
                        self._on_wake()
        except Exception as exc:
            log.error("Porcupine stream error: %s", exc)
        finally:
            porcupine.delete()
            self._running = False

    def stop(self):
        self._running = False
