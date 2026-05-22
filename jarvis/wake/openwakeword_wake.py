"""
OpenWakeWord always-on listener — no API key required.

Say "hey Jarvis" to trigger wake. Processes 80 ms frames at ~16 kHz
using the hey_jarvis_v0.1 onnx model.

Requires: uv pip install openwakeword onnxruntime sounddevice
"""

import logging
import numpy as np
from typing import Callable

log = logging.getLogger("jarvis.wake.openwakeword")

try:
    import openwakeword as _oww_pkg
    from openwakeword.model import Model as _OWWModel
    # Redirect feature models to onnx (tflite-runtime unavailable on Python 3.13 / macOS ARM)
    for _k in _oww_pkg.FEATURE_MODELS:
        _oww_pkg.FEATURE_MODELS[_k]["model_path"] = (
            _oww_pkg.FEATURE_MODELS[_k]["model_path"].replace(".tflite", ".onnx")
        )
    _OWW_OK = True
except ImportError:
    _OWW_OK = False
    log.warning("openwakeword not installed — run: uv pip install openwakeword onnxruntime")

try:
    import sounddevice as sd
    _SD_OK = True
except ImportError:
    _SD_OK = False


FRAME_SAMPLES = 1280   # 80 ms at 16 kHz — recommended chunk size
SAMPLE_RATE   = 16000


class OpenWakeWordWake:
    def __init__(self, cfg: dict, on_wake: Callable[[], None]):
        wake_cfg = cfg.get("wake", {})
        self._threshold: float = float(wake_cfg.get("oww_threshold", 0.5))
        self._on_wake = on_wake
        self._running = False

    @property
    def available(self) -> bool:
        return _OWW_OK and _SD_OK

    def run(self):
        """Blocking loop — run in a daemon thread."""
        if not _OWW_OK:
            log.warning("openwakeword not installed. Run: uv pip install openwakeword onnxruntime")
            return
        if not _SD_OK:
            log.warning("sounddevice not installed — OpenWakeWord unavailable.")
            return

        try:
            model = _OWWModel(wakeword_models=["hey_jarvis"], inference_framework="onnx")
        except Exception as exc:
            log.error("OpenWakeWord model load failed: %s", exc)
            return

        self._running = True
        log.info("OpenWakeWord listening for 'hey Jarvis' (threshold=%.2f).", self._threshold)

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=FRAME_SAMPLES,
            ) as stream:
                while self._running:
                    pcm, _ = stream.read(FRAME_SAMPLES)
                    audio = pcm.flatten().astype(np.int16)
                    predictions = model.predict(audio)
                    for name, score in predictions.items():
                        if score >= self._threshold:
                            log.info("Wake word detected ('%s', score=%.3f).", name, score)
                            model.reset()
                            self._on_wake()
                            break
        except Exception as exc:
            log.error("OpenWakeWord stream error: %s", exc)
        finally:
            self._running = False

    def stop(self):
        self._running = False
