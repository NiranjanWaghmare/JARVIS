"""
Voice Activity Detection recorder.

Uses webrtcvad to filter silence before feeding audio to STT.
Falls back to a simple energy-threshold approach if webrtcvad is not installed.
"""

import array
import logging
import struct
import time
from typing import Optional

import numpy as np

log = logging.getLogger("jarvis.vad")

try:
    import webrtcvad as _wvad
    _WEBRTCVAD_OK = True
except ImportError:
    _WEBRTCVAD_OK = False
    log.warning("webrtcvad not installed — using energy-threshold VAD. "
                "Install: pip install webrtcvad-wheels")

try:
    import sounddevice as sd
    _SD_OK = True
except ImportError:
    _SD_OK = False
    log.warning("sounddevice not installed — audio recording unavailable.")


class VADRecorder:
    """Always-on recorder that uses VAD to clip silence, then returns audio."""

    def __init__(self, cfg: dict):
        vad_cfg = cfg.get("vad", {})
        self.sample_rate: int = vad_cfg.get("sample_rate", 16000)
        self.frame_ms: int = vad_cfg.get("frame_duration_ms", 30)
        self.aggressiveness: int = vad_cfg.get("aggressiveness", 2)
        self.pause_threshold: float = vad_cfg.get("pause_threshold_sec", 0.6)
        self.max_duration: float = vad_cfg.get("max_duration_sec", 15.0)
        self.pre_roll: int = vad_cfg.get("pre_roll_frames", 10)
        self.energy_floor: float = vad_cfg.get("energy_floor", 0.003)

        self._frame_samples = int(self.sample_rate * self.frame_ms / 1000)
        self._silence_frames_max = int(self.pause_threshold * 1000 / self.frame_ms)
        self._max_frames = int(self.max_duration * 1000 / self.frame_ms)

        if _WEBRTCVAD_OK:
            self._vad = _wvad.Vad(self.aggressiveness)
        else:
            self._vad = None

    # ── public ─────────────────────────────────────────────────────────────────
    def record(self) -> Optional[np.ndarray]:
        """Block until speech is captured and silence detected. Returns float32 array."""
        if not _SD_OK:
            raise ImportError("sounddevice required for audio recording.")

        log.info("Listening…")
        frames: list[np.ndarray] = []
        pre_roll_buf: list[np.ndarray] = []
        speaking = False
        silence_count = 0

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self._frame_samples,
        ) as stream:
            while True:
                block, _ = stream.read(self._frame_samples)
                block = block.flatten()

                is_speech = self._is_speech(block)

                if not speaking:
                    pre_roll_buf.append(block.copy())
                    if len(pre_roll_buf) > self.pre_roll:
                        pre_roll_buf.pop(0)

                if is_speech:
                    if not speaking:
                        speaking = True
                        frames.extend(pre_roll_buf)
                        log.debug("Speech started.")
                    silence_count = 0
                    frames.append(block.copy())
                elif speaking:
                    frames.append(block.copy())
                    silence_count += 1
                    if silence_count >= self._silence_frames_max:
                        log.debug("Silence threshold reached — end of utterance.")
                        break

                if len(frames) >= self._max_frames:
                    log.debug("Max duration reached.")
                    break

        if not frames:
            log.info("No speech detected.")
            return None

        audio = np.concatenate(frames)
        log.debug("Recorded %.2fs of audio.", len(audio) / self.sample_rate)
        return audio

    # ── private ────────────────────────────────────────────────────────────────
    def _is_speech(self, frame: np.ndarray) -> bool:
        if self._vad is not None:
            return self._webrtcvad_check(frame)
        return self._energy_check(frame)

    def _webrtcvad_check(self, frame: np.ndarray) -> bool:
        """Convert float32 frame to int16 PCM and run through webrtcvad."""
        try:
            pcm = (frame * 32767).astype("int16")
            pcm_bytes = struct.pack(f"{len(pcm)}h", *pcm)
            return self._vad.is_speech(pcm_bytes, self.sample_rate)
        except Exception:
            return self._energy_check(frame)

    def _energy_check(self, frame: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(frame ** 2)))
        return rms > self.energy_floor
