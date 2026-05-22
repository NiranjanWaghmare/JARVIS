"""
Audio playback via sounddevice — gapless, no silence between chunks.

Falls back to subprocess afplay (macOS) or aplay (Linux) if sounddevice unavailable.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

log = logging.getLogger("jarvis.audio")

try:
    import sounddevice as sd
    import soundfile as sf
    _SD_OK = True
except ImportError:
    _SD_OK = False
    log.warning("sounddevice/soundfile not installed — using afplay fallback.")


class AudioPlayer:
    def __init__(self, cfg: dict):
        audio_cfg = cfg.get("audio", {})
        self.output_device = audio_cfg.get("output_device", None)
        self.sample_rate: int = audio_cfg.get("sample_rate", 22050)
        self.blocksize: int   = audio_cfg.get("blocksize", 1024)

    def play(self, file_path: str) -> bool:
        """Play a WAV/MP3/AIFF file synchronously. Returns True on success."""
        path = Path(file_path)
        if not path.exists():
            log.warning("Audio file not found: %s", file_path)
            return False

        try:
            if _SD_OK and path.suffix.lower() in {".wav", ".flac", ".ogg"}:
                return self._play_sounddevice(file_path)
            return self._play_subprocess(file_path)
        except Exception as exc:
            log.error("Playback error: %s", exc)
            return False
        finally:
            try:
                os.unlink(file_path)
            except OSError:
                pass

    # ── sounddevice (gapless) ──────────────────────────────────────────────────
    def _play_sounddevice(self, file_path: str) -> bool:
        try:
            data, sr = sf.read(file_path, dtype="float32")
            sd.play(data, samplerate=sr, device=self.output_device, blocksize=self.blocksize)
            sd.wait()
            return True
        except Exception as exc:
            log.error("sounddevice playback failed: %s", exc)
            return self._play_subprocess(file_path)

    # ── subprocess fallback ────────────────────────────────────────────────────
    def _play_subprocess(self, file_path: str) -> bool:
        import platform
        system = platform.system()
        try:
            if system == "Darwin":
                subprocess.run(["afplay", file_path], check=True, capture_output=True)
            elif system == "Linux":
                # Try aplay for WAV, mpg123 for MP3
                if file_path.endswith(".mp3"):
                    subprocess.run(["mpg123", "-q", file_path], check=True, capture_output=True)
                else:
                    subprocess.run(["aplay", "-q", file_path], check=True, capture_output=True)
            else:
                log.warning("No audio playback available on %s", system)
                return False
            return True
        except FileNotFoundError as exc:
            log.error("Playback command not found: %s", exc)
            return False
        except subprocess.CalledProcessError as exc:
            log.error("Playback failed: %s", exc)
            return False

    def play_beep(self, beep_path: Optional[str] = None):
        """Play a short wake-acknowledged beep."""
        if beep_path and Path(beep_path).exists():
            self.play(beep_path)
        else:
            # macOS system beep
            try:
                subprocess.run(["afplay", "/System/Library/Sounds/Tink.aiff"],
                               capture_output=True, timeout=2)
            except Exception:
                pass
