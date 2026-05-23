"""
Audio Device Manager — runtime switching between Mac and iPhone audio.

iPhone input:  Continuity Microphone (shows as "iPhone Microphone" when
               iPhone is on same WiFi + Apple ID, macOS Ventura+, iOS 16+)
               OR Bluetooth headset mode.

iPhone output: AirPlay (switched via osascript) or Bluetooth audio.

Usage:
  mgr = AudioDeviceManager(vad, player)
  mgr.switch_to_iphone()     # use iPhone mic + speaker
  mgr.switch_to_mac()        # restore Mac defaults
  mgr.list_devices()         # see what's available right now
"""

import logging
import re
import subprocess
from typing import Optional

log = logging.getLogger("jarvis.audio_device")

# Patterns that identify iPhone-related audio devices
_IPHONE_INPUT_PATTERNS  = re.compile(r'iphone|continuity|ios mic', re.IGNORECASE)
_IPHONE_OUTPUT_PATTERNS = re.compile(r'iphone|airplay.*iphone|iphone.*airplay', re.IGNORECASE)

# Broad Bluetooth/external pattern as fallback
_BT_IPHONE_PATTERNS = re.compile(
    r"iphone|niranjan.*iphone|iphone.*niranjan|[\w\s]+'s\s+iphone", re.IGNORECASE
)


class AudioDeviceManager:
    """
    Manages runtime audio device switching for VADRecorder and AudioPlayer.
    Constructed once in _main() and passed to commands that need it.
    """

    def __init__(self, vad, player):
        self._vad    = vad       # VADRecorder — set ._input_device at runtime
        self._player = player    # AudioPlayer  — set .output_device at runtime
        self._mac_input:  Optional[int]   = None   # defaults restored on switch_to_mac()
        self._mac_output: Optional[str]   = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def list_devices(self) -> str:
        """Return a human-readable list of available input/output audio devices."""
        try:
            import sounddevice as sd
            devs = sd.query_devices()
            lines = ["Available audio devices:"]
            for i, d in enumerate(devs):
                tags = []
                if d["max_input_channels"] > 0:
                    tags.append("mic")
                if d["max_output_channels"] > 0:
                    tags.append("speaker")
                flag = " ← iPhone" if (_IPHONE_INPUT_PATTERNS.search(d["name"])
                                        or _BT_IPHONE_PATTERNS.search(d["name"])) else ""
                lines.append(f"  [{i}] {'/'.join(tags):12} {d['name']}{flag}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Could not list devices: {exc}"

    def find_iphone_input(self) -> Optional[int]:
        """Return device index of iPhone mic, or None if not available."""
        try:
            import sounddevice as sd
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] > 0:
                    if _IPHONE_INPUT_PATTERNS.search(d["name"]) or _BT_IPHONE_PATTERNS.search(d["name"]):
                        return i
        except Exception:
            pass
        return None

    def find_iphone_output(self) -> Optional[str]:
        """Return device name of iPhone speaker (sounddevice), or None."""
        try:
            import sounddevice as sd
            for d in sd.query_devices():
                if d["max_output_channels"] > 0:
                    if _IPHONE_OUTPUT_PATTERNS.search(d["name"]) or _BT_IPHONE_PATTERNS.search(d["name"]):
                        return d["name"]
        except Exception:
            pass
        return None

    def switch_to_iphone(self) -> str:
        """Switch both mic and speaker to iPhone. Returns status message."""
        messages = []

        # Save current Mac defaults before switching
        try:
            import sounddevice as sd
            self._mac_input  = sd.default.device[0]
            self._mac_output = sd.default.device[1]
        except Exception:
            pass

        # ── Input: Continuity Mic or Bluetooth ──────────────────────────────
        iphone_in = self.find_iphone_input()
        if iphone_in is not None:
            self._vad._input_device = iphone_in
            try:
                import sounddevice as sd
                name = sd.query_devices(iphone_in)["name"]
                messages.append(f"Mic → {name}")
                log.info("Switched input to iPhone: %s (index %d)", name, iphone_in)
            except Exception:
                messages.append("Mic → iPhone")
        else:
            messages.append("iPhone mic not found. Make sure it's nearby on the same Wi-Fi, or pair via Bluetooth.")

        # ── Output: sounddevice Bluetooth OR system AirPlay ─────────────────
        iphone_out = self.find_iphone_output()
        if iphone_out:
            self._player.output_device = iphone_out
            messages.append(f"Speaker → {iphone_out}")
            log.info("Switched output to iPhone: %s", iphone_out)
        else:
            airplay_result = self._try_airplay_switch()
            if airplay_result:
                messages.append(f"Speaker → AirPlay ({airplay_result})")
            else:
                messages.append("iPhone speaker not found. Connect via Bluetooth or enable AirPlay from Control Centre.")

        return "; ".join(messages) if messages else "Could not find iPhone audio devices."

    def switch_to_mac(self) -> str:
        """Restore Mac default mic and speaker."""
        # Restore input
        self._vad._input_device = None  # None = system default
        log.info("Restored input to Mac default.")

        # Restore output
        if self._mac_output is not None:
            self._player.output_device = self._mac_output
        else:
            self._player.output_device = None
        self._restore_system_output()
        log.info("Restored output to Mac default.")

        return "Switched back to Mac audio."

    def switch_input_only(self) -> str:
        """Switch only the microphone to iPhone."""
        iphone_in = self.find_iphone_input()
        if iphone_in is None:
            return "iPhone mic not found. Make sure it's nearby on the same Wi-Fi, or pair via Bluetooth."
        self._vad._input_device = iphone_in
        try:
            import sounddevice as sd
            name = sd.query_devices(iphone_in)["name"]
            log.info("Switched mic to: %s", name)
            return f"Now using {name} as microphone."
        except Exception:
            return "Now using iPhone microphone."

    def switch_output_only(self) -> str:
        """Switch only the speaker to iPhone."""
        iphone_out = self.find_iphone_output()
        if iphone_out:
            self._player.output_device = iphone_out
            log.info("Switched speaker to: %s", iphone_out)
            return f"Now playing through {iphone_out}."

        airplay = self._try_airplay_switch()
        if airplay:
            return f"Now playing through AirPlay: {airplay}."

        return "iPhone speaker not found. Connect via Bluetooth or enable AirPlay from Control Centre."

    # ── AirPlay via osascript ──────────────────────────────────────────────────

    def _try_airplay_switch(self) -> Optional[str]:
        """
        Attempt to switch macOS system audio output to an AirPlay device named
        like 'iPhone'. Returns device name on success, None on failure.
        """
        # List system outputs via SwitchAudioSource if installed
        try:
            out = subprocess.check_output(
                ["SwitchAudioSource", "-a", "-t", "output"],
                stderr=subprocess.DEVNULL, text=True, timeout=3,
            )
            for line in out.splitlines():
                if _BT_IPHONE_PATTERNS.search(line):
                    subprocess.run(
                        ["SwitchAudioSource", "-s", line.strip(), "-t", "output"],
                        check=True, timeout=3,
                    )
                    log.info("SwitchAudioSource → %s", line.strip())
                    return line.strip()
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass

        # Fall back to osascript (slower but no extra deps)
        script = '''
            tell application "System Preferences"
                activate
            end tell
        '''
        # Just return None — user needs to manually select AirPlay for now
        # (Programmatic AirPlay switching via osascript is very fragile)
        log.debug("AirPlay switching requires SwitchAudioSource. Install: brew install switchaudio-osx")
        return None

    def _restore_system_output(self):
        """Best-effort: restore system audio output to Mac default."""
        try:
            subprocess.run(
                ["SwitchAudioSource", "-s", "MacBook Pro Speakers", "-t", "output"],
                timeout=3, capture_output=True,
            )
        except Exception:
            pass


# ── Device-aware VADRecorder patch ─────────────────────────────────────────────

def patch_vad_for_device_switching(vad_instance):
    """
    Monkey-patch a VADRecorder so its record() method respects
    vad._input_device at call time (not just at init).
    """
    import sounddevice as sd
    import numpy as np
    import struct

    original_record = vad_instance.record

    def _record_with_device():
        device = getattr(vad_instance, "_input_device", None)
        log.debug("VAD recording on device: %s", device or "system default")

        frames = []
        pre_roll_buf = []
        speaking = False
        silence_count = 0

        with sd.InputStream(
            samplerate=vad_instance.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=vad_instance._frame_samples,
            device=device,
        ) as stream:
            while True:
                block, _ = stream.read(vad_instance._frame_samples)
                block = block.flatten()
                is_speech = vad_instance._is_speech(block)

                if not speaking:
                    pre_roll_buf.append(block.copy())
                    if len(pre_roll_buf) > vad_instance.pre_roll:
                        pre_roll_buf.pop(0)

                if is_speech:
                    if not speaking:
                        speaking = True
                        frames.extend(pre_roll_buf)
                    silence_count = 0
                    frames.append(block.copy())
                elif speaking:
                    frames.append(block.copy())
                    silence_count += 1
                    if silence_count >= vad_instance._silence_frames_max:
                        break

                if len(frames) >= vad_instance._max_frames:
                    break

        if not frames:
            return None
        return np.concatenate(frames)

    vad_instance._input_device = None   # add the attribute
    vad_instance.record = _record_with_device
    log.info("VADRecorder patched for runtime device switching.")
