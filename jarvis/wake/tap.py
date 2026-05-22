"""
Triple-tap and hotkey wake triggers via pynput.

Triple-tap: press the configured key 3× within the tap_window_sec.
Hotkey:     Cmd+Shift+J (configurable via config.yaml → wake.hotkey_combo).

Requires: pip install pynput
"""

import logging
import time
from typing import Callable

log = logging.getLogger("jarvis.wake.tap")

import threading as _threading

try:
    from pynput import keyboard as _kb
    _PYNPUT_OK = True
except ImportError:
    _PYNPUT_OK = False
    log.warning("pynput not installed — triple-tap/hotkey unavailable. "
                "Install: pip install pynput")

# Python 3.13 + pyobjc: AXIsProcessTrusted lookup fails with KeyError inside
# pynput's daemon thread. Intercept it so the traceback doesn't pollute output.
_orig_excepthook = _threading.excepthook

def _pynput_excepthook(args):
    if isinstance(args.exc_value, KeyError) and "AXIsProcessTrusted" in str(args.exc_value):
        log.warning(
            "Triple-tap unavailable: pyobjc/pynput incompatibility on Python 3.13 "
            "(AXIsProcessTrusted). Use Enter key or Cmd+Shift+J to wake instead."
        )
        return
    _orig_excepthook(args)

_threading.excepthook = _pynput_excepthook


class TapWake:
    def __init__(self, cfg: dict, on_wake: Callable[[], None]):
        wake_cfg = cfg.get("wake", {})
        self._tap_key: str       = wake_cfg.get("triple_tap_key", "t")
        self._tap_window: float  = float(wake_cfg.get("triple_tap_window_sec", 1.0))
        self._hotkey_combo: str  = wake_cfg.get("hotkey_combo", "<cmd>+<shift>+j")
        self._on_wake = on_wake
        self._tap_times: list[float] = []
        self._running = False

    def run(self):
        """Blocking loop — run in a daemon thread."""
        if not _PYNPUT_OK:
            return

        self._running = True

        def _on_press(key):
            if not self._running:
                return False  # stop listener

            # Triple-tap detection
            try:
                char = key.char
            except AttributeError:
                char = None

            if char == self._tap_key:
                now = time.monotonic()
                self._tap_times.append(now)
                # Expire old taps outside the window
                self._tap_times = [t for t in self._tap_times if now - t <= self._tap_window]
                if len(self._tap_times) >= 3:
                    self._tap_times.clear()
                    log.info("Triple-tap detected → wake.")
                    self._on_wake()

        def _hotkey_cb():
            log.info("Hotkey %s detected → wake.", self._hotkey_combo)
            self._on_wake()

        # Hotkey listener (Cmd+Shift+J)
        hotkey_listener = None
        try:
            hotkey_listener = _kb.GlobalHotKeys({self._hotkey_combo: _hotkey_cb})
            hotkey_listener.start()
            log.info("Hotkey listener active: %s", self._hotkey_combo)
        except Exception as exc:
            log.warning("GlobalHotKeys unavailable: %s", exc)

        # Key press listener for triple-tap
        try:
            with _kb.Listener(on_press=_on_press) as listener:
                log.info("Triple-tap listener active (press '%s' ×3).", self._tap_key)
                listener.join()
        except Exception as exc:
            log.warning("Triple-tap listener unavailable (pyobjc/pynput incompatibility): %s", exc)
            log.warning("Wake via: press Enter, or use Cmd+Shift+J (if Accessibility is granted).")

        if hotkey_listener:
            try:
                hotkey_listener.stop()
            except Exception:
                pass
        self._running = False

    def stop(self):
        self._running = False
