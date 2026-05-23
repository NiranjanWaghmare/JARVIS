"""
Jarvis overlay — Siri-style luminous orb that peeks up from the bottom of the screen.

States
------
  idle      → tiny dim dot
  session   → gentle breathing orb
  listening → orb with outward sound-wave rings
  thinking  → orb with rapid flicker (processing)
  speaking  → orb with strong rhythmic beat pulse

Architecture
------------
Runs in a separate process so it never blocks the async pipeline.
A background *thread* inside that process blocks on queue.get() — this is
far more reliable than queue.empty() which lies across process boundaries.
"""

import logging
import multiprocessing
import os
import signal
import time
from pathlib import Path

log = logging.getLogger("jarvis.overlay")

_PID_FILE = Path("/tmp/jarvis_overlay.pid")

IDLE      = "idle"
SESSION   = "session"
LISTENING = "listening"
THINKING  = "thinking"
SPEAKING  = "speaking"


def _kill_existing_overlay():
    """Kill any overlay process left over from a previous run."""
    if not _PID_FILE.exists():
        return
    try:
        pid = int(_PID_FILE.read_text().strip())
        os.kill(pid, signal.SIGKILL)
        log.info("Killed stale overlay process (pid=%d).", pid)
    except (ProcessLookupError, ValueError):
        pass  # already dead or bad PID
    finally:
        _PID_FILE.unlink(missing_ok=True)


# ── Controller (lives in the main Jarvis process) ─────────────────────────────
class OverlayController:
    def __init__(self, cfg: dict | None = None):
        self._queue: multiprocessing.Queue = multiprocessing.Queue()
        self._proc: multiprocessing.Process | None = None

    def start(self):
        _kill_existing_overlay()
        self._proc = multiprocessing.Process(
            target=_run_overlay_process,
            args=(self._queue,),
            daemon=True,
        )
        self._proc.start()
        _PID_FILE.write_text(str(self._proc.pid))
        log.info("Overlay process started (pid=%d).", self._proc.pid)

    def set_state(self, state: str):
        if self._proc and self._proc.is_alive():
            self._queue.put_nowait(state)

    def stop(self):
        if self._proc and self._proc.is_alive():
            self._queue.put_nowait("quit")
            self._proc.join(timeout=2)
            if self._proc.is_alive():
                self._proc.kill()
        _PID_FILE.unlink(missing_ok=True)


# ── Child-process entry point ──────────────────────────────────────────────────
def _run_overlay_process(queue: multiprocessing.Queue):
    import logging as _L, traceback as _tb
    _L.basicConfig(level=_L.INFO,
                   format="%(asctime)s [OVERLAY] %(message)s",
                   handlers=[_L.StreamHandler()])
    _log = _L.getLogger("jarvis.overlay.proc")
    _log.info("Overlay process started.")
    try:
        _log.info("Trying Cocoa overlay…")
        _run_cocoa_overlay(queue)
    except Exception as exc:
        _log.warning("Cocoa failed (%s) — falling back to tkinter.", exc)
        _log.debug(_tb.format_exc())
        try:
            _run_tkinter_overlay(queue)
        except Exception as e2:
            _log.error("tkinter also failed: %s", e2)


# ══════════════════════════════════════════════════════════════════════════════
#  macOS Cocoa overlay
# ══════════════════════════════════════════════════════════════════════════════
def _run_cocoa_overlay(queue: multiprocessing.Queue):
    import objc          # type: ignore
    from AppKit import ( # type: ignore
        NSApplication, NSWindow, NSView, NSColor, NSBezierPath,
        NSScreen, NSBackingStoreBuffered, NSBorderlessWindowMask,
        NSWindowCollectionBehaviorCanJoinAllSpaces,
        NSWindowCollectionBehaviorStationary,
        NSTimer, NSGraphicsContext,
    )
    import math, threading, os

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(1)           # NSApplicationActivationPolicyAccessory
    app.activateIgnoringOtherApps_(True)

    screen  = NSScreen.mainScreen().frame()
    visible = NSScreen.mainScreen().visibleFrame()   # excludes menu bar + Dock
    W       = 400                         # window width
    H       = 150                         # taller so orb clears the Dock
    x       = (screen.size.width - W) / 2
    y       = visible.origin.y            # window bottom sits at top of Dock

    win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        ((x, y), (W, H)),
        NSBorderlessWindowMask, NSBackingStoreBuffered, False,
    )
    win.setOpaque_(False)
    win.setBackgroundColor_(NSColor.clearColor())
    win.setLevel_(1000)                   # NSScreenSaverWindowLevel
    win.setCollectionBehavior_(
        NSWindowCollectionBehaviorCanJoinAllSpaces |
        NSWindowCollectionBehaviorStationary
    )
    win.setIgnoresMouseEvents_(True)

    state_box = [IDLE]
    phase_box = [0.0]
    quit_flag = [False]

    # ── Background thread: blocks on queue.get() reliably ─────────────────────
    parent_pid = os.getppid()

    def _queue_thread():
        while not quit_flag[0]:
            # Die with parent
            if os.getppid() != parent_pid:
                quit_flag[0] = True
                return
            try:
                cmd = queue.get(timeout=0.5)
                if cmd == "quit":
                    quit_flag[0] = True
                    return
                if cmd in (IDLE, SESSION, LISTENING, THINKING, SPEAKING):
                    state_box[0] = cmd
            except Exception:
                pass  # queue.Empty or similar — just loop

    threading.Thread(target=_queue_thread, daemon=True, name="overlay-queue").start()

    # ── Orb drawing ────────────────────────────────────────────────────────────
    # Siri-style: the orb is centred near the BOTTOM of the window so only the
    # upper portion peeks above the screen edge (y=0 in Cocoa = screen bottom).
    CX = W / 2.0
    CY = 10.0         # orb centre near window bottom; window y sits at top of Dock

    class OrbView(NSView):
        def drawRect_(self, rect):
            NSColor.clearColor().setFill()
            NSBezierPath.fillRect_(rect)
            _draw_orb(CX, CY, phase_box[0], state_box[0])

        def isOpaque(self): return False

        def tickTimer_(self, _timer):
            if quit_flag[0]:
                app.terminate_(None)
                return
            phase_box[0] += 0.04
            if state_box[0] == IDLE:
                win.orderOut_(None)   # hide window entirely when idle
            else:
                win.orderFrontRegardless()
                self.setNeedsDisplay_(True)

        tickTimer_ = objc.selector(tickTimer_, signature=b"v@:@")

    def _draw_orb(cx, cy, phase, state):
        # ── IDLE: hidden (window is ordered out) ──────────────────────────────
        if state == IDLE:
            return

        # ── Orb radius (pulsing per state) ────────────────────────────────────
        if state == LISTENING:
            R = 88 + 5 * math.sin(phase * 1.6)
        elif state == THINKING:
            R = 84 + 4 * math.sin(phase * 1.2) + 3 * math.sin(phase * 11.3)
        elif state == SPEAKING:
            R = 90 + 10 * abs(math.sin(phase * 3.8))
        else:                                           # SESSION
            R = 84 + 5 * math.sin(phase * 0.8)

        # ── Outer aura ────────────────────────────────────────────────────────
        if state == LISTENING:
            for i in range(1, 4):
                wp = phase * 2.4 - i * 0.9
                r  = R + 11 * i + 7 * abs(math.sin(wp))
                a  = 0.22 * (4 - i) * (0.5 + 0.5 * math.sin(wp))
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.15, 0.5, 1.0, a).setFill()
                NSBezierPath.bezierPathWithOvalInRect_(
                    ((cx - r, cy - r), (r * 2, r * 2))).fill()

        elif state == THINKING:
            flicker = 0.3 + 0.7 * abs(math.sin(phase * 11.3 + 0.7))
            for i in range(1, 3):
                r = R + 14 * i
                a = 0.25 * (3 - i) * flicker
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.25, 0.55, 1.0, a).setFill()
                NSBezierPath.bezierPathWithOvalInRect_(
                    ((cx - r, cy - r), (r * 2, r * 2))).fill()

        elif state == SPEAKING:
            beat = abs(math.sin(phase * 4.2))
            for i in range(1, 5):
                r = R + 20 * i * beat
                a = 0.24 * (5 - i) / 4.0 * beat
                NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.1, 0.4, 1.0, a).setFill()
                NSBezierPath.bezierPathWithOvalInRect_(
                    ((cx - r, cy - r), (r * 2, r * 2))).fill()

        else:                                           # SESSION
            r = R + 16
            a = 0.16 * (0.5 + 0.5 * math.sin(phase * 0.7))
            NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.2, 0.5, 1.0, a).setFill()
            NSBezierPath.bezierPathWithOvalInRect_(
                ((cx - r, cy - r), (r * 2, r * 2))).fill()

        # ── Clip inner drawing to orb circle ──────────────────────────────────
        clip = NSBezierPath.bezierPathWithOvalInRect_(
            ((cx - R, cy - R), (R * 2, R * 2))
        )
        ctx = NSGraphicsContext.currentContext()
        ctx.saveGraphicsState()
        clip.setClip()

        # Deep blue base — 0.72 alpha keeps it translucent
        NSColor.colorWithCalibratedRed_green_blue_alpha_(
            0.04, 0.17, 0.88, 0.72).setFill()
        NSBezierPath.bezierPathWithRect_(
            ((cx - R - 2, cy - R - 2), (R * 2 + 4, R * 2 + 4))
        ).fill()

        # Cloud speed + flicker
        if state == THINKING:
            spd     = 3.0
            flicker = 0.45 + 0.55 * abs(math.sin(phase * 17.1))
        elif state == SPEAKING:
            spd     = 1.7
            flicker = 0.75 + 0.25 * abs(math.sin(phase * 4.2))
        else:
            spd     = 0.85
            flicker = 1.0

        clouds = [
            (  0.0,  16.0, 60, 0.95, 0.00),
            (-19.0,   5.0, 42, 0.85, 0.55),
            ( 21.0,  -4.0, 36, 0.75, 1.10),
            ( -4.0, -21.0, 30, 0.62, 1.68),
        ]
        for ox, oy, size, max_a, ph_off in clouds:
            t  = phase * spd + ph_off
            dx = ox + 9 * math.sin(t * 0.87)
            dy = oy + 6 * math.cos(t * 1.13)
            a  = max(0.0, min(1.0, max_a * (0.65 + 0.35 * math.sin(t * 0.62)) * flicker))
            rc = min(1.0, 0.58 + 0.32 * a)
            gc = min(1.0, 0.76 + 0.18 * a)
            NSColor.colorWithCalibratedRed_green_blue_alpha_(rc, gc, 1.0, a).setFill()
            NSBezierPath.bezierPathWithOvalInRect_(
                ((cx + dx - size / 2, cy + dy - size / 2), (size, size))
            ).fill()

        # Bright white core
        t_c = phase * 1.4 * spd
        cs  = 25 + 9 * math.sin(t_c)
        ca  = max(0.0, min(1.0, (0.82 + 0.18 * math.sin(t_c * 0.7)) * flicker))
        NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 1.0, 1.0, ca).setFill()
        NSBezierPath.bezierPathWithOvalInRect_(
            ((cx - cs / 2, cy + 8 - cs / 2), (cs, cs))
        ).fill()

        ctx.restoreGraphicsState()

    view = OrbView.alloc().initWithFrame_(((0, 0), (W, H)))
    win.setContentView_(view)
    app.finishLaunching()
    win.orderFrontRegardless()

    NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        0.033, view, view.tickTimer_, None, True
    )
    app.run()


# ══════════════════════════════════════════════════════════════════════════════
#  tkinter fallback
# ══════════════════════════════════════════════════════════════════════════════
def _run_tkinter_overlay(queue: multiprocessing.Queue):
    import tkinter as tk
    import math, threading, os

    W, H = 260, 120
    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.configure(bg="black")
    root.attributes("-alpha", 0.95)

    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    # Bottom of window flush with bottom of screen (Siri-style)
    root.geometry(f"{W}x{H}+{(sw - W) // 2}+{sh - H}")

    canvas = tk.Canvas(root, width=W, height=H, bg="black", highlightthickness=0)
    canvas.pack()

    state_box = [IDLE]
    phase_box = [0.0]
    quit_flag = [False]

    parent_pid = os.getppid()

    def _queue_thread():
        while not quit_flag[0]:
            if os.getppid() != parent_pid:
                quit_flag[0] = True
                return
            try:
                cmd = queue.get(timeout=0.5)
                if cmd == "quit":
                    quit_flag[0] = True
                    return
                if cmd in (IDLE, SESSION, LISTENING, THINKING, SPEAKING):
                    state_box[0] = cmd
            except Exception:
                pass

    threading.Thread(target=_queue_thread, daemon=True, name="overlay-queue").start()

    # Orb centre near the BOTTOM of canvas (tkinter y=0 is top)
    CX = W // 2
    CY = H - 8     # 8 px above canvas bottom = near screen bottom

    def _hex(r, g, b):
        return "#{:02x}{:02x}{:02x}".format(
            max(0, min(255, int(r * 255))),
            max(0, min(255, int(g * 255))),
            max(0, min(255, int(b * 255))),
        )

    def _animate():
        if quit_flag[0]:
            root.destroy()
            return

        phase = phase_box[0]
        phase_box[0] += 0.05
        state = state_box[0]

        canvas.delete("all")
        canvas.create_rectangle(0, 0, W, H, fill="black", outline="")

        cx, cy = CX, CY

        if state == IDLE:
            root.withdraw()           # hide window entirely when idle
            root.after(100, _animate)
            return

        root.deiconify()

        if state == LISTENING:
            R = int(88 + 5 * math.sin(phase * 1.6))
        elif state == THINKING:
            R = int(84 + 4 * math.sin(phase * 1.2) + 3 * math.sin(phase * 11.3))
        elif state == SPEAKING:
            R = int(90 + 10 * abs(math.sin(phase * 3.8)))
        else:
            R = int(84 + 5 * math.sin(phase * 0.8))

        # Outer aura
        if state == SPEAKING:
            beat = abs(math.sin(phase * 4.2))
            for i in range(4, 0, -1):
                r = R + int(20 * i * beat)
                a = 0.24 * (5 - i) / 4.0 * beat
                canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                   outline=_hex(0.1 * a, 0.4 * a, a), width=1)
        elif state == LISTENING:
            for i in range(3, 0, -1):
                wp = phase * 2.4 - i * 0.9
                r  = R + 11 * i + int(7 * abs(math.sin(wp)))
                a  = 0.22 * (4 - i) * (0.5 + 0.5 * math.sin(wp))
                canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                   outline=_hex(0.15 * a, 0.5 * a, a), width=1)
        elif state == THINKING:
            flicker = 0.3 + 0.7 * abs(math.sin(phase * 11.3 + 0.7))
            for i in range(2, 0, -1):
                r = R + 14 * i
                a = 0.25 * (3 - i) * flicker
                canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                   outline=_hex(0.25 * a, 0.55 * a, a), width=1)

        # Main orb
        canvas.create_oval(cx - R, cy - R, cx + R, cy + R,
                           fill=_hex(0.04, 0.17, 0.88), outline="")

        # Cloud spots
        if state == THINKING:
            spd     = 3.0
            flicker = 0.45 + 0.55 * abs(math.sin(phase * 17.1))
        elif state == SPEAKING:
            spd     = 1.7
            flicker = 0.75 + 0.25 * abs(math.sin(phase * 4.2))
        else:
            spd     = 0.85
            flicker = 1.0

        clouds = [
            (  0,  16, 60, 0.95, 0.00),
            (-19,   5, 42, 0.85, 0.55),
            ( 21,  -4, 36, 0.75, 1.10),
            ( -4, -21, 30, 0.62, 1.68),
        ]
        for ox, oy, size, max_a, ph_off in clouds:
            t  = phase * spd + ph_off
            dx = int(ox + 9 * math.sin(t * 0.87))
            dy = int(oy + 6 * math.cos(t * 1.13))
            a  = max(0.0, min(1.0, max_a * (0.65 + 0.35 * math.sin(t * 0.62)) * flicker))
            gx, gy = cx + dx, cy - dy   # tkinter y is inverted
            s2 = int(size / 2)
            if abs(dx) + abs(dy) < R:
                rc = min(1.0, 0.58 + 0.32 * a)
                gc = min(1.0, 0.76 + 0.18 * a)
                canvas.create_oval(gx - s2, gy - s2, gx + s2, gy + s2,
                                   fill=_hex(rc * a, gc * a, min(1.0, a)), outline="")

        # White core
        t_c = phase * 1.4 * spd
        cs  = int(25 + 9 * math.sin(t_c))
        ca  = max(0.0, min(1.0, (0.82 + 0.18 * math.sin(t_c * 0.7)) * flicker))
        canvas.create_oval(cx - cs // 2, cy - 8 - cs // 2,
                           cx + cs // 2, cy - 8 + cs // 2,
                           fill=_hex(ca, ca, ca), outline="")

        root.lift()
        root.attributes("-topmost", True)
        root.after(33, _animate)

    root.after(50, _animate)
    root.mainloop()


# ── standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    q: multiprocessing.Queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_run_overlay_process, args=(q,), daemon=True)
    p.start()
    time.sleep(1)
    for state in [SESSION, LISTENING, THINKING, SPEAKING, IDLE]:
        print(f"→ {state}")
        q.put(state)
        time.sleep(3)
    q.put("quit")
    p.join()
