"""
System tools — function-callable by the LLM core.

All destructive operations (shutdown, restart, kill) check the confirm_destructive
flag from config and call confirm_cb before executing.
"""

import asyncio
import logging
import os
import re
import subprocess
from typing import Callable, Awaitable, Optional

import psutil

log = logging.getLogger("jarvis.tools.system")

_DESTRUCTIVE_TRIGGERS = re.compile(
    r'\b(shutdown|restart|reboot|kill|terminate|sleep)\b', re.I
)


def get_system_stats() -> str:
    """Return a formatted multi-line system vitals string."""
    cpu   = psutil.cpu_percent(interval=0.5)
    mem   = psutil.virtual_memory()
    disk  = psutil.disk_usage("/")
    uptime_sec = int(psutil.boot_time())
    import time
    uptime = str(__import__("datetime").timedelta(seconds=int(time.time() - uptime_sec)))

    lines = [
        f"CPU:    {cpu:.1f}%",
        f"RAM:    {mem.percent:.1f}%  ({mem.used/1e9:.1f}/{mem.total/1e9:.1f} GB)",
        f"Disk:   {disk.percent:.1f}%  ({disk.used/1e9:.0f}/{disk.total/1e9:.0f} GB)",
        f"Uptime: {uptime}",
    ]

    # Top 3 CPU consumers
    try:
        procs = sorted(
            psutil.process_iter(["name", "cpu_percent"]),
            key=lambda p: p.info["cpu_percent"] or 0,
            reverse=True,
        )[:3]
        top = ", ".join(
            f"{p.info['name']}({(p.info['cpu_percent'] or 0):.0f}%)" for p in procs
        )
        lines.append(f"Top:    {top}")
    except Exception:
        pass

    return "\n".join(lines)


def list_processes() -> str:
    """Return top 10 processes by CPU usage."""
    try:
        procs = sorted(
            psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]),
            key=lambda p: p.info["cpu_percent"] or 0,
            reverse=True,
        )[:10]
        lines = [f"{'PID':>6}  {'CPU':>5}  {'MEM':>5}  NAME"]
        for p in procs:
            cpu = p.info["cpu_percent"] or 0.0
            mem = p.info["memory_percent"] or 0.0
            lines.append(
                f"{p.info['pid']:>6}  {cpu:>4.1f}%  {mem:>4.1f}%  {p.info['name']}"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"Error listing processes: {exc}"


class SystemTools:
    """Registers and dispatches system tool calls from the LLM."""

    def __init__(self, cfg: dict):
        self.confirm_destructive: bool = cfg.get("tools", {}).get("confirm_destructive", True)
        self._confirm_cb: Optional[Callable[[str], Awaitable[bool]]] = None
        self._llm = None        # injected via set_llm() after LLMCore is created
        self._audio_mgr = None  # injected via set_audio_manager() in voice mode

    def set_llm(self, llm) -> None:
        """Inject LLMCore so agents can do intelligent resolution."""
        self._llm = llm

    def set_audio_manager(self, mgr) -> None:
        """Inject AudioDeviceManager for iPhone audio switching."""
        self._audio_mgr = mgr

    def set_confirm_callback(self, cb: Callable[[str], Awaitable[bool]]):
        self._confirm_cb = cb

    # ── LLM natural-language dispatch ─────────────────────────────────────────
    async def dispatch(self, text: str, confirm_cb=None, memory_context: str = "") -> Optional[str]:
        """Check if text is a system command. Return result string or None."""
        if confirm_cb:
            self._confirm_cb = confirm_cb
        t = text.lower().strip()

        if re.search(r'\bshutdown\b|\bshut down\b|\bpower off\b', t):
            return await self.system_shutdown()
        if re.search(r'\brestart\b|\breboot\b', t):
            return await self.system_restart()
        if re.search(r'\bsleep\b|\bsuspend\b', t):
            return await self.system_sleep()
        if m := re.search(r'\bkill\s+(?:process\s+)?([a-zA-Z0-9\-_.]+)', text, re.I):
            return await self.kill_process(m.group(1).strip())
        if re.search(r'\b(stats|vitals|health|cpu|ram|disk|uptime)\b', t):
            return get_system_stats()
        if re.search(r'\blist\s+processes\b|\bps\b|\btop processes\b', t):
            return list_processes()

        # Audio device switching — iPhone mic/speaker on demand
        if self._audio_mgr is not None:
            if re.search(r'\b(use|switch\s+to|enable)\s+(iphone|phone)\b', t):
                if re.search(r'\bmic(rophone)?\b|\binput\b', t):
                    return self._audio_mgr.switch_input_only()
                if re.search(r'\bspeaker\b|\boutput\b|\baudio\b', t):
                    return self._audio_mgr.switch_output_only()
                return self._audio_mgr.switch_to_iphone()
            if re.search(r'\b(switch\s+back|use\s+mac|use\s+macbook|mac\s+audio|back\s+to\s+mac)\b', t):
                return self._audio_mgr.switch_to_mac()
            if re.search(r'\b(list|show)\s+(audio\s+)?(devices?|inputs?|outputs?|mics?|speakers?)\b', t):
                return self._audio_mgr.list_devices()

        # Navigation intent — use URL agent to resolve "take me there", "open that", etc.
        from jarvis.agents.url_agent import NAV_TRIGGERS, resolve_url
        if NAV_TRIGGERS.search(t):
            url = await resolve_url(text, memory_context, self._llm)
            if url:
                return await self.open_url(url)

        # Explicit "open <something>" — only when "open" is the leading command verb,
        # not mid-sentence (e.g. "Hermes or open claw" should NOT trigger this)
        _open_cmd = re.match(
            r'^\s*(?:please\s+|can\s+you\s+|could\s+you\s+)?(?:open|launch|start)\s+(?:app\s+)?([a-zA-Z0-9][a-zA-Z0-9\s\.\-]{1,40})',
            text, re.I
        )
        if _open_cmd:
            candidate = _open_cmd.group(1).strip()
            skip_words = {"it", "that", "this", "the", "a", "an", "up", "here", "there"}
            if candidate.lower() not in skip_words:
                from jarvis.agents.app_agent import resolve_app
                resolved = await resolve_app(candidate, self._llm)
                return await self.open_app(resolved)

        return None

    # ── individual tools (LLM function calling) ────────────────────────────────
    async def system_shutdown(self) -> str:
        if self.confirm_destructive:
            ok = await self._confirm("Shut down the system?")
            if not ok:
                return "Shutdown cancelled."
        log.warning("Executing: shutdown -h now")
        subprocess.Popen(["sudo", "shutdown", "-h", "now"])
        return "Shutting down system."

    async def system_restart(self) -> str:
        if self.confirm_destructive:
            ok = await self._confirm("Restart the system?")
            if not ok:
                return "Restart cancelled."
        log.warning("Executing: shutdown -r now")
        subprocess.Popen(["sudo", "shutdown", "-r", "now"])
        return "Restarting system."

    async def system_sleep(self) -> str:
        log.info("Executing: pmset sleepnow")
        subprocess.Popen(["pmset", "sleepnow"])
        return "Going to sleep."

    # Generic names users say → actual macOS app names
    async def open_app(self, name: str) -> str:
        """Open a macOS application by name. Name should already be resolved by app_agent."""
        clean = re.sub(r"[^\w\s\.\-]", "", name).strip()
        if not clean:
            return ""
        try:
            r = subprocess.run(["open", "-a", clean], capture_output=True, timeout=5)
            if r.returncode == 0:
                log.info("Opened app: %s", clean)
                return ""  # silent — action is visible, no need to speak it
            r2 = subprocess.run(["open", clean], capture_output=True, timeout=5)
            if r2.returncode == 0:
                return ""
            return f"Couldn't find {clean}."
        except Exception as exc:
            log.warning("open_app failed for %s: %s", clean, exc)
            return ""

    async def open_url(self, url: str) -> str:
        url = url.strip().rstrip(".,;)")
        if not re.match(r"https?://", url, re.I):
            url = "https://" + url
        try:
            subprocess.Popen(["open", url])
            log.info("Opened URL: %s", url)
            return ""  # silent — no speech for URL opens
        except Exception as exc:
            log.warning("open_url failed: %s", exc)
            return ""

    async def set_reminder(self, text: str = "", time: str = "", message: str = "", **kwargs) -> str:
        """Add a reminder to macOS Reminders app via AppleScript."""
        note = text or message or kwargs.get("reminder", kwargs.get("task", ""))
        if not note:
            return "What would you like me to remind you about, sir?"
        note = note.replace('"', "'")
        # Try to add with due date if time string provided
        if time and time.strip():
            t = time.strip().replace('"', "'")
            script = (
                f'tell application "Reminders" to make new reminder '
                f'with properties {{name:"{note}", due date:date "{t}"}}'
            )
        else:
            script = (
                f'tell application "Reminders" to make new reminder '
                f'with properties {{name:"{note}"}}'
            )
        try:
            subprocess.run(["osascript", "-e", script], timeout=5, capture_output=True)
            if time and time.strip():
                return f"Reminder set: {note} — {time}."
            return f"Added to Reminders: {note}."
        except Exception as exc:
            log.warning("set_reminder failed: %s", exc)
            return f"Couldn't set the reminder, sir. You may want to add it manually."

    async def kill_process(self, name: str) -> str:
        safe_name = re.sub(r"[^\w\.\-]", "", name).strip()
        if not safe_name:
            return "Invalid process name."
        if self.confirm_destructive:
            ok = await self._confirm(f"Kill process '{safe_name}'?")
            if not ok:
                return f"Kill '{safe_name}' cancelled."
        try:
            r = subprocess.run(["pkill", safe_name], capture_output=True, timeout=5)
            if r.returncode == 0:
                return f"Process '{safe_name}' terminated."
            return f"No process named '{safe_name}' found."
        except Exception as exc:
            return f"Error killing {safe_name}: {exc}"

    # ── helpers ────────────────────────────────────────────────────────────────
    async def _confirm(self, prompt: str) -> bool:
        if self._confirm_cb:
            return await self._confirm_cb(prompt)
        # Non-interactive fallback: require explicit env var to skip confirmation
        return os.getenv("JARVIS_SKIP_CONFIRM", "").lower() in {"1", "true"}
