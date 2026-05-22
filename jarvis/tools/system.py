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

    def set_confirm_callback(self, cb: Callable[[str], Awaitable[bool]]):
        self._confirm_cb = cb

    # ── LLM natural-language dispatch ─────────────────────────────────────────
    async def dispatch(self, text: str, confirm_cb=None) -> Optional[str]:
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
        # Match against original text to preserve app/process name casing
        if m := re.search(r'\bopen\s+(?:app\s+)?([a-zA-Z0-9\s]+)', text, re.I):
            return await self.open_app(m.group(1).strip())
        if m := re.search(r'\bkill\s+(?:process\s+)?([a-zA-Z0-9\-_.]+)', text, re.I):
            return await self.kill_process(m.group(1).strip())
        if re.search(r'\b(stats|vitals|health|cpu|ram|disk|uptime)\b', t):
            return get_system_stats()
        if re.search(r'\blist\s+processes\b|\bps\b|\btop processes\b', t):
            return list_processes()
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

    async def open_app(self, name: str) -> str:
        safe_name = re.sub(r"[^\w\s\.\-]", "", name).strip()
        if not safe_name:
            return "Invalid application name."
        try:
            r = subprocess.run(["open", "-a", safe_name], capture_output=True, timeout=5)
            if r.returncode == 0:
                return f"Opened {safe_name}."
            return f"Could not open {safe_name}: {r.stderr.decode().strip()}"
        except Exception as exc:
            return f"Error opening {safe_name}: {exc}"

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
