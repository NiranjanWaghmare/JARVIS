#!/usr/bin/env python3
"""Minimal local MCP-style tool server for Jarvis.

This is intentionally dependency-free for now. It speaks newline-delimited JSON
over stdio with two methods:
- tools/list
- tools/call
"""

import json
import os
import subprocess
import sys


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _system_vitals(_args):
    try:
        from system_monitor import get_system_vitals, format_vitals_summary

        vitals = get_system_vitals()
        return {
            "summary": format_vitals_summary(vitals),
            "raw": vitals,
        }
    except Exception as exc:
        return {"error": str(exc)}


def _ollama_models(_args):
    try:
        result = subprocess.run(
            ["ollama", "list"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return {
            "ok": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _project_context(_args):
    files = [
        "run_jarvis.py",
        "system_monitor.py",
        "email_handler.py",
        "pyproject.toml",
        "src/my_log_agent/crew.py",
    ]
    existing = [path for path in files if os.path.exists(os.path.join(PROJECT_ROOT, path))]
    return {
        "root": PROJECT_ROOT,
        "files": existing,
        "purpose": "Jarvis local AI companion, CrewAI orchestrator, and tool automation project.",
    }


TOOLS = {
    "system_vitals": {
        "description": "Return current CPU, RAM, disk, and other system health information.",
        "handler": _system_vitals,
    },
    "ollama_models": {
        "description": "List locally installed Ollama models.",
        "handler": _ollama_models,
    },
    "project_context": {
        "description": "Return high-level context about the Jarvis project files.",
        "handler": _project_context,
    },
}


def _list_tools():
    return {
        name: {"description": spec["description"]}
        for name, spec in TOOLS.items()
    }


def handle_message(message):
    method = message.get("method")
    params = message.get("params") or {}

    if method == "tools/list":
        return {"tools": _list_tools()}

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        spec = TOOLS.get(name)
        if not spec:
            return {"error": f"Unknown tool: {name}"}
        return {"result": spec["handler"](args)}

    return {"error": f"Unknown method: {method}"}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
            response = handle_message(message)
        except Exception as exc:
            response = {"error": str(exc)}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
