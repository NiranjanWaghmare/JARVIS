"""
App Agent — resolves vague app descriptions to real macOS application names.

Uses a fast LLM call so Jarvis doesn't need a hardcoded alias table for every
possible way a user might refer to an application.
"""

import asyncio
import logging

log = logging.getLogger("jarvis.agent.app")

# Fast lookup for the most common aliases — avoids LLM call for obvious cases.
_QUICK_ALIASES: dict[str, str] = {
    "browser": "Safari",
    "web browser": "Safari",
    "internet": "Safari",
    "internet browser": "Safari",
    "chrome": "Google Chrome",
    "firefox": "Firefox",
    "edge": "Microsoft Edge",
    "music": "Music",
    "spotify": "Spotify",
    "mail": "Mail",
    "email": "Mail",
    "messages": "Messages",
    "imessage": "Messages",
    "facetime": "FaceTime",
    "maps": "Maps",
    "files": "Finder",
    "finder": "Finder",
    "photos": "Photos",
    "calendar": "Calendar",
    "notes": "Notes",
    "reminders": "Reminders",
    "terminal": "Terminal",
    "iterm": "iTerm2",
    "iterm2": "iTerm2",
    "code": "Visual Studio Code",
    "vscode": "Visual Studio Code",
    "vs code": "Visual Studio Code",
    "xcode": "Xcode",
    "slack": "Slack",
    "zoom": "Zoom",
    "discord": "Discord",
    "calculator": "Calculator",
    "clock": "Clock",
    "vlc": "VLC",
    "video": "VLC",
    "video player": "VLC",
    "word": "Microsoft Word",
    "excel": "Microsoft Excel",
    "powerpoint": "Microsoft PowerPoint",
    "keynote": "Keynote",
    "pages": "Pages",
    "numbers": "Numbers",
    "preview": "Preview",
    "pdf": "Preview",
    "contacts": "Contacts",
    "activity monitor": "Activity Monitor",
    "system preferences": "System Preferences",
    "system settings": "System Settings",
    "app store": "App Store",
    "whatsapp": "WhatsApp",
    "telegram": "Telegram",
}


async def resolve_app(description: str, llm) -> str:
    """
    Return the real macOS app name for a natural-language description.
    Tries the quick alias table first; falls back to an LLM call.
    """
    key = description.strip().lower()

    # Fast path
    if key in _QUICK_ALIASES:
        return _QUICK_ALIASES[key]

    if llm is None:
        return description

    prompt = (
        "You are a macOS app name resolver. "
        "Output ONLY the exact application name as it appears in /Applications. "
        "No explanation, no punctuation, just the name.\n\n"
        "Examples:\n"
        "  music player → Music\n"
        "  web → Safari\n"
        "  text editor → TextEdit\n"
        "  video conferencing → Zoom\n"
        "  code → Visual Studio Code\n\n"
        f"Description: {description}\n"
        "App name:"
    )
    try:
        result = await asyncio.wait_for(
            llm.generate(prompt, model=llm.analysis_model, num_predict=15),
            timeout=6.0,
        )
        resolved = result.strip().strip('"').strip("'").splitlines()[0].strip()
        log.debug("App agent resolved '%s' → '%s'", description, resolved)
        return resolved or description
    except Exception as exc:
        log.debug("App agent failed for '%s': %s", description, exc)
        return description
