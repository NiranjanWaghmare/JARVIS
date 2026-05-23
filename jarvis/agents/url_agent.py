"""
URL Agent — resolves navigation intent to a concrete URL.

Handles cases like:
  "take me there"   → finds the URL mentioned in recent conversation
  "open YouTube"    → resolves to https://youtube.com
  "show me that"    → looks at context for what "that" refers to

Uses the analysis LLM with conversation context to figure out the right URL.
"""

import asyncio
import logging
import re

log = logging.getLogger("jarvis.agent.url")

# Navigation phrases that signal the user wants to open something
NAV_TRIGGERS = re.compile(
    r'\b(take me there|go there|open that|show me|navigate there|visit that|'
    r'open it|go to it|bring me there|take me to it)\b',
    re.IGNORECASE,
)

# Well-known sites — skip LLM for these
_QUICK_URLS: dict[str, str] = {
    "youtube": "https://youtube.com",
    "google": "https://google.com",
    "gmail": "https://mail.google.com",
    "github": "https://github.com",
    "stackoverflow": "https://stackoverflow.com",
    "stack overflow": "https://stackoverflow.com",
    "reddit": "https://reddit.com",
    "twitter": "https://x.com",
    "x": "https://x.com",
    "netflix": "https://netflix.com",
    "spotify": "https://open.spotify.com",
    "amazon": "https://amazon.com",
    "wikipedia": "https://wikipedia.org",
    "chatgpt": "https://chatgpt.com",
    "openai": "https://openai.com",
    "anthropic": "https://anthropic.com",
    "huggingface": "https://huggingface.co",
    "linkedin": "https://linkedin.com",
    "facebook": "https://facebook.com",
    "instagram": "https://instagram.com",
}


def _normalise(url: str) -> str:
    url = url.strip().rstrip(".,;)'\"")
    if re.match(r'https?://', url, re.I):
        return url
    if re.match(r'www\.', url, re.I):
        return "https://" + url
    if "." in url:
        return "https://" + url
    return url


async def resolve_url(request: str, context: str, llm) -> str | None:
    """
    Given a navigation request and recent conversation context, return the URL
    to open — or None if nothing suitable can be determined.
    """
    # Check quick site table first
    req_low = request.lower()
    for keyword, url in _QUICK_URLS.items():
        if keyword in req_low:
            log.debug("URL agent quick match '%s' → %s", keyword, url)
            return url

    # Extract any URL-like string from context (most recent wins)
    if context:
        urls_in_context = re.findall(
            r'https?://[^\s<>"\',;\)\]]+|(?:www\.)?[a-zA-Z0-9\-]+\.(?:com|org|net|io|co|app|dev)[^\s<>"\',;\)\]]*',
            context,
        )
        if urls_in_context:
            candidate = _normalise(urls_in_context[-1])
            log.debug("URL agent found in context: %s", candidate)
            return candidate

    if llm is None:
        return None

    # Ask LLM to figure out the URL from context
    prompt = (
        "You are a URL resolver for a voice assistant. "
        "Given the conversation context and the user's request, output ONLY a single full URL to open. "
        "If no suitable URL can be determined, output 'none'.\n\n"
        f"Recent conversation:\n{context}\n\n"
        f"User request: {request}\n"
        "URL (or 'none'):"
    )
    try:
        result = await asyncio.wait_for(
            llm.generate(prompt, model=llm.analysis_model, num_predict=60),
            timeout=8.0,
        )
        raw = result.strip().split()[0] if result.strip() else ""
        raw = raw.strip('"').strip("'").rstrip(".,;)")
        if not raw or raw.lower() in ("none", "no", "n/a", "null"):
            return None
        url = _normalise(raw)
        log.debug("URL agent LLM resolved '%s' → %s", request, url)
        return url
    except Exception as exc:
        log.debug("URL agent failed for '%s': %s", request, exc)
        return None
