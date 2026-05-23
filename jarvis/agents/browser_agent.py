"""
Browser Agent — LLM-guided browser automation via Playwright.

Jarvis can navigate, click, type, search, and read websites on command.

Architecture:
  - A single persistent browser session (lazy-started, stays open between tasks)
  - ReAct loop: observe page state → LLM decides next action → execute → repeat
  - Action format: flat JSON so the existing LLM pipeline handles it cleanly
  - Max steps per task prevents runaway loops
  - Returns a plain-English summary that gets spoken by TTS
"""

import asyncio
import json
import logging
import re

log = logging.getLogger("jarvis.agent.browser")

# User-intent patterns that signal browser interaction is needed
BROWSER_TRIGGERS = re.compile(
    r'\b('
    r'search\s+(for\s+|on\s+|in\s+)?'
    r'|look\s+up\s+'
    r'|find\s+.{0,30}(on|in)\s+(google|youtube|amazon|reddit|web|internet|website|the\s+site)'
    r'|on\s+the\s+(page|site|website)'
    r'|click\s+(the|on)\s+'
    r'|scroll\s+(down|up)'
    r'|fill\s+(in|out)\s+the'
    r'|type\s+in\s+'
    r'|read\s+(me\s+)?(the\s+)?(page|site|article|results?)'
    r'|go\s+back'
    r'|next\s+page'
    r'|interact\s+with'
    r'|on\s+that\s+(page|site|website)'
    r')',
    re.IGNORECASE,
)

_MAX_STEPS = 8
_PAGE_TEXT_LIMIT = 2500
_ELEMENTS_LIMIT = 25


class BrowserAgent:
    """
    Persistent browser session with LLM-directed action loop.

    Call execute_task(task, context, llm) to run a multi-step browser task.
    The browser window stays open between tasks for continuity.
    """

    def __init__(self):
        self._playwright = None
        self._browser = None
        self._page = None
        self._lock = asyncio.Lock()

    @property
    def is_open(self) -> bool:
        return self._page is not None

    async def start(self) -> None:
        """Launch the browser (called lazily on first use)."""
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise RuntimeError(
                "Playwright not installed. Run: uv pip install playwright && playwright install chromium"
            )
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=False, slow_mo=80)
        self._page = await self._browser.new_page()
        self._page.set_default_timeout(10_000)
        log.info("Browser session started.")

    async def stop(self) -> None:
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._page = self._browser = self._playwright = None
        log.info("Browser session closed.")

    # ── Public entry point ─────────────────────────────────────────────────────

    async def execute_task(self, task: str, context: str, llm) -> str:
        """
        Run the ReAct loop for a browser task.

        Returns a plain-English summary of what was done / found.
        """
        async with self._lock:
            if not self.is_open:
                await self.start()

            # If page is blank/new, auto-navigate to a search URL before the ReAct loop
            if self._page.url in ("about:blank", ""):
                search_query = task.strip()
                search_url = "https://www.google.com/search?q=" + search_query.replace(" ", "+")
                await self._execute_action({"action": "navigate", "url": search_url})
                await asyncio.sleep(1.0)

            for step in range(_MAX_STEPS):
                state = await self._get_page_state()
                action = await self._decide_action(task, context, state, llm)

                if action is None:
                    return "I couldn't figure out what to do on the page."

                act = action.get("action", "")
                log.info("Browser step %d/%d — action: %s", step + 1, _MAX_STEPS, act)

                if act == "done":
                    return action.get("summary", "Task complete.")

                result = await self._execute_action(action)
                if result.startswith("ERROR"):
                    log.warning("Browser action failed: %s", result)
                    # Feed the error back on the next step so the LLM can recover
                    context = f"{context}\n[Last action failed: {result}]"

                # Small wait for page to settle after actions
                await asyncio.sleep(0.6)

            return "I've done as much as I can — here's where things stand."

    # ── Page state extraction ──────────────────────────────────────────────────

    async def _get_page_state(self) -> dict:
        """Snapshot of the current page: URL, title, body text, interactive elements."""
        try:
            url = self._page.url
            title = await self._page.title()
            body = await self._page.evaluate("document.body.innerText")
            body = " ".join(body.split())[:_PAGE_TEXT_LIMIT]

            elements = await self._page.evaluate(f"""
                () => Array.from(document.querySelectorAll(
                    'a, button, input, textarea, select, [role=button], [role=link]'
                ))
                .filter(el => {{
                    const rect = el.getBoundingClientRect();
                    return rect.width > 0 && rect.height > 0;
                }})
                .map(el => ({{
                    tag:         el.tagName.toLowerCase(),
                    text:        (el.innerText || el.value || el.placeholder
                                 || el.getAttribute('aria-label') || '').slice(0, 60).trim(),
                    type:        el.type || '',
                    placeholder: el.placeholder || '',
                }}))
                .filter(el => el.text)
                .slice(0, {_ELEMENTS_LIMIT})
            """)

            return {"url": url, "title": title, "body": body, "elements": elements}
        except Exception as exc:
            log.warning("Page state error: %s", exc)
            return {"url": self._page.url, "title": "", "body": "", "elements": []}

    # ── LLM action decision ────────────────────────────────────────────────────

    async def _decide_action(self, task: str, context: str, state: dict, llm) -> dict | None:
        elements_text = "\n".join(
            f"  [{e['tag']}] \"{e['text']}\""
            + (f" (placeholder: {e['placeholder']})" if e.get("placeholder") else "")
            for e in state.get("elements", [])
        )
        prompt = (
            "You are controlling a web browser to complete a task. "
            "Output ONE JSON action — nothing else.\n\n"
            f"Task: {task}\n"
            f"Context: {context or 'none'}\n\n"
            f"Current page:\n"
            f"  URL:   {state['url']}\n"
            f"  Title: {state['title']}\n"
            f"  Text:  {state['body'][:1500]}\n"
            f"  Interactive elements:\n{elements_text or '  (none visible)'}\n\n"
            "Available actions (output one, flat JSON only):\n"
            '  {"action": "navigate",    "url": "https://..."}\n'
            '  {"action": "click_text",  "text": "exact visible text of element"}\n'
            '  {"action": "type_in",     "placeholder": "field placeholder", "text": "what to type"}\n'
            '  {"action": "press",       "key": "Enter"}\n'
            '  {"action": "scroll",      "direction": "down"}\n'
            '  {"action": "read",        "what": "describe what to extract"}\n'
            '  {"action": "go_back"}\n'
            '  {"action": "done",        "summary": "plain English result for the user"}\n\n'
            "Output ONLY the JSON, no explanation:"
        )
        try:
            raw = await asyncio.wait_for(
                llm.generate(prompt, model=llm.chat_model, num_predict=120),
                timeout=15.0,
            )
            # Extract JSON from the response
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if not m:
                return None
            return json.loads(m.group())
        except (asyncio.TimeoutError, json.JSONDecodeError, Exception) as exc:
            log.warning("Browser LLM decision error: %s", exc)
            return None

    # ── Action execution ───────────────────────────────────────────────────────

    async def _execute_action(self, action: dict) -> str:
        act = action.get("action", "")
        try:
            if act == "navigate":
                url = action.get("url", "")
                if not url.startswith("http"):
                    url = "https://" + url
                await self._page.goto(url, wait_until="domcontentloaded")
                return f"Navigated to {url}"

            elif act == "click_text":
                text = action.get("text", "")
                # Try increasingly broad selectors
                for loc in [
                    self._page.get_by_text(text, exact=True),
                    self._page.get_by_role("link", name=text),
                    self._page.get_by_role("button", name=text),
                    self._page.locator(f"text={text}"),
                ]:
                    try:
                        await loc.first.click(timeout=4000)
                        await self._page.wait_for_load_state("domcontentloaded", timeout=5000)
                        return f"Clicked '{text}'"
                    except Exception:
                        continue
                return f"ERROR: Could not find '{text}' to click"

            elif act == "type_in":
                placeholder = action.get("placeholder", "")
                text = action.get("text", "")
                for loc in [
                    self._page.get_by_placeholder(placeholder),
                    self._page.locator("input[type=search]"),
                    self._page.locator("input[type=text]"),
                    self._page.locator("textarea"),
                ]:
                    try:
                        await loc.first.fill(text, timeout=4000)
                        return f"Typed '{text}'"
                    except Exception:
                        continue
                return f"ERROR: Could not find input field to type in"

            elif act == "press":
                key = action.get("key", "Enter")
                await self._page.keyboard.press(key)
                await self._page.wait_for_load_state("domcontentloaded", timeout=5000)
                return f"Pressed {key}"

            elif act == "scroll":
                direction = action.get("direction", "down")
                dy = 600 if direction == "down" else -600
                await self._page.mouse.wheel(0, dy)
                return f"Scrolled {direction}"

            elif act == "go_back":
                await self._page.go_back(wait_until="domcontentloaded")
                return "Went back"

            elif act == "read":
                state = await self._get_page_state()
                return state["body"][:_PAGE_TEXT_LIMIT]

            else:
                return f"ERROR: Unknown action '{act}'"

        except Exception as exc:
            return f"ERROR: {exc}"
