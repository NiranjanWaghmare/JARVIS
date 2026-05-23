"""
LLM Core — async Ollama streaming with Claude API fallback and tool dispatch.

Tool dispatch uses JSON function-calling: the LLM can emit a JSON block like
  {"tool": "open_app", "args": {"name": "Spotify"}}
which is intercepted before yielding to the TTS stage.
"""

import asyncio
import json
import logging
import os
import re
from typing import AsyncIterator, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.persona.loader import PersonaManager

log = logging.getLogger("jarvis.llm")

# Flat tool call format: {"tool": "name", "param": "value", ...}
# Flat (no nested objects) so the regex can match without recursive parsing.
_TOOL_RE = re.compile(r'\{"tool"\s*:\s*"[^"]+?"(?:\s*,\s*"[^"]+?"\s*:\s*"[^"]*?")*\s*\}', re.DOTALL)

# Lines the LLM narrates about its own actions — strip from spoken output.
_NARRATION_RE = re.compile(
    r'(?:^|\n)?[ \t]*(?:opening|navigating to|loading|launching|going to)\s+https?://[^\s]+\.?\s*',
    re.IGNORECASE | re.MULTILINE,
)


class LLMCore:
    def __init__(self, cfg: dict):
        llm_cfg = cfg.get("llm", {})
        self.chat_model: str    = llm_cfg.get("chat_model", "llama3.2:3b")
        self.analysis_model: str = llm_cfg.get("analysis_model", "lfm2.5-thinking:latest")
        self.base_url: str      = llm_cfg.get("ollama_base", "http://localhost:11434")
        self.temperature: float = llm_cfg.get("temperature", 0.8)
        self.num_predict: int   = llm_cfg.get("num_predict", 90)
        self.keep_alive: str    = llm_cfg.get("keep_alive", "10m")
        self.ctx_turns: int     = llm_cfg.get("session_context_turns", 6)
        self.claude_api_key: str = llm_cfg.get("claude_api_key", "") or os.getenv("ANTHROPIC_API_KEY", "")
        self.claude_model: str  = llm_cfg.get("claude_model", "claude-sonnet-4-6")

        # Tool registry — populated by SystemTools and EmailAgent at startup
        self._tools: dict = {}

    def register_tool(self, name: str, fn):
        """Register a callable that can be dispatched by name."""
        self._tools[name] = fn

    # ── main streaming interface ───────────────────────────────────────────────
    async def stream(self, text: str, persona: "PersonaManager") -> AsyncIterator[str]:
        """Yield response tokens. Tool calls are intercepted and executed inline."""
        messages = self._build_messages(text, persona)
        model = persona.llm_model or self.chat_model
        num_predict = persona.num_predict or self.num_predict

        buffer = ""
        async for token in self._ollama_stream(model, messages, num_predict):
            buffer += token
            # Check for complete tool call
            m = _TOOL_RE.search(buffer)
            if m:
                before = buffer[:m.start()]
                after  = buffer[m.end():]
                if before.strip():
                    yield before
                tool_result = await self._dispatch_tool(m.group())
                if tool_result:
                    yield tool_result
                buffer = after
                continue
            # Emit safe prefix (everything before a potential opening brace)
            safe, buffer = self._split_safe(buffer)
            if safe:
                safe = _NARRATION_RE.sub("", safe)
                if safe:
                    yield safe

        if buffer.strip():
            cleaned = _NARRATION_RE.sub("", buffer)
            if cleaned.strip():
                yield cleaned

    async def generate(self, prompt: str, model: str | None = None, num_predict: int | None = None) -> str:
        """Non-streaming generation for internal use."""
        messages = [{"role": "user", "content": prompt}]
        parts = []
        async for token in self._ollama_stream(
            model or self.chat_model,
            messages,
            num_predict or self.num_predict,
        ):
            parts.append(token)
        return "".join(parts)

    # ── Ollama streaming ───────────────────────────────────────────────────────
    async def _ollama_stream(self, model: str, messages: list[dict], num_predict: int) -> AsyncIterator[str]:
        loop = asyncio.get_event_loop()
        try:
            import ollama
        except ImportError:
            raise ImportError("ollama package required: pip install ollama")

        options = {"temperature": self.temperature, "num_predict": num_predict}
        token_queue: asyncio.Queue = asyncio.Queue()

        def _stream_in_thread():
            """Consume the blocking Ollama chat stream in a worker thread."""
            try:
                for chunk in ollama.chat(
                    model=model,
                    messages=messages,
                    options=options,
                    keep_alive=self.keep_alive,
                    stream=True,
                ):
                    if isinstance(chunk, dict):
                        token = chunk.get("message", {}).get("content", "")
                    else:
                        msg = getattr(chunk, "message", None)
                        token = (getattr(msg, "content", "") or "") if msg else ""
                    if token:
                        loop.call_soon_threadsafe(token_queue.put_nowait, token)
            except Exception as exc:
                loop.call_soon_threadsafe(token_queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(token_queue.put_nowait, None)  # sentinel

        import concurrent.futures
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        executor.submit(_stream_in_thread)

        try:
            while True:
                item = await token_queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    log.error("Ollama streaming error: %s", item)
                    if self.claude_api_key:
                        # Build a flat prompt for Claude fallback
                        flat = "\n".join(
                            f"{m['role'].capitalize()}: {m['content']}" for m in messages
                        )
                        async for token in self._claude_stream(flat, num_predict):
                            yield token
                    return
                yield item
        finally:
            executor.shutdown(wait=False)

    async def _claude_stream(self, prompt: str, num_predict: int) -> AsyncIterator[str]:
        """Claude API fallback — requires ANTHROPIC_API_KEY."""
        try:
            import anthropic
        except ImportError:
            log.error("anthropic package not installed for Claude fallback.")
            yield "[Claude fallback unavailable]"
            return

        client = anthropic.Anthropic(api_key=self.claude_api_key)
        loop = asyncio.get_event_loop()

        def _blocking():
            with client.messages.stream(
                model=self.claude_model,
                max_tokens=num_predict,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                return "".join(stream.text_stream)

        try:
            text = await loop.run_in_executor(None, _blocking)
            yield text
        except Exception as exc:
            log.error("Claude fallback error: %s", exc)
            yield f"[Cloud fallback failed: {exc}]"

    # ── tool dispatch ──────────────────────────────────────────────────────────
    async def _dispatch_tool(self, json_str: str) -> str | None:
        try:
            call = json.loads(json_str)
            name = call.get("tool", "")
            # Flat format: every key except "tool" is an arg
            args = {k: v for k, v in call.items() if k != "tool"}
        except json.JSONDecodeError:
            return None

        fn = self._tools.get(name)
        if fn is None:
            log.warning("Tool not registered: %s", name)
            return None  # silent — don't speak internal tool errors

        try:
            if asyncio.iscoroutinefunction(fn):
                result = await fn(**args)
            else:
                result = await asyncio.get_event_loop().run_in_executor(None, lambda: fn(**args))
            return str(result) if result is not None else None
        except Exception as exc:
            log.error("Tool %s error: %s", name, exc)
            return None  # silent — tool errors are logged, not spoken

    # ── helpers ────────────────────────────────────────────────────────────────
    def _build_messages(self, text: str, persona: "PersonaManager") -> list[dict]:
        system = persona.system_prompt
        ctx = persona.memory.get_context(self.ctx_turns)
        profile = persona.memory.user_profile_str()

        system_content = system
        if profile:
            system_content += (
                "\n\n[What I know about this user — use this when relevant, "
                "especially if they ask about past discussions or preferences]\n"
                + profile
            )
        if ctx:
            system_content += f"\n\n[This session so far]\n{ctx}"

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": text},
        ]

    @staticmethod
    def _split_safe(buf: str) -> tuple[str, str]:
        """Return (safe_prefix, remainder). Hold back from the last '{' to avoid splitting a tool call."""
        idx = buf.rfind("{")
        if idx == -1:
            return buf, ""
        return buf[:idx], buf[idx:]
