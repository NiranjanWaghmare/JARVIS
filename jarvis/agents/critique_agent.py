"""
Critique Agent — evaluates Jarvis responses against quality criteria.

Performance design:
  - Runs only every RUN_EVERY_N_TURNS turns (default: 3) to avoid Ollama contention
  - Waits STARTUP_DELAY seconds before starting, so the next listen cycle is already
    underway before any background LLM call is made
  - Guidelines are cached in memory (no disk read per prompt)
  - Total background time: ~5-10 s, fully invisible to the user

Used in two modes:
  1. Live background: fires after every Nth real turn (via main.py)
  2. Training loop: called directly per question (scripts/train_loop.py)
"""

import asyncio
import json
import logging
import re
import time
from pathlib import Path

log = logging.getLogger("jarvis.agent.critique")

_PASS_THRESHOLD   = 7       # score below this triggers a lesson
_GUIDELINES_FILE  = Path("memory/refined_guidelines.json")
RUN_EVERY_N_TURNS = 3       # critique only every Nth turn
STARTUP_DELAY_SEC = 4.0     # wait this long before issuing any LLM call

# ── In-memory guidelines cache ─────────────────────────────────────────────────
_guidelines_cache: str = ""
_guidelines_mtime: float = 0.0


def load_refined_guidelines() -> str:
    """Return saved lessons for injection into the system prompt (cached, fast)."""
    global _guidelines_cache, _guidelines_mtime
    try:
        if _GUIDELINES_FILE.exists():
            mtime = _GUIDELINES_FILE.stat().st_mtime
            if mtime != _guidelines_mtime:          # reload only if file changed
                data = json.loads(_GUIDELINES_FILE.read_text(encoding="utf-8"))
                lessons = data.get("lessons", [])
                if lessons:
                    lines = ["Learned improvements (apply these):"]
                    for lesson in lessons[-10:]:
                        lines.append(f"- {lesson['suggestion']}")
                    _guidelines_cache = "\n".join(lines)
                else:
                    _guidelines_cache = ""
                _guidelines_mtime = mtime
    except Exception:
        pass
    return _guidelines_cache


# ── Core critique call (also used by training loop) ────────────────────────────

async def critique(user_input: str, response: str, llm) -> dict:
    """
    Evaluate a Jarvis response. Returns:
      {"score": 1-10, "pass": bool, "issues": [...], "suggestion": "..."}
    """
    prompt = (
        'You are a quality control agent for "Jarvis", a voice AI assistant.\n'
        "Score this response 1-10 against these criteria:\n"
        "  in_character  — No AI disclaimers. No 'As an AI...'. Stays as Jarvis.\n"
        "  direct_answer — Answers from own knowledge. Does NOT suggest visiting "
        "websites, review sites, or external resources.\n"
        "  concise       — ≤ 60 words. Tight for voice.\n"
        "  helpful       — Actually addresses what was asked.\n"
        "  clean         — No filler openers, no meta-notes at the end.\n\n"
        f"User asked: {user_input}\n"
        f"Jarvis replied: {response}\n\n"
        "Output ONLY this JSON (no explanation):\n"
        '{"score": N, "issues": ["issue1", ...], "suggestion": "one-line improvement hint"}\n'
        "Score 10 if perfect. issues=[] if clean."
    )
    try:
        raw = await asyncio.wait_for(
            llm.generate(prompt, model=llm.analysis_model, num_predict=150),
            timeout=20.0,
        )
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if not m:
            return {"score": 10, "pass": True, "issues": [], "suggestion": ""}
        data = json.loads(m.group())
        score = int(data.get("score", 10))
        return {
            "score": score,
            "pass": score >= _PASS_THRESHOLD,
            "issues": data.get("issues", []),
            "suggestion": data.get("suggestion", ""),
        }
    except Exception as exc:
        log.debug("Critique failed: %s", exc)
        return {"score": 10, "pass": True, "issues": [], "suggestion": ""}


# ── Background task (called from main.py via asyncio.create_task) ───────────────

async def background_tasks(
    user_input: str,
    response: str,
    llm,
    persona,
) -> None:
    """
    Runs memory consolidation THEN critique sequentially in the background.

    Delays startup so the next voice listen cycle is already running before
    any LLM call is issued. This prevents Ollama contention with the live pipeline.
    """
    await asyncio.sleep(STARTUP_DELAY_SEC)

    # ── Memory consolidation ────────────────────────────────────────────────────
    extraction_prompt = (
        "You are a memory extraction system. Read this exchange and identify "
        "facts about the USER that would be useful to remember in future conversations.\n\n"
        "Rules:\n"
        "- Extract facts about the USER only (preferences, habits, name, context, goals)\n"
        "- Only what is clearly stated or strongly implied\n"
        "- Maximum 3 facts, or 'none'\n"
        "- Each fact under 15 words, one per line\n\n"
        f"User said: {user_input}\n"
        f"Jarvis replied: {response}\n\nFacts:"
    )
    try:
        result = await asyncio.wait_for(
            llm.generate(extraction_prompt, model=llm.analysis_model, num_predict=100),
            timeout=20.0,
        )
        for line in result.strip().splitlines():
            fact = line.strip().lstrip("-•*0123456789. ")
            if fact and fact.lower() != "none" and len(fact) > 4:
                persona.memory.save_fact(fact)
                log.debug("Memory: %s", fact)
    except Exception as exc:
        log.debug("Memory consolidation error: %s", exc)

    # Small gap so the two LLM calls don't feel like one burst to Ollama
    await asyncio.sleep(1.0)

    # ── Critique ────────────────────────────────────────────────────────────────
    result = await critique(user_input, response, llm)
    if result["pass"]:
        return

    log.info(
        "Critique %d/10 | '%s...' | issues: %s",
        result["score"], user_input[:40], result["issues"],
    )
    if result["suggestion"]:
        await _save_guideline(result["suggestion"], result["issues"])


async def _save_guideline(suggestion: str, issues: list[str]) -> None:
    global _guidelines_mtime  # force cache invalidation on next read
    _GUIDELINES_FILE.parent.mkdir(exist_ok=True)
    try:
        data: dict = {}
        if _GUIDELINES_FILE.exists():
            data = json.loads(_GUIDELINES_FILE.read_text(encoding="utf-8"))
        lessons: list = data.get("lessons", [])
        entry = {"suggestion": suggestion, "issues": issues}
        if entry not in lessons:
            lessons.append(entry)
        data["lessons"] = lessons[-50:]
        _GUIDELINES_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        _guidelines_mtime = 0.0   # invalidate cache
        log.debug("Saved guideline: %s", suggestion)
    except Exception as exc:
        log.debug("Could not save guideline: %s", exc)
