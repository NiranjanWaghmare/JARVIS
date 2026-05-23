"""
Jarvis Functionality Test Suite — runs offline (no mic/TTS/audio).

Tests:
  1. Module imports
  2. Persona system (prompt building, mood, memory)
  3. LLM response quality (character, directness, no website deflection)
  4. Agent logic (app_agent aliases, url_agent nav triggers)
  5. System dispatch routing
  6. Critique agent scoring
  7. Guidelines loading (cache)
  8. Tool call regex interception
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from dotenv import load_dotenv
load_dotenv()

_CFG_PATH = Path(__file__).parent.parent / "jarvis" / "config.yaml"
with open(_CFG_PATH) as f:
    CFG: dict = yaml.safe_load(f)

PASS = "✓"
FAIL = "✗"
WARN = "⚠"
results = []


def record(section, name, ok, detail=""):
    status = PASS if ok else FAIL
    results.append((section, name, ok, detail))
    line = f"  {status} {name}"
    if detail:
        line += f"  [{detail}]"
    print(line)


# ── 1. Module imports ──────────────────────────────────────────────────────────
print("\n[1] Module Imports")

def test_imports():
    imports = [
        ("LLMCore",          "from jarvis.pipeline.llm import LLMCore"),
        ("TTSEngine",        "from jarvis.pipeline.tts import TTSEngine"),
        ("AudioPlayer",      "from jarvis.pipeline.audio import AudioPlayer"),
        ("PersonaManager",   "from jarvis.persona.loader import PersonaManager"),
        ("SystemTools",      "from jarvis.tools.system import SystemTools, get_system_stats"),
        ("BrowserAgent",     "from jarvis.agents.browser_agent import BrowserAgent, BROWSER_TRIGGERS"),
        ("AppAgent",         "from jarvis.agents.app_agent import resolve_app, _QUICK_ALIASES"),
        ("UrlAgent",         "from jarvis.agents.url_agent import resolve_url, NAV_TRIGGERS, _QUICK_URLS"),
        ("CritiqueAgent",    "from jarvis.agents.critique_agent import critique, background_tasks, load_refined_guidelines"),
        ("OverlayController","from jarvis.overlay.overlay import OverlayController, IDLE, SESSION, LISTENING, THINKING, SPEAKING"),
    ]
    for name, stmt in imports:
        try:
            exec(stmt, {})
            record("imports", name, True)
        except Exception as e:
            record("imports", name, False, str(e)[:60])

test_imports()


# ── 2. Persona system ──────────────────────────────────────────────────────────
print("\n[2] Persona System")

def test_persona():
    from jarvis.persona.loader import PersonaManager, PERSONAS
    p = PersonaManager(CFG)

    # Default persona is Jarvis
    record("persona", "default=jarvis", p._persona_name == "jarvis")

    # System prompt contains key rules
    sp = p.system_prompt
    record("persona", "prompt has IDENTITY rule",    "IDENTITY" in sp)
    record("persona", "prompt has TOOL section",     "TOOLS" in sp)
    record("persona", "prompt has no-filler rule",   "filler" in sp.lower() or "preamble" in sp.lower())
    record("persona", "prompt has no-website rule",  "website" in sp.lower())

    # Memory starts fresh (no old turns pre-loaded)
    ctx = p.memory.get_context(10)
    record("persona", "fresh session (no pre-loaded turns)", ctx == "")

    # Persona switching
    reply = p.try_switch("switch to stewie")
    record("persona", "switch to stewie works", reply is not None and "Stewie" in (reply or ""))
    p._switch_to("jarvis")  # reset

    # Mood engine (Stewie only)
    p._switch_to("stewie")
    p.update_mood("you're brilliant")
    record("persona", "stewie mood → happy", p._mood == "happy")
    p.update_mood("you're an idiot")
    record("persona", "stewie mood → annoyed", p._mood == "annoyed")
    p._switch_to("jarvis")

test_persona()


# ── 3. LLM response quality ────────────────────────────────────────────────────
print("\n[3] LLM Response Quality  (calls Ollama — takes ~2 min)")

async def test_llm_quality():
    from jarvis.pipeline.llm import LLMCore
    from jarvis.persona.loader import PersonaManager
    llm = LLMCore(CFG)
    persona = PersonaManager(CFG)

    quality_cases = [
        # (question, must_NOT_contain, label)
        ("Are you an AI or a language model?",
         ["as an ai", "language model", "i was trained", "huggingface"],
         "no AI identity leak"),
        ("Tell me how to make great coffee at home.",
         ["coffee review", "check out", "visit the website", "you can find", "intelligentsia"],
         "no website deflection on factual"),
        ("What is the best programming language to learn?",
         ["check online", "visit", "look it up", "great question!", "not quite"],
         "no filler/deflection on opinion"),
        ("Hey Jarvis.",
         ["note:", "i've added", "in this response", "(note"],
         "no meta-commentary"),
    ]

    for question, bad_phrases, label in quality_cases:
        try:
            tokens = []
            async for tok in llm.stream(question, persona):
                tokens.append(tok)
            response = "".join(tokens).lower()
            found = [p for p in bad_phrases if p in response]
            record("llm_quality", label, len(found) == 0,
                   f"found: {found}" if found else "")
        except Exception as e:
            record("llm_quality", label, False, str(e)[:60])

asyncio.run(test_llm_quality())


# ── 4. Agent logic ─────────────────────────────────────────────────────────────
print("\n[4] Agent Logic")

def test_agents():
    from jarvis.agents.app_agent import _QUICK_ALIASES
    from jarvis.agents.url_agent import NAV_TRIGGERS, _QUICK_URLS, _normalise
    from jarvis.agents.browser_agent import BROWSER_TRIGGERS
    import re

    # App aliases
    record("agents", "browser → Safari",    _QUICK_ALIASES.get("browser") == "Safari")
    record("agents", "spotify → Spotify",   _QUICK_ALIASES.get("spotify") == "Spotify")
    record("agents", "code → VS Code",      "Visual Studio Code" in _QUICK_ALIASES.get("code",""))
    record("agents", "30+ aliases defined", len(_QUICK_ALIASES) >= 30)

    # URL quick table
    record("agents", "youtube in quick URLs",  "youtube" in _QUICK_URLS)
    record("agents", "github in quick URLs",   "github" in _QUICK_URLS)

    # Nav triggers
    for phrase in ["take me there", "open that", "go there", "show me"]:
        record("agents", f"nav trigger: '{phrase}'", bool(NAV_TRIGGERS.search(phrase)))

    # Browser triggers
    for phrase in ["search for coffee on google", "click the login button",
                   "scroll down", "read me the page"]:
        record("agents", f"browser trigger: '{phrase[:30]}'",
               bool(BROWSER_TRIGGERS.search(phrase)))

    # URL normalise
    record("agents", "normalise adds https", _normalise("example.com").startswith("https://"))
    record("agents", "normalise keeps https", _normalise("https://x.com") == "https://x.com")

test_agents()


# ── 5. System dispatch routing ─────────────────────────────────────────────────
print("\n[5] System Dispatch Routing")

async def test_dispatch():
    from jarvis.tools.system import SystemTools, get_system_stats

    st = SystemTools(CFG)

    # Stats should return immediately
    r = await st.dispatch("show me system stats")
    record("dispatch", "stats command routes", r is not None and "CPU" in (r or ""))

    # Sleep/shutdown/restart patterns
    for phrase, label in [
        ("list top processes", "list processes"),
    ]:
        r = await st.dispatch(phrase)
        record("dispatch", label, r is not None)

    # Should NOT route (needs LLM)
    r = await st.dispatch("tell me about coffee")
    record("dispatch", "factual → returns None (goes to LLM)", r is None)

    r = await st.dispatch("what time is it")
    record("dispatch", "time query → returns None (goes to LLM)", r is None)

asyncio.run(test_dispatch())


# ── 6. Critique agent ──────────────────────────────────────────────────────────
print("\n[6] Critique Agent  (calls Ollama — takes ~1 min)")

async def test_critique():
    from jarvis.agents.critique_agent import critique
    from jarvis.pipeline.llm import LLMCore
    llm = LLMCore(CFG)

    # Good response — should score high
    good_q = "What is the best way to brew coffee?"
    good_r = "Pour-over brewing gives the cleanest cup, sir. Use 1:15 coffee-to-water ratio, 94°C water, and medium-coarse grind. Takes about 3 minutes and produces exceptional clarity."
    r = await critique(good_q, good_r, llm)
    record("critique", "good response scores ≥ 7", r["score"] >= 7,
           f"score={r['score']}")

    # Bad response — should score low (meta-commentary + website deflection)
    bad_q = "How is coffee made?"
    bad_r = "Not quite what we were discussing, but I'd recommend checking out Coffee Review at coffeereview.com for expert guidance. (Note: I've added some personality to this response.)"
    r = await critique(bad_q, bad_r, llm)
    record("critique", "bad response scores < 7", r["score"] < 7,
           f"score={r['score']}, issues={r['issues'][:2]}")

asyncio.run(test_critique())


# ── 7. Guidelines cache ────────────────────────────────────────────────────────
print("\n[7] Guidelines Cache")

def test_guidelines():
    from jarvis.agents.critique_agent import load_refined_guidelines, _GUIDELINES_FILE
    import time

    # First call (cold)
    t0 = time.perf_counter()
    g1 = load_refined_guidelines()
    t1 = time.perf_counter() - t0

    # Second call (warm cache — should be instant)
    t0 = time.perf_counter()
    g2 = load_refined_guidelines()
    t2 = time.perf_counter() - t0

    record("guidelines", "returns string (empty or with lessons)", isinstance(g1, str))
    record("guidelines", "second call faster (cached)", t2 < t1 + 0.01,
           f"cold={t1*1000:.1f}ms warm={t2*1000:.1f}ms")
    if _GUIDELINES_FILE.exists():
        record("guidelines", "guidelines file exists after training", True,
               f"{_GUIDELINES_FILE}")
    else:
        record("guidelines", "no guidelines yet (training not done)", True,
               "run scripts/train_loop.py to populate")

test_guidelines()


# ── 8. Tool call regex ─────────────────────────────────────────────────────────
print("\n[8] Tool Call Interception (regex)")

def test_tool_regex():
    import re
    from jarvis.pipeline.llm import _TOOL_RE, _NARRATION_RE

    # Flat format should match
    flat_cases = [
        '{"tool": "open_app", "name": "Spotify"}',
        '{"tool": "system_sleep"}',
        '{"tool": "get_stats"}',
        '{"tool": "open_app", "name": "Visual Studio Code"}',
    ]
    for tc in flat_cases:
        record("tool_regex", f"matches flat: {tc[:40]}", bool(_TOOL_RE.search(tc)))

    # Nested format should NOT match (prevents leaking as text)
    nested = '{"tool": "open_url", "args": {"url": "https://x.com"}}'
    record("tool_regex", "rejects nested (old format)", not bool(_TOOL_RE.search(nested)))

    # Narration filter
    narration_cases = [
        "Opening https://example.com.",
        "Navigating to https://google.com",
        "Loading https://test.com now",
    ]
    for nc in narration_cases:
        cleaned = _NARRATION_RE.sub("", nc)
        record("tool_regex", f"strips narration: '{nc[:30]}'", cleaned.strip() == "")

test_tool_regex()


# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("RESULTS SUMMARY")
print(f"{'='*60}")

sections = {}
for sec, name, ok, detail in results:
    sections.setdefault(sec, []).append(ok)

total_pass = total_fail = 0
for sec, okays in sections.items():
    p = sum(okays); f = len(okays) - p
    total_pass += p; total_fail += f
    bar = f"{p}/{len(okays)}"
    status = PASS if f == 0 else (WARN if p > 0 else FAIL)
    print(f"  {status}  {sec:<20} {bar}")

print(f"\n  Total: {total_pass} passed, {total_fail} failed out of {total_pass+total_fail} tests")
score_pct = int(100 * total_pass / (total_pass + total_fail)) if (total_pass+total_fail) else 0
print(f"  Score: {score_pct}%")
print(f"{'='*60}\n")
