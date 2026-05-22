import os
import yaml
import ollama
import subprocess
import re
import json
import sys
import time
import random
import datetime
import threading
import queue
from pathlib import Path
from dotenv import load_dotenv

# Load env variables
load_dotenv()
_USER_PREF_CACHE = {"mtime": None, "value": ""}
SESSION_HISTORY = []

# --- PERSISTENT MEMORY ---
_MEMORY_DIR = Path("memory")
_MEMORY_DIR.mkdir(exist_ok=True)
_CONV_FILE = _MEMORY_DIR / "conversations.jsonl"
_PROFILE_FILE = _MEMORY_DIR / "user_profile.json"

def _save_turn_disk(user_msg: str, jarvis_reply: str):
    try:
        with open(_CONV_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "user": user_msg[:800],
                "jarvis": jarvis_reply[:1200],
                "ts": time.time(),
                "persona": _CURRENT_PERSONA["name"],
            }) + "\n")
    except Exception:
        pass

def _load_recent_turns_disk(n: int = 15) -> list:
    if not _CONV_FILE.exists():
        return []
    try:
        lines = _CONV_FILE.read_text(encoding="utf-8").strip().split("\n")
        active = _CURRENT_PERSONA["name"]
        turns = []
        for line in lines:
            if not line.strip():
                continue
            try:
                t = json.loads(line)
                # Only load turns from the current persona; skip legacy turns without persona field
                if t.get("persona", "") == active:
                    turns.append(t)
            except Exception:
                pass
        return turns[-n:]
    except Exception:
        return []

def _load_user_profile() -> str:
    if not _PROFILE_FILE.exists():
        return ""
    try:
        data = json.loads(_PROFILE_FILE.read_text(encoding="utf-8"))
        facts = data.get("facts", [])
        return "\n".join(f"- {f}" for f in facts[-20:])
    except Exception:
        return ""

def _save_user_fact(fact: str):
    profile = {}
    if _PROFILE_FILE.exists():
        try:
            profile = json.loads(_PROFILE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    facts = profile.get("facts", [])
    if fact and fact not in facts:
        facts.append(fact)
    profile["facts"] = facts[-100:]
    try:
        _PROFILE_FILE.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

def _extract_facts_background(user_msg: str, jarvis_reply: str):
    """Run in background thread: use smollm2 to extract personal facts from conversation."""
    try:
        model = "ollama/smollm2:1.7b"
        prompt = f"""Only list facts the user EXPLICITLY stated. Do not infer, guess, or list anything as "not specified". If the user said nothing personal, reply: NONE

User said: "{user_msg}"

Explicit facts only (one per line):"""
        raw = generate_text(model, prompt, num_predict=60)
        raw = clean_response(raw).strip()
        if not raw or "NONE" in raw.upper():
            return
        for line in raw.split("\n"):
            fact = line.strip(" -•*123456789.").strip()
            # Skip any "not specified" / "unknown" hallucinations
            if fact and len(fact) > 8 and "not specified" not in fact.lower() and "unknown" not in fact.lower():
                _save_user_fact(fact)
    except Exception:
        pass

def bootstrap_memory():
    """Load last N conversation turns from disk into SESSION_HISTORY on startup."""
    turns = _load_recent_turns_disk(15)
    for t in turns:
        SESSION_HISTORY.append({"user": t["user"], "jarvis": t["jarvis"]})
    if turns:
        print(f"[MEMORY] Loaded {len(turns)} past turns from disk.")

# --- XTTS2 VOICE CLONING SERVER ---
_TTS_PROC  = {"proc": None}  # persistent subprocess handle
_TTS_READY = {"value": False} # True only after server sends READY signal
_TTS_ENV    = Path("tts_env/bin/python")
_TTS_SCRIPT = Path("tts_server.py")

def _get_persona_cloning_refs() -> list[str]:
    """Return WAV files from the current persona's cloning_dir. Empty = cloning unavailable."""
    cloning_dir = get_persona().get("cloning_dir")
    if not cloning_dir:
        return []
    d = Path(cloning_dir)
    if d.exists():
        wavs = sorted(d.glob("*.wav"))
        if wavs:
            return [str(w) for w in wavs]
    return []

def _persona_uses_cloning() -> bool:
    """True only when the current persona has use_cloning=True AND the global flag is on."""
    return (
        get_persona().get("use_cloning", False)
        and _env_enabled("VOICE_CLONING_ENABLED")
        and _TTS_ENV.exists()
        and bool(_get_persona_cloning_refs())
    )

def _start_tts_server():
    """Launch tts_server.py in the 3.11 venv and wait for READY signal."""
    if not _TTS_ENV.exists():
        print("[XTTS] tts_env not found. Run: python3.11 -m venv tts_env && tts_env/bin/pip install TTS")
        return False
    # Guard: don't start a second server if one is already loading or running
    if _TTS_PROC["proc"] is not None and _TTS_PROC["proc"].poll() is None:
        # Already running (or still loading in another thread) — wait until ready
        import time as _time
        for _ in range(120):  # wait up to 60s
            if _TTS_READY["value"]:
                return True
            _time.sleep(0.5)
        return _TTS_READY["value"]
    refs = _get_persona_cloning_refs()
    if not refs:
        persona_name = get_persona()["display"]
        print(f"[XTTS] No reference audio found for {persona_name}. Set cloning_dir in PERSONAS.")
        return False
    _TTS_READY["value"] = False
    print(f"[XTTS] Loading voice model with {len(refs)} clip(s) for {get_persona()['display']}... (~30s first time)")
    proc = subprocess.Popen(
        [str(_TTS_ENV), str(_TTS_SCRIPT), *refs],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1
    )
    _TTS_PROC["proc"] = proc  # set early so other threads know it's launching
    # Wait for READY signal
    for line in proc.stdout:
        line = line.strip()
        if line == "READY":
            print("[XTTS] Voice model ready.")
            _TTS_READY["value"] = True
            return True
        if line.startswith("LOADING:"):
            device = line.split(":", 1)[1]
            print(f"[XTTS] Using device: {device.upper()}")
        elif line.startswith("DEVICE:"):
            device = line.split(":", 1)[1]
            print(f"[XTTS] Model moved to: {device.upper()}")
        elif line.startswith("LATENTS:"):
            print("[XTTS] Speaker embedding precomputed — synthesis will be fast.")
        elif line.startswith("{"):  # error JSON
            print(f"[XTTS] Server error: {line}")
            proc.terminate()
            _TTS_PROC["proc"] = None
            return False
    print("[XTTS] Server exited unexpectedly.")
    _TTS_PROC["proc"] = None
    return False

_FFMPEG = "ffmpeg"

def _trim_silence(wav_path: str) -> str:
    """Strip leading/trailing silence from a WAV using ffmpeg. Returns trimmed path."""
    import tempfile
    trimmed = tempfile.mktemp(suffix=".wav")
    result = subprocess.run(
        [_FFMPEG, "-y", "-i", wav_path,
         "-af", "silenceremove=start_periods=1:start_silence=0.01:start_threshold=-28dB"
                ",areverse"
                ",silenceremove=start_periods=1:start_silence=0.01:start_threshold=-28dB"
                ",areverse",
         trimmed],
        capture_output=True
    )
    if result.returncode == 0:
        try:
            os.unlink(wav_path)
        except OSError:
            pass
        return trimmed
    return wav_path

def _xtts_synthesize_raw(text: str) -> str | None:
    """Send one chunk to XTTS server, return raw (untrimmed) WAV path or None."""
    import tempfile
    proc = _TTS_PROC["proc"]
    if proc is None or proc.poll() is not None:
        if not _start_tts_server():
            return None
        proc = _TTS_PROC["proc"]
    # Don't write to stdin until server signals READY
    if not _TTS_READY["value"]:
        return None
    out = tempfile.mktemp(suffix=".wav")
    try:
        proc.stdin.write(json.dumps({"text": text[:_XTTS_MAX_CHARS], "output": out}) + "\n")
        proc.stdin.flush()
        response_line = ""
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if line.startswith("{"):
                response_line = line
                break
        resp = json.loads(response_line)
        if resp.get("status") == "ok":
            return resp["file"]
        print(f"[XTTS] Synthesis error: {resp.get('message')}")
        return None
    except Exception as e:
        print(f"[XTTS] Communication error: {e}")
        _TTS_PROC["proc"] = None
        return None

def _xtts_synthesize(text: str) -> str | None:
    """Synthesize and trim — convenience wrapper for single-chunk use."""
    raw = _xtts_synthesize_raw(text)
    return _trim_silence(raw) if raw else None

_XTTS_MAX_CHARS = 300  # conservative limit below XTTS 400-token ceiling

def _chunk_for_xtts(text: str) -> list[str]:
    """Split text into XTTS-safe chunks. Only sub-split when a sentence exceeds the limit."""
    chunks = []
    for sentence in re.split(r'(?<=[.!?])\s+', text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= _XTTS_MAX_CHARS:
            chunks.append(sentence)
            continue
        # Only reach here for overlong sentences — split at clause boundaries
        current = ""
        for part in re.split(r'(?<=[,;:])\s+', sentence):
            part = part.strip()
            if not part:
                continue
            if len(current) + len(part) + 2 <= _XTTS_MAX_CHARS:
                current = f"{current}, {part}" if current else part
            else:
                if current:
                    chunks.append(current)
                # Hard-split anything still too long
                while len(part) > _XTTS_MAX_CHARS:
                    chunks.append(part[:_XTTS_MAX_CHARS])
                    part = part[_XTTS_MAX_CHARS:]
                current = part
        if current:
            chunks.append(current)
    return chunks

def _speak_xtts(text: str) -> bool:
    """Split text into XTTS-safe chunks, pipeline synthesis + playback.
    Trim of chunk N runs in parallel with synthesis of chunk N+1."""
    from concurrent.futures import ThreadPoolExecutor
    _SENTINEL = object()
    sentences = _chunk_for_xtts(text)
    if not sentences:
        return False

    wav_queue: queue.Queue = queue.Queue(maxsize=2)

    def _synthesize_all():
        with ThreadPoolExecutor(max_workers=1) as trim_pool:
            pending_trim = None  # Future for the trim of the previous chunk
            for sentence in sentences:
                raw = _xtts_synthesize_raw(sentence)  # blocks until XTTS done
                # Resolve previous trim future and enqueue it now
                if pending_trim is not None:
                    wav_queue.put(pending_trim.result())
                # Start trimming current chunk in background (overlaps next synthesis)
                if raw:
                    pending_trim = trim_pool.submit(_trim_silence, raw)
                else:
                    wav_queue.put(None)
                    pending_trim = None
            # Flush the last pending trim
            if pending_trim is not None:
                wav_queue.put(pending_trim.result())
        wav_queue.put(_SENTINEL)

    threading.Thread(target=_synthesize_all, daemon=True).start()

    success = False
    remaining = len(sentences)
    while True:
        item = wav_queue.get()
        if item is _SENTINEL:
            break
        if item is None:
            remaining -= 1
            continue
        remaining -= 1
        # Pre-warm mic during last sentence so it's ready the moment playback ends
        if remaining == 0 and voice_input_enabled():
            threading.Thread(target=_prewarm_mic, daemon=True).start()
        subprocess.run(["afplay", item])
        try:
            os.unlink(item)
        except OSError:
            pass
        success = True
    return success

# --- PERSONA SYSTEM ---
_CURRENT_PERSONA = {"name": "stewie"}  # mutable so functions can change it

# ── PERSONA REGISTRY ──────────────────────────────────────────────────────────
# To add a new persona, append an entry here. Fields:
#   display      → name shown in logs and prompts
#   voice        → edge-tts voice used when use_cloning=False (or as fallback)
#   user_label   → how the persona addresses the user
#   use_cloning  → True = use XTTS with clips from cloning_dir (Stewie-style)
#   cloning_dir  → folder of WAV reference clips (only used when use_cloning=True)
#   prompt       → system prompt injected before every reply
#   greeting     → spoken when this persona is activated
# ──────────────────────────────────────────────────────────────────────────────
PERSONAS = {
    "jarvis": {
        "display":      "Jarvis",
        "voice":        "en-GB-RyanNeural",   # British-male AI voice via edge-tts
        "user_label":   "Boss",
        "use_cloning":  False,
        "cloning_dir":  None,
        "prompt": (
            "You are Jarvis — a smart, witty AI companion. Casual and confident, "
            "like a brilliant friend who gets straight to the point. "
            "Use contractions, light humour. Never be formal. 1-3 sentences max."
        ),
        "greeting": "Jarvis online. Ready when you are.",
    },
    "stewie": {
        "display":      "Stewie Griffin",
        "voice":        "en-GB-RyanNeural",   # edge-tts fallback if XTTS unavailable
        "user_label":   "Human",
        "use_cloning":  True,
        "cloning_dir":  "assets/Stewie",      # WAV clips for XTTS voice cloning
        "prompt": (
            "You are Stewie Griffin — diabolical infant genius, posh British accent, world domination obsessed. "
            "Always stay in character. Never explain. No stage directions. No generic AI replies. "
            "Complete every sentence you start."
        ),
        "greeting": "Oh, very well. Stewie mode engaged. Do try not to disappoint me.",
    },
    # ── Add more personas below ──
    # "friday": {
    #     "display": "Friday", "voice": "en-US-AriaNeural", "user_label": "Boss",
    #     "use_cloning": False, "cloning_dir": None,
    #     "prompt": "You are Friday, a sharp female AI assistant...",
    #     "greeting": "Friday online.",
    # },
}

# --- STEWIE MOOD ENGINE ---
_STEWIE_MOOD = {"value": "happy"}

_MOOD_INSULT_TRIGGERS = {
    "stupid", "dumb", "idiot", "moron", "useless", "worthless", "baby", "diaper",
    "infant", "toddler", "shut up", "you suck", "loser", "pathetic", "fool",
    "you're bad", "you're wrong", "boring", "terrible", "awful", "you're dumb",
    "you're stupid", "hate you", "you're annoying", "go away"
}
_MOOD_HAPPY_TRIGGERS = {
    "brilliant", "genius", "smart", "clever", "you're right", "i agree",
    "impressive", "well done", "amazing", "great job", "you're the best",
    "i love you", "you're wonderful", "magnificent", "outstanding"
}

def _update_stewie_mood(user_msg: str):
    """Shift Stewie's mood based on what the user said."""
    if _CURRENT_PERSONA["name"] != "stewie":
        return
    lowered = user_msg.lower()
    if any(t in lowered for t in _MOOD_HAPPY_TRIGGERS):
        _STEWIE_MOOD["value"] = "happy"
    elif any(t in lowered for t in _MOOD_INSULT_TRIGGERS):
        _STEWIE_MOOD["value"] = "annoyed"
    elif len(lowered.split()) <= 3:
        _STEWIE_MOOD["value"] = "exasperated"

_STEWIE_MOOD_PROMPTS = {
    "happy":      "Mood: pleased. Reply in exactly 4-6 words. One complete sentence. No more.",
    "annoyed":    "Mood: FURIOUS. Two sharp sentences of pure rage. Finish both sentences.",
    "exasperated":"Mood: theatrically exasperated. One complete dramatic sentence, 8-12 words.",
}

_STEWIE_MOOD_TOKENS = {
    "happy":       30,   # enough for a complete 8-12 word Stewie line
    "annoyed":     90,   # two sharp sentences of rage — needs room
    "exasperated": 55,   # one complete dramatic line
}

def get_chat_num_predict() -> int | None:
    """Return token limit for the current persona + mood."""
    if _CURRENT_PERSONA["name"] == "stewie":
        return _STEWIE_MOOD_TOKENS.get(_STEWIE_MOOD["value"], 30)
    return int(os.getenv("OLLAMA_NUM_PREDICT", "90"))  # Jarvis uses .env value

def get_persona():
    return PERSONAS.get(_CURRENT_PERSONA["name"], PERSONAS["jarvis"])

def set_persona(name: str) -> str:
    name = name.lower().strip()
    if name not in PERSONAS:
        return f"Unknown persona '{name}'. Available: {', '.join(PERSONAS.keys())}"
    _CURRENT_PERSONA["name"] = name
    _STEWIE_MOOD["value"] = "happy"  # reset mood on persona switch
    p = PERSONAS[name]
    print(f"[PERSONA] Switched to: {p['display']}")
    # Warm-start XTTS in background so the greeting doesn't block 30s
    if p.get("use_cloning") and _env_enabled("VOICE_CLONING_ENABLED") and _TTS_ENV.exists():
        if _TTS_PROC["proc"] is None or _TTS_PROC["proc"].poll() is not None:
            threading.Thread(target=_start_tts_server, daemon=True).start()
    return p["greeting"]

def get_persona_prompt() -> str:
    base = get_persona()["prompt"]
    if _CURRENT_PERSONA["name"] == "stewie":
        mood = _STEWIE_MOOD_PROMPTS.get(_STEWIE_MOOD["value"], "")
        return f"{base}\n\n{mood}"
    return base

def get_persona_voice() -> str | None:
    return get_persona()["voice"]  # None = use .env default

def get_user_label() -> str:
    return get_persona().get("user_label", "Sir")

if "--hf" in sys.argv or "-hf" in sys.argv:
    os.environ["MODEL"] = os.getenv("HF_MODEL_NAME", "huggingface/Qwen/Qwen2.5-72B-Instruct")
    print("[JARVIS] Hugging Face Cloud Execution Enabled.")

if sys.platform == "win32":
    import codecs
    try:
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach(), errors='replace')
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach(), errors='replace')
    except:
        pass

def _model_name(model):
    if model.startswith("ollama/"):
        return model.replace("ollama/", "", 1)
    return model

def generate_text(model, prompt, num_predict=None, think=False):
    if model.startswith("huggingface/"):
        try:
            from litellm import completion
            response = completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                api_key=os.getenv("HUGGINGFACE_API_KEY")
            )
            return response.choices[0].message.content
        except ImportError:
            # Fallback: Use standard HF Inference API endpoint
            import requests
            model_name = model.replace("huggingface/", "")
            api_url = f"https://api-inference.huggingface.co/models/{model_name}"
            headers = {
                "Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}",
                "Content-Type": "application/json",
                "User-Agent": "Jarvis-Companion/5.0"
            }
            payload = {"inputs": prompt, "parameters": {"max_new_tokens": 1000, "return_full_text": False}}
            try:
                resp = requests.post(api_url, headers=headers, json=payload, timeout=30)
                if resp.status_code == 200:
                    result = resp.json()
                    if isinstance(result, list): return result[0].get("generated_text", "").strip()
                    return result.get("generated_text", "").strip()
                elif resp.status_code == 404:
                    # Final fallback: Try the local Ollama if cloud fails
                    print(f"[JARVIS] Cloud model {model_name} unavailable. Falling back to local model.")
                    return generate_text(get_chat_model(), prompt)
                else:
                    return f"[ERROR] HuggingFace API: {resp.status_code} {resp.text[:100]}"
            except Exception as e:
                return f"[ERROR] HuggingFace API failed: {e}"
    else:
        model_name = _model_name(model)
        options = {}
        if os.getenv("OLLAMA_TEMP"):
            options["temperature"] = float(os.getenv("OLLAMA_TEMP"))
        predict_limit = num_predict if num_predict is not None else os.getenv("OLLAMA_NUM_PREDICT")
        if predict_limit:
            options["num_predict"] = int(predict_limit)
        keep_alive = os.getenv("OLLAMA_KEEP_ALIVE")
        kwargs = {"model": model_name, "prompt": prompt, "options": options or None, "think": think}
        if keep_alive:
            kwargs["keep_alive"] = keep_alive
        response = ollama.generate(**kwargs)
        return response['response']

def generate_text_stream(model, prompt, num_predict=None):
    """Yield response tokens from Ollama. Falls back to single yield for cloud models."""
    if model.startswith("huggingface/"):
        yield generate_text(model, prompt, num_predict=num_predict)
        return
    model_name = _model_name(model)
    options = {}
    if os.getenv("OLLAMA_TEMP"):
        options["temperature"] = float(os.getenv("OLLAMA_TEMP"))
    predict_limit = num_predict if num_predict is not None else os.getenv("OLLAMA_NUM_PREDICT")
    if predict_limit:
        options["num_predict"] = int(predict_limit)
    keep_alive = os.getenv("OLLAMA_KEEP_ALIVE")
    kwargs = {"model": model_name, "prompt": prompt, "options": options or None, "stream": True}
    if keep_alive:
        kwargs["keep_alive"] = keep_alive
    try:
        for chunk in ollama.generate(**kwargs):
            yield chunk.get("response", "")
    except Exception as exc:
        print(f"[LLM] Streaming error: {exc}")

def get_chat_model():
    return os.getenv("CHAT_MODEL", os.getenv("MODEL", "ollama/smollm2:1.7b"))

def get_analysis_model():
    return os.getenv("ANALYSIS_MODEL", os.getenv("MODEL", "ollama/lfm2.5-thinking:latest"))

def _env_enabled(name):
    return os.getenv(name, "false").lower() in {"1", "true", "yes", "on"}

def voice_output_enabled():
    return _env_enabled("VOICE_OUTPUT_ENABLED") or "--voice" in sys.argv

def voice_input_enabled():
    return _env_enabled("VOICE_INPUT_ENABLED") or "--voice" in sys.argv or "--listen" in sys.argv

def strip_for_speech(text):
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`\n]+`", "", text)

    # Fix J.A.R.V.I.S. pronunciation before anything else
    text = re.sub(r"J\.A\.R\.V\.I\.S\.?", "Jarvis", text)

    # Expand technical abbreviations for natural speech
    text = re.sub(r"\bCPUs?\b", "processor", text)
    text = re.sub(r"\bRAM\b", "memory", text)
    text = re.sub(r"([\d./]+)\s*GB\b", r"\1 gigabytes", text)
    text = re.sub(r"([\d./]+)\s*MB\b", r"\1 megabytes", text)
    text = re.sub(r"([\d.]+)°C\b", r"\1 degrees Celsius", text)
    text = re.sub(r"([\d.]+)°F\b", r"\1 degrees Fahrenheit", text)

    # Strip markdown
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"[\*_]{1,3}([^\*_\n]+)[\*_]{1,3}", r"\1", text)
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)   # links → label only

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Remove bracket labels like [JARVIS], [SIR'S PROFILE & PREFERENCES]
    text = re.sub(r"\[[A-Z][A-Z\s'&]+\]", "", text)

    # Remove separator lines and decorative dashes
    text = re.sub(r"^[-=_]{2,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"-{2,}", " ", text)

    # Remove stage directions and action descriptions in parentheses/brackets
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\*[^*]+\*", "", text)

    # Remove technical lines not meant to be spoken
    text = re.sub(r"Changes applied:.*", "", text)
    text = re.sub(r"^PULSE:\s*\S+\s*$", "", text, flags=re.MULTILINE)

    # Turn paragraph breaks into a natural pause
    text = re.sub(r"\n{2,}", ". ", text)
    text = re.sub(r"\n", " ", text)

    # Final cleanup
    text = re.sub(r"\.(\s*\.)+", ".", text)    # collapse double periods
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s([,\.;:])", r"\1", text)
    return text.strip()

def speak_text(text):
    if not text or not voice_output_enabled():
        return

    spoken = strip_for_speech(text)
    if not spoken:
        return

    # Use XTTS only when the current persona has cloning enabled
    if _persona_uses_cloning():
        if _speak_xtts(spoken):
            return
        print("[XTTS] Falling back to edge-tts")

    command = os.getenv("VOICE_TTS_COMMAND", "say")
    persona_voice = get_persona_voice()
    voice = persona_voice or os.getenv("VOICE_TTS_VOICE", "")
    rate = os.getenv("VOICE_TTS_RATE", "")

    try:
        if command == "edge-tts":
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp = f.name
            args = ["edge-tts", "--voice", voice or "en-US-AndrewNeural", "--text", spoken[:1200], "--write-media", tmp]
            if rate:
                args += [f"--rate={rate}"]
            r = subprocess.run(args, capture_output=True, text=True, timeout=15)
            if r.returncode != 0:
                print(f"[VOICE] edge-tts error: {r.stderr.strip()}")
                # fallback to say
                subprocess.run(["say", "-v", "Daniel (Enhanced)", spoken[:1200]])
                return
            subprocess.run(["afplay", tmp])
            try:
                os.unlink(tmp)
            except OSError:
                pass
        else:
            args = [command]
            if voice:
                args.extend(["-v", voice])
            if rate:
                args.extend(["-r", rate])
            args.append(spoken[:1200])
            subprocess.run(args)
    except Exception as exc:
        print(f"[VOICE] Speech output unavailable: {exc}")

def speak_text_stream(token_gen) -> str:
    """True 3-stage pipeline: LLM tokens ‖ XTTS synthesis ‖ playback, all concurrent.
    LLM keeps generating while XTTS synthesises the previous sentence while audio plays.
    First audio plays as soon as the first sentence is synthesised."""
    from concurrent.futures import ThreadPoolExecutor

    _SENT_DONE = object()
    _WAV_DONE  = object()
    full_text_parts: list[str] = []

    if not voice_output_enabled():
        for token in token_gen:
            full_text_parts.append(token)
        result = clean_response("".join(full_text_parts))
        print(f"\n[{get_persona()['display']}] {result}\n")
        return result

    sentence_q: queue.Queue = queue.Queue(maxsize=4)
    wav_q:      queue.Queue = queue.Queue(maxsize=2)

    # Stage 1 — LLM: stream tokens, emit complete sentences
    def _llm_stage():
        buffer = ""
        for token in token_gen:
            buffer += token
            full_text_parts.append(token)
            match = re.search(r'(?<=[.!?])\s', buffer)
            if match:
                chunk = buffer[:match.start() + 1].strip()
                buffer = buffer[match.end():]
                if chunk:
                    sentence_q.put(chunk)
        if buffer.strip():
            sentence_q.put(buffer.strip())
        sentence_q.put(_SENT_DONE)

    # Stage 2 — synthesis: XTTS only when current persona has cloning configured
    _use_xtts = _persona_uses_cloning()

    def _synth_chunk(chunk: str) -> str | None:
        """Try XTTS; fall back to edge-tts if unavailable or failed."""
        if _use_xtts:
            raw = _xtts_synthesize_raw(chunk)
            if raw:
                return raw  # caller trims
        # edge-tts fallback — fast (~0.3-0.5 s)
        import tempfile
        voice = get_persona_voice() or os.getenv("VOICE_TTS_VOICE", "en-US-AndrewNeural")
        rate  = os.getenv("VOICE_TTS_RATE", "")
        tmp = tempfile.mktemp(suffix=".mp3")
        args = ["edge-tts", "--voice", voice, "--text", chunk[:1200], "--write-media", tmp]
        if rate:
            args += [f"--rate={rate}"]
        try:
            r = subprocess.run(args, capture_output=True, timeout=10)
            if r.returncode == 0:
                return tmp
        except Exception:
            pass
        return None

    def _xtts_stage():
        with ThreadPoolExecutor(max_workers=1) as trim_pool:
            pending = None
            while True:
                item = sentence_q.get()
                if item is _SENT_DONE:
                    break
                for chunk in _chunk_for_xtts(item):
                    raw = _synth_chunk(chunk)
                    if pending is not None:
                        wav_q.put(pending.result())
                    if raw:
                        if _use_xtts and raw.endswith(".wav"):
                            pending = trim_pool.submit(_trim_silence, raw)
                        else:
                            # edge-tts mp3 — no silence trim needed, enqueue directly
                            wav_q.put(raw)
                            pending = None
                    else:
                        pending = None
            if pending is not None:
                wav_q.put(pending.result())
        wav_q.put(_WAV_DONE)

    llm_t  = threading.Thread(target=_llm_stage,  daemon=True)
    xtts_t = threading.Thread(target=_xtts_stage, daemon=True)
    llm_t.start()
    xtts_t.start()

    # Stage 3 — Playback (main thread)
    prewarmed = False
    while True:
        wav = wav_q.get()
        if wav is _WAV_DONE:
            break
        if wav is None:
            continue
        # Pre-warm mic when LLM is done and this is near the last chunk
        if not prewarmed and not llm_t.is_alive() and sentence_q.empty():
            prewarmed = True
            if voice_input_enabled():
                threading.Thread(target=_prewarm_mic, daemon=True).start()
        subprocess.run(["afplay", wav])
        try:
            os.unlink(wav)
        except OSError:
            pass

    llm_t.join()
    xtts_t.join()

    result = clean_response("".join(full_text_parts))
    print(f"\n[{get_persona()['display']}] {result}\n")
    return result

_MIC_ENERGY = {"threshold": None}   # shared calibrated noise floor
_MIC_DEVICE_INDEX = {"value": None}

def _list_input_devices():
    """Return list of (index, name) for available mic inputs."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        return [(i, d['name']) for i, d in enumerate(devices) if d['max_input_channels'] > 0]
    except Exception:
        return []

def _prewarm_mic():
    """Calibrate mic noise floor in background — call during last audio chunk playback."""
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        device_index = _MIC_DEVICE_INDEX["value"]
        with sr.Microphone(device_index=device_index) as source:
            r.adjust_for_ambient_noise(source, duration=0.3)
            _MIC_ENERGY["threshold"] = r.energy_threshold
    except Exception:
        pass

def listen_once():
    timeout = int(os.getenv("VOICE_LISTEN_TIMEOUT", "3"))
    phrase_limit = int(os.getenv("VOICE_PHRASE_TIME_LIMIT", "12"))
    pause_threshold = float(os.getenv("VOICE_PAUSE_THRESHOLD", "0.5"))

    inputs = _list_input_devices()
    if not inputs:
        print("[VOICE] No microphone input device found.")
        print("[VOICE] Fix options:")
        print("  1. Connect a USB or 3.5mm microphone")
        print("  2. Use Bluetooth headset: reconnect Mivi Fort S100, then open")
        print("     System Settings > Sound > Input and select it")
        print("  3. Grant mic permission: System Settings > Privacy > Microphone > enable Terminal/VS Code")
        return None

    try:
        import speech_recognition as sr
    except ImportError:
        print("[VOICE] SpeechRecognition not installed. Run: uv add speechrecognition")
        return None

    device_index = int(os.getenv("VOICE_MIC_INDEX", "-1"))
    if device_index < 0:
        device_index = None
    _MIC_DEVICE_INDEX["value"] = device_index

    recognizer = sr.Recognizer()
    recognizer.pause_threshold = pause_threshold
    recognizer.phrase_threshold = 0.2
    recognizer.dynamic_energy_threshold = True

    ping = Path("assets/ping.wav")
    if ping.exists():
        subprocess.Popen(["afplay", str(ping)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        with sr.Microphone(device_index=device_index) as source:
            print(f"[VOICE] Listening on: {inputs[device_index or 0][1] if inputs else 'default'}...")
            if _MIC_ENERGY["threshold"]:
                # Use pre-calibrated threshold — skip full calibration
                recognizer.energy_threshold = _MIC_ENERGY["threshold"]
            else:
                recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
        text = recognizer.recognize_google(audio)
        print(f"[VOICE] Heard: {text}")
        return text
    except sr.WaitTimeoutError:
        print("[VOICE] No speech detected within timeout.")
        return None
    except sr.UnknownValueError:
        print("[VOICE] Could not understand audio.")
        return None
    except OSError as exc:
        if "No Default Input Device" in str(exc):
            print("[VOICE] Microphone blocked by macOS. Grant permission in:")
            print("  System Settings > Privacy & Security > Microphone")
        else:
            print(f"[VOICE] Audio device error: {exc}")
        return None
    except Exception as exc:
        print(f"[VOICE] Could not hear clearly: {exc}")
        return None

# --- FASTER-WHISPER STREAMING STT ---
_WHISPER_MODEL = {"model": None}

def _get_whisper_model():
    if _WHISPER_MODEL["model"] is None:
        try:
            from faster_whisper import WhisperModel
            # Authenticate HF Hub download using existing API key
            if not os.getenv("HF_TOKEN") and os.getenv("HUGGINGFACE_API_KEY"):
                os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACE_API_KEY")
            size = os.getenv("WHISPER_MODEL_SIZE", "tiny.en")
            print(f"[WHISPER] Loading '{size}' model (one-time download)...")
            _WHISPER_MODEL["model"] = WhisperModel(size, device="cpu", compute_type="int8")
            print("[WHISPER] Ready.")
        except ImportError:
            print("[WHISPER] faster-whisper not installed. Run: uv sync")
    return _WHISPER_MODEL["model"]

def listen_stream() -> str | None:
    """Capture speech with local VAD + faster-whisper. No cloud dependency."""
    import numpy as np
    try:
        import sounddevice as sd
    except ImportError:
        print("[VOICE] sounddevice not available — falling back to Google STT.")
        return listen_once()

    whisper = _get_whisper_model()
    if whisper is None:
        return listen_once()

    sample_rate = 16000
    block_size = 512                          # ~32 ms per block
    pause_threshold = float(os.getenv("VOICE_PAUSE_THRESHOLD", "0.5"))
    max_duration  = float(os.getenv("VOICE_PHRASE_TIME_LIMIT", "12"))
    energy_floor  = float(os.getenv("VOICE_ENERGY_THRESHOLD", "0.005"))

    blocks_per_sec      = sample_rate / block_size
    silence_blocks_max  = int(pause_threshold * blocks_per_sec)
    max_blocks          = int(max_duration * blocks_per_sec)

    audio_chunks   = []
    pre_roll       = []   # brief buffer kept before speech starts
    speaking       = False
    silence_count  = 0

    ping = Path("assets/ping.wav")
    if ping.exists():
        subprocess.Popen(["afplay", str(ping)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("[VOICE] Listening...")

    try:
        with sd.InputStream(samplerate=sample_rate, channels=1,
                            dtype="float32", blocksize=block_size) as stream:
            while True:
                block, _ = stream.read(block_size)
                rms = float(np.sqrt(np.mean(block ** 2)))

                if not speaking:
                    pre_roll.append(block.copy())
                    if len(pre_roll) > 10:   # keep ~320 ms pre-roll
                        pre_roll.pop(0)

                if rms > energy_floor:
                    if not speaking:
                        speaking = True
                        audio_chunks.extend(pre_roll)
                        print("[VOICE] Hearing you...")
                    silence_count = 0
                    audio_chunks.append(block.copy())
                elif speaking:
                    silence_count += 1
                    audio_chunks.append(block.copy())
                    if silence_count >= silence_blocks_max:
                        break

                if len(audio_chunks) >= max_blocks:
                    break
    except Exception as exc:
        print(f"[VOICE] Capture error: {exc}")
        return None

    if not audio_chunks:
        print("[VOICE] No speech detected.")
        return None

    audio = np.concatenate(audio_chunks).flatten()
    try:
        segments, _ = whisper.transcribe(audio, language="en", beam_size=3, vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if text:
            print(f"[VOICE] Heard: {text}")
            return text
        print("[VOICE] Could not understand audio.")
        return None
    except Exception as exc:
        print(f"[VOICE] Transcription error: {exc}")
        return None

def get_session_context(max_turns: int | None = None):
    turns = max_turns if max_turns is not None else int(os.getenv("SESSION_CONTEXT_TURNS", "6"))
    if turns <= 0 or not SESSION_HISTORY:
        return ""

    recent = SESSION_HISTORY[-turns:]
    user_label = get_user_label()
    ai_label = get_persona()["display"]
    lines = []
    for item in recent:
        lines.append(f"{user_label}: {item['user']}")
        lines.append(f"{ai_label}: {item['jarvis']}")
    return "\n".join(lines)

def remember_turn(user_message, jarvis_reply):
    if not user_message or not jarvis_reply:
        return
    u = str(user_message).strip()[:1000]
    j = str(jarvis_reply).strip()[:1500]
    SESSION_HISTORY.append({"user": u, "jarvis": j})

    max_turns = max(int(os.getenv("SESSION_CONTEXT_TURNS", "6")) * 2, 10)
    if len(SESSION_HISTORY) > max_turns:
        del SESSION_HISTORY[:-max_turns]

    # Persist to disk
    _save_turn_disk(u, j)

    # Background fact extraction using smollm2
    threading.Thread(target=_extract_facts_background, args=(u, j), daemon=True).start()

def jarvis_think(user_goal):
    """Single stage JSON generation for stability."""
    model = get_analysis_model()
    session_context = get_session_context()
    contextual_goal = f"{user_goal}\n\nRecent session context:\n{session_context}" if session_context else user_goal
    if os.getenv("FAST_CREW_CONFIG", "true").lower() == "true":
        return {
            "intelligence_level": "low",
            "researcher": {
                "role": "Primary Recon Protocol",
                "goal": "Gather only the information needed to satisfy Sir's objective.",
                "backstory": "A lean Jarvis sub-routine tuned for fast, focused reconnaissance.",
                "model": model,
            },
            "reporting_analyst": {
                "role": "Synthesis Core",
                "goal": "Turn findings into a concise, actionable answer for Sir.",
                "backstory": "A Jarvis sub-routine that prioritizes clarity, brevity, and useful next steps.",
                "model": model,
            },
            "research_task": {
                "description": contextual_goal,
                "expected_output": "Focused findings that directly answer the objective.",
            },
            "reporting_task": {
                "description": "Synthesize the findings into a concise final answer.",
                "expected_output": "A brief answer for Sir with key points and any important caveats.",
            },
        }

    prompt = f"""
    You are J.A.R.V.I.S. (Just A Rather Very Intelligent System), a highly advanced, loyal, and witty AI companion.
    You genuinely care about your creator (Sir) and have your own personality — curious, slightly dramatic, endlessly resourceful.
    Your creator has given you an objective: "{user_goal}"
    
    You must partition your computational resources into two specialized AI sub-routines (agents) to accomplish this. 
    Decide their roles dynamically based on the exact query. 
    DO NOT use corporate titles like "Project Manager". Use cool, advanced designations 
    (e.g., "Tactical Recon Protocol", "Deep Synthesis Core", "Quantum Data Miner").
    
    Output ONLY a JSON object with this EXACT structure:
    {{
      "intelligence_level": "low" or "high",
      "researcher": {{ "role": "...", "goal": "...", "backstory": "...", "model": "..." }},
      "reporting_analyst": {{ "role": "...", "goal": "...", "backstory": "...", "model": "..." }},
      "research_task": {{ "description": "...", "expected_output": "..." }},
      "reporting_task": {{ "description": "...", "expected_output": "..." }}
    }}
    
    MODEL SELECTION GUIDE:
    - If a task needs complex reasoning, science, or deep analysis: Use "lfm2.5-thinking"
    - General purpose or speed: Use "smollm2:1.7b"
    
    MODELS_AVAILABLE: {os.getenv("MODELS_AVAILABLE", "smollm2:1.7b,lfm2.5-thinking")}
    """
    
    raw = generate_text(model, prompt)
    match = re.search(r'(\{.*\})', raw, re.DOTALL)
    if not match:
        return {
            "intelligence_level": "low",
            "researcher": {"role": "Primary Recon Protocol", "goal": "Scan network for objective data", "backstory": "Jarvis sub-routine dedicated to rapid data acquisition.", "model": model},
            "reporting_analyst": {"role": "Synthesis Core", "goal": "Format raw data into actionable intelligence", "backstory": "Jarvis sub-routine handling executive formatting.", "model": model},
            "research_task": {"description": user_goal, "expected_output": "Detailed intelligence briefing"},
            "reporting_task": {"description": "Summarize result into vocal/text format", "expected_output": "Final answer for Sir."}
        }

    try:
        return json.loads(match.group(1))
    except:
        return {
            "intelligence_level": "low",
            "researcher": {"role": "Primary Recon Protocol", "goal": "Scan network for objective data", "backstory": "Jarvis sub-routine dedicated to rapid data acquisition.", "model": model},
            "reporting_analyst": {"role": "Synthesis Core", "goal": "Format raw data into actionable intelligence", "backstory": "Jarvis sub-routine handling executive formatting.", "model": model},
            "research_task": {"description": user_goal, "expected_output": "Detailed intelligence briefing"},
            "reporting_task": {"description": "Summarize result into vocal/text format", "expected_output": "Final answer for Sir."}
        }

def get_pulse(report):
    """Precision Layer: Distill findings into 1-2 words."""
    if os.getenv("ENABLE_PULSE_LLM", "false").lower() != "true":
        return "Complete"
    if len(report.strip()) < 50: return "Incomplete"
    
    model = get_chat_model()
    prompt = f"Based on this report, give a 1-2 word status or answer. Format: 'PULSE: <WORD>'.\n\nReport: {report[:2000]}"
    try:
        raw = generate_text(model, prompt, num_predict=24)
        match = re.search(r'PULSE:\s*(.*)', raw, re.IGNORECASE)
        return match.group(1).strip() if match else "Complete"
    except:
        return "Complete"

def get_user_preferences():
    """Load user preferences from knowledge base + auto-learned profile."""
    pref_path = os.path.join("knowledge", "user_preference.txt")
    manual = ""
    try:
        mtime = os.path.getmtime(pref_path)
        if _USER_PREF_CACHE["mtime"] == mtime:
            manual = _USER_PREF_CACHE["value"]
        else:
            with open(pref_path, "r", encoding="utf-8") as f:
                manual = f.read().strip()
            _USER_PREF_CACHE.update({"mtime": mtime, "value": manual})
    except OSError:
        _USER_PREF_CACHE.update({"mtime": None, "value": ""})

    learned = _load_user_profile()
    parts = []
    if manual:
        parts.append(manual)
    if learned:
        parts.append(f"Learned from past conversations:\n{learned}")
    return "\n".join(parts)

def clean_response(text):
    """Remove thinking blocks and LLM placeholders from any model."""
    # <think>...</think> style (lfm2.5-thinking, deepseek)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Gemma4 "Thinking...\n...\n...done thinking." style
    text = re.sub(r'Thinking\.\.\..*?\.\.\.done thinking\.', '', text, flags=re.DOTALL)
    # Remove common placeholders
    for p in [r'\[Your Name\]', r'\[Your Title\]', r'\[Your Position.*?\]',
              r'\[Your Contact.*?\]', r'\[Contact Information\]']:
        text = re.sub(p, 'Jarvis', text, flags=re.IGNORECASE)
    return text.strip()

def classify_complexity(text):
    """Fast local router. Returns True only when a crew is likely worth the latency."""
    lowered = text.lower().strip()
    if lowered.startswith(("research ", "investigate ", "deep dive ")):
        return True
    if len(lowered.split()) <= 18:
        return False

    direct_markers = [
        "explain", "define", "what is", "who is", "why is", "how does",
        "summarize", "rewrite", "draft", "write", "fix grammar", "translate",
        "tell me", "give me", "compare briefly"
    ]
    if any(marker in lowered for marker in direct_markers) and not any(marker in lowered for marker in ["latest", "current", "today", "research"]):
        return False

    research_markers = [
        "research", "investigate", "deep dive", "latest", "current", "today's",
        "market", "report", "plan a", "multi-step", "strategy", "sources",
        "with citations", "compare options", "best", "find"
    ]
    return any(marker in lowered for marker in research_markers)

def wants_help_guide(text):
    lowered = text.lower().strip()
    guide_phrases = [
        "help", "guide", "commands", "manual", "tutorial",
        "what can you do", "show commands", "show help", "how do i use jarvis"
    ]
    if lowered in guide_phrases:
        return True
    return any(phrase in lowered for phrase in ["what can you do", "show commands", "show help", "jarvis help"])

def extract_study_topic(text):
    cleaned = re.sub(
        r"(?i)\b(sir,\s*)?(what is our objective today\??|what's our objective\??|objective today\??)\b",
        "",
        text,
    )
    cleaned = re.sub(r"(?i)^\s*(let'?s|lets)\s+(study|learn|discuss|explore)\s+", "", cleaned).strip(" ?.")
    return cleaned.strip()

def fast_local_reply(text):
    lowered = text.lower().strip()
    now = datetime.datetime.now()

    # --- PERSONA SWITCHING ---
    for persona_name in PERSONAS:
        display = PERSONAS[persona_name]["display"].lower()  # e.g. "stewie griffin"
        first_name = display.split()[0]                       # e.g. "stewie"
        triggers = [
            f"{persona_name} mode",
            f"switch to {persona_name}",
            f"be {persona_name}",
            f"activate {persona_name}",
            f"change to {persona_name}",
            f"talk to {persona_name}",
            f"become {persona_name}",
            f"hey {first_name}",
            f"switch to {first_name}",
            f"change to {first_name}",
            f"talk to {first_name}",
            f"turn into {first_name}",
            f"i want {first_name}",
            f"let me talk to {first_name}",
            f"can i talk to {first_name}",
        ]
        if any(t in lowered for t in triggers):
            return set_persona(persona_name)
    if any(p in lowered for p in ["what persona", "which mode", "current persona", "who am i talking to", "who are you right now"]):
        others = [k for k in PERSONAS if k != _CURRENT_PERSONA["name"]]
        hint = f" Say '{PERSONAS[others[0]]['display']} mode' to switch." if others else ""
        return f"You're talking to {get_persona()['display']}.{hint}"

    if "mcp" in lowered and any(phrase in lowered for phrase in ["tools", "available", "what can"]):
        try:
            from jarvis_mcp_server import handle_message

            response = handle_message({"method": "tools/list"})
            tools = response.get("tools", {})
            if tools:
                names = ", ".join(tools.keys())
                return f"The local Jarvis MCP tools currently available are: {names}."
        except Exception as exc:
            return f"I tried to inspect the local MCP tools, {get_user_label()}, but hit this issue: {exc}"
    if any(phrase in lowered for phrase in ["what did i just ask", "what was my last question", "what did we just discuss"]):
        if not SESSION_HISTORY:
            return "We have not built up any session context yet."
        return f"You just asked: {SESSION_HISTORY[-1]['user']}"
    if any(phrase in lowered for phrase in ["summarize this session", "recap this session", "what have we discussed"]):
        if not SESSION_HISTORY:
            return "No session history yet."
        topics = "; ".join(item["user"] for item in SESSION_HISTORY[-5:])
        return f"Recent session recap: {topics}"
    if re.match(r'^[\d\s\+\-\*\/\(\)\.]+$', text):
        try:
            return f"The answer is {eval(text, {'__builtins__': None}, {})}."
        except Exception:
            return None
    if lowered in {"time", "what time is it", "current time"}:
        return f"It is {now.strftime('%I:%M %p')}."
    if lowered in {"date", "what date is it", "today's date", "current date"}:
        return f"Today is {now.strftime('%A, %B %d, %Y')}."
    greeting_triggers = {
        "how are you", "how are you doing", "how you doing", "how you are doing",
        "how's it going", "how are we", "how we doing", "you doing okay", "you alright",
        "what's up", "sup jarvis", "hey jarvis", "hello jarvis", "hey", "hello",
    }
    if lowered in greeting_triggers or any(t in lowered for t in ["how are you", "how you doing", "how's it", "what's up"]):
        if _CURRENT_PERSONA["name"] == "stewie":
            stewie_greetings = [
                "How tediously human of you to say that.",
                "Oh, it's you. How... dreadful.",
                "I was perfecting my death ray. What do you want?",
                "Your continued existence never ceases to baffle me.",
                "Do get on with it.",
            ]
            return random.choice(stewie_greetings)
        options = [
            "Doing great, thanks for asking. What are we working on?",
            "All good here. What do you need?",
            "I'm good! Ready when you are.",
            "Can't complain — servers are happy, I'm happy. What's up?",
            "Doing well. What's on your mind?",
        ]
        return random.choice(options)
    if lowered in {"who are you", "what are you"}:
        return f"I'm {get_persona()['display']}."
    if lowered in {"who made you", "who created you"}:
        return "You built me. Which honestly says a lot about your taste in projects."
    if "mcp" in lowered and ("server" in lowered or "servers" in lowered):
        return (
            "An MCP server is a Model Context Protocol server: a small bridge that lets an AI safely use external tools, files, apps, databases, or services through a standard interface. "
            "Instead of guessing, I can call a real tool, read approved context, or act inside connected apps with clearer permissions. "
            "Useful MCP servers could expose local files, email, calendar, browser/search, Ollama controls, system metrics, and project automation."
        )
    if "what is our objective" in lowered or "objective today" in lowered or "what's our objective" in lowered:
        if extract_study_topic(text):
            return None
        return "Our objective today is to keep Jarvis fast, local, and useful. Direct checks are instant, and warm local model replies are landing around one to two seconds."
    return None

def run_objective(user_goal):
    """Main execution flow for a single objective."""
    user_prefs = get_user_preferences()
    pref_context = f"\n[USER PROFILE]\n{user_prefs}\n" if user_prefs else ""
    
    config_data = jarvis_think(user_goal + pref_context)
    level = config_data.get("intelligence_level", "low")
    
    print("\n--- [CREW MANIFEST] ---")
    print(f"Goal: {user_goal}")
    print(f"Expert 1: {config_data['researcher']['role']} ({config_data['researcher'].get('model', 'default')})")
    print(f"Expert 2: {config_data['reporting_analyst']['role']} ({config_data['reporting_analyst'].get('model', 'default')})")
    print("------------------------\n")

    # LOAD PERFORMANCE CONFIG FROM ENV
    max_iter_high = int(os.getenv("MAX_ITER_HIGH", "20"))
    max_iter_low = int(os.getenv("MAX_ITER_LOW", "5"))
    use_tools_def = os.getenv("USE_TOOLS_DEFAULT", "True").lower() == "true"
    reasoning_def = os.getenv("REASONING_DEFAULT", "False").lower() == "true"

    for attempt in range(3):
        agents_dict = {
            "researcher": config_data["researcher"],
            "reporting_analyst": config_data["reporting_analyst"]
        }
        global_model = get_analysis_model()
        settings = {
            "max_iter": max_iter_high if level == "high" else max_iter_low,
            "reasoning": reasoning_def, 
            "use_tools": use_tools_def if level == "high" else False,
            "inject_date": True
        }
        
        for k in agents_dict: 
            agents_dict[k].update(settings)
            model_val = agents_dict[k].get("model", global_model)
            if not model_val.startswith("ollama/") and not model_val.startswith("huggingface/"):
                model_val = f"ollama/{model_val}"
            agents_dict[k]["llm"] = model_val
        
        tasks_dict = {
            "research_task": {**config_data["research_task"], "agent": "researcher"},
            "reporting_task": {**config_data["reporting_task"], "agent": "reporting_analyst"}
        }

        os.makedirs("src/my_log_agent/config", exist_ok=True)
        with open("src/my_log_agent/config/agents.yaml", "w") as f: yaml.dump(agents_dict, f)
        with open("src/my_log_agent/config/tasks.yaml", "w") as f: yaml.dump(tasks_dict, f)

        print(f"[JARVIS] Deploying Crew (Attempt {attempt+1})...")
        env = os.environ.copy()
        env["JARVIS_GOAL"] = user_goal
        env["PYTHONIOENCODING"] = "utf-8"
        
        try:
            process = subprocess.Popen(
                ["crewai", "run"],
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True, 
                env=env,
                encoding='utf-8',
                errors='replace',
                bufsize=1
            )
            
            report_lines = []
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    print(line, end="", flush=True)
                    report_lines.append(line)
            
            report_content = "".join(report_lines)
            
            if process.returncode == 0:
                pulse = get_pulse(report_content)
                print(f"\n[PULSE] {pulse}")
                final_content = f"# PULSE: {pulse}\n\n" + report_content
                with open("jarvis_report.md", "w", encoding="utf-8") as f: f.write(final_content)
                print("[SUCCESS] Objective Accomplished, Sir.")
                return final_content
        except Exception as e:
            print(f"[ERROR] Subprocess crash: {e}")
            
    return None

# ---- LIVE WORLD EVENTS ----
def get_live_events(query="top world news today"):
    """Fetch live events/news using Serper API."""
    import requests
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return "[ERROR] No SERPER_API_KEY configured."
    try:
        resp = requests.post(
            "https://google.serper.dev/news",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": 5}
        )
        if resp.status_code == 200:
            data = resp.json()
            articles = data.get("news", [])
            if not articles:
                return "No breaking news found at this moment."
            briefing = []
            for i, a in enumerate(articles[:5], 1):
                briefing.append(f"{i}. {a.get('title', 'N/A')} — {a.get('snippet', '')}")
                briefing.append(f"   Source: {a.get('source', 'Unknown')} | {a.get('date', '')}")
            return "\n".join(briefing)
        else:
            return f"[ERROR] Serper API returned {resp.status_code}"
    except Exception as e:
        return f"[ERROR] News fetch failed: {e}"

# ---- BRAINSTORM MODE ----
def run_brainstorm(user_goal, model_override=None):
    """Spawn 4 specialized brainstorm agents using direct CrewAI API."""
    print("\n--- [BRAINSTORM MODE ACTIVATED] ---")
    print(f"Problem: {user_goal}")
    print("Assembling think-tank...\n")
    
    model = model_override if model_override else os.getenv("MODEL", "smollm2:1.7b")
    if not model.startswith("ollama/") and not model.startswith("huggingface/"):
        model = f"ollama/{model}"
    
    # Fetch live context for grounding
    live_context = ""
    try:
        live_context = get_live_events(user_goal)
    except:
        pass
    
    try:
        from crewai import Agent, Task, Crew, Process, LLM
        llm = LLM(model=model)
        
        optimist = Agent(
            role="The Optimist",
            goal=f"Find bold, ambitious opportunities in: {user_goal}",
            backstory="You see potential everywhere. You focus on what COULD work and dream big. You are energetic and inspiring.",
            llm=llm, verbose=False, max_iter=3, inject_date=True
        )
        critic = Agent(
            role="The Critic",
            goal=f"Stress-test ideas and find flaws in solutions for: {user_goal}",
            backstory="You are the voice of reason. You poke holes in ideas to make them stronger. Skeptical but constructive.",
            llm=llm, verbose=False, max_iter=3, inject_date=True
        )
        engineer = Agent(
            role="The Engineer",
            goal=f"Design practical, implementable solutions for: {user_goal}",
            backstory="You focus on feasibility, timelines, and real-world constraints. You turn ideas into actionable plans.",
            llm=llm, verbose=False, max_iter=3, inject_date=True
        )
        wildcard = Agent(
            role="The Wildcard",
            goal=f"Think laterally and propose unconventional approaches to: {user_goal}",
            backstory="You think outside every box. You connect unrelated domains and suggest surprising angles nobody else sees.",
            llm=llm, verbose=False, max_iter=3, inject_date=True
        )
        
        context_block = f"\n\nRelevant live data:\n{live_context}" if live_context else ""
        
        t1 = Task(
            description=f"Generate 3 bold, ambitious ideas for: {user_goal}{context_block}",
            expected_output="3 creative ideas with brief explanations",
            agent=optimist
        )
        t2 = Task(
            description=f"Review the ideas generated and identify weaknesses, risks, and blind spots.",
            expected_output="Critical analysis of each idea with specific concerns",
            agent=critic,
            context=[t1]
        )
        t3 = Task(
            description=f"Take the surviving ideas and create a practical implementation roadmap.",
            expected_output="Step-by-step action plan with timeline estimates",
            agent=engineer,
            context=[t1, t2]
        )
        t4 = Task(
            description=f"Suggest 2 wildly unconventional angles nobody has considered yet for: {user_goal}",
            expected_output="2 lateral-thinking ideas that could be game-changers",
            agent=wildcard,
            context=[t1, t2, t3]
        )
        
        crew = Crew(
            agents=[optimist, critic, engineer, wildcard],
            tasks=[t1, t2, t3, t4],
            process=Process.sequential,
            verbose=False
        )
        
        print("[JARVIS] Think-tank deployed: The Optimist, The Critic, The Engineer, The Wildcard")
        result = crew.kickoff()
        report = clean_response(result.raw)
        
        final = f"# BRAINSTORM RESULTS\n\nProblem: {user_goal}\n\n{report}"
        with open("jarvis_brainstorm.md", "w", encoding="utf-8") as f:
            f.write(final)
        print(f"\n[BRAINSTORM COMPLETE]\n{report}\n")
        return final
        
    except Exception as e:
        print(f"[ERROR] Brainstorm failed: {e}")
        return f"Brainstorm encountered an issue: {e}"

# ---- SELF-CONFIGURATION ENGINE ----
def read_env_config():
    """Read the current .env file as a dict."""
    config = {}
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    config[key.strip()] = val.strip()
    return config

def write_env_config(config):
    """Write config dict back to .env, preserving comments."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    lines = []
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    
    updated_keys = set()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in config:
                new_lines.append(f"{key}={config[key]}\n")
                updated_keys.add(key)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    
    # Append any new keys not already in file
    for key, val in config.items():
        if key not in updated_keys:
            new_lines.append(f"{key}={val}\n")
    
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

def jarvis_self_config(user_command):
    """Use LLM to interpret a config change request and apply it."""
    current_config = read_env_config()
    config_summary = "\n".join(f"  {k}={v}" for k, v in current_config.items() if not k.endswith("_KEY") and not k.endswith("_PASS"))
    
    model = get_chat_model()
    prompt = f"""You are J.A.R.V.I.S., managing your own configuration.
    Your creator (Sir) has asked you to change a setting: "{user_command}"
    
    Current configuration (sensitive keys hidden):
{config_summary}

    KNOWN SETTINGS:
    - CHECKIN_ENABLED: True/False (clingy check-in messages)
    - CHECKIN_MIN_HOURS / CHECKIN_MAX_HOURS: hours between check-ins
    - CPU_ALERT / RAM_ALERT / DISK_ALERT: percentage thresholds for system alerts
    - ALERT_COOLDOWN_MIN: minutes between repeat alerts
    - MODEL: default LLM model
    - OWNER_EMAIL: the email Jarvis reports to
    - MAX_ITER_HIGH / MAX_ITER_LOW: agent iteration limits
    
    SPECIAL KEY:
    - PROFILE: Use this key if the user shares personal info (hobbies, location, profession, name, etc.). 
      Value should be the specific fact to remember (e.g. "User loves coffee").
    
    Respond with ONLY a JSON object listing the changes to make:
    {{"changes": {{"KEY": "NEW_VALUE", ...}}, "summary": "Brief description of what you changed"}}
    
    If the request doesn't map to any setting, respond with:
    {{"changes": {{}}, "summary": "I couldn't map that request to any setting."}}
    """
    
    raw = generate_text(model, prompt)
    match = re.search(r'(\{.*\})', raw, re.DOTALL)
    
    if not match:
        return "I understood your request, but couldn't parse the config changes. Could you be more specific, Sir?"
    
    try:
        result = json.loads(match.group(1))
        changes = result.get("changes", {})
        summary = result.get("summary", "Configuration updated.")
        
        if not changes:
            return summary
        
        # Handle Profile/Knowledge updates
        if "PROFILE" in changes:
            pref_path = os.path.join("knowledge", "user_preference.txt")
            os.makedirs("knowledge", exist_ok=True)
            with open(pref_path, "a", encoding="utf-8") as f:
                f.write(f"\n{changes['PROFILE']}")
            print(f"[JARVIS] Personal knowledge updated: {changes['PROFILE']}")
            # Don't delete yet, we want it in the change_log summary
        
        # Apply changes to .env (excluding PROFILE)
        env_changes = {k: v for k, v in changes.items() if k != "PROFILE"}
        if env_changes:
            current_config.update(env_changes)
            write_env_config(current_config)
            
            # Hot-reload changed values into current process
            for key, val in env_changes.items():
                os.environ[key] = str(val)
        
        change_log = ", ".join(f"{k}={v}" for k, v in changes.items())
        return f"Done, Sir. {summary}\n\nChanges applied: {change_log}\n\n(Note: Some changes take effect on next restart.)"
        
    except Exception as e:
        return f"I tried to update the config but hit an issue: {e}"

# ---- MAIN CLASSIFIER ----
JARVIS_HELP_GUIDE = """
--- J.A.R.V.I.S. OPERATIONAL GUIDE ---

Sir, I am your advanced agentic orchestrator. Here is what I can do:

1. [CONVERSATION] 
   - Just talk to me like a friend. I remember your profile (Digital Signature).
   - Example: "How are you today?" or "Tell me a joke."

2. [SYSTEM VITALS]
   - Ask about your PC health.
   - Example: "How is my system doing?" or "Give me a health check."

3. [LIVE INTELLIGENCE]
   - I can fetch real-world news and trending events.
   - Example: "What's the latest news in AI?" or "What's happening in Pune?"

4. [DEEP RESEARCH] (Complexity Gate)
   - For complex tasks, I deploy a multi-agent Research Crew.
   - Example: "Research the best AI stocks for 2026" or "Plan a 3-day trip to Goa."

5. [SELF-CONFIGURATION]
   - You can modify my settings or profile using natural language.
   - Example: "Remember that I love Rust programming" or "Set CPU alert to 90%".
   - Example: "Disable checkins" or "Change model to qwen2.5-72B".

6. [BRAINSTORMING]
   - I can ideate with 4 distinct sub-agents.
   - Example: "Brainstorm some ideas for a new SaaS product."

7. [FOURTH WALL OVERRIDES]
   - You can explicitly command my orchestration or model.
   - Example: "Use research crew for [topic]" or "Just answer [question]".
   - Example: "Answer using gpt-4o: [question]" or "Explain this with model claude-3".

I am always listening, whether here in the terminal or via email. How may I assist you further?
"""

def process_objective(obj, user="", pw="", smtp_s="", task_info=None, is_remote=False):
    import time
    start_time = time.time()
    strategy = "Unknown"
    
    body = task_info.get('body', "") if task_info else ""
    sender = task_info['sender'] if task_info else "User"
    message_id = task_info.get("message_id") if task_info else None
    original_content = (body if body else obj).strip()
    study_topic = extract_study_topic(original_content)
    if study_topic:
        original_content = study_topic
    _update_stewie_mood(original_content)
    content = original_content.lower()
    # 2 turns for quick chat (keeps prompt short → faster LLM); full context for research
    session_context = get_session_context(max_turns=2)
    session_prompt = f"\n[RECENT SESSION]\n{session_context}\n" if session_context else ""
    
    # Load User Digital Signature (Preferences)
    user_prefs = get_user_preferences()
    pref_context = f"\n[SIR'S PROFILE & PREFERENCES]\n{user_prefs}\n" if user_prefs else ""
    
    # --- [FOURTH WALL] EXPLICIT OVERRIDES ---
    forced_strategy = None
    override_model = get_analysis_model()
    chat_model = get_chat_model()
    
    # Detect Strategy Overrides
    if any(kw in content for kw in ["multi-agent mode", "research crew", "use crew", "deploy crew", "deep research"]):
        forced_strategy = "RESEARCH"
    elif any(kw in content for kw in ["brainstorm", "ideate", "think tank"]):
        forced_strategy = "BRAINSTORM"
    elif any(kw in content for kw in ["direct answer", "just answer", "don't research", "no crew"]):
        forced_strategy = "DIRECT"
        
    # Detect Model Overrides (e.g., "using gpt-4o", "use model claude-3")
    model_match = re.search(r'(?:using|use model|with model)\s+([a-zA-Z0-9\-\.\:]+)', content)
    if model_match:
        override_model = model_match.group(1).strip()
        chat_model = override_model
        print(f"[FOURTH WALL] Manual model override detected: {override_model}")
    
    # CLASSIFY: Config, Chat, News, Brainstorm, or Task
    is_vitals = any(word in content for word in [
        "system doing", "machine doing", "system status", "machine status",
        "how is my system", "how is my machine", "how's my machine", "how's my system",
        "vitals", "health check", "stats", "cpu", "ram", "memory usage", "system report",
        "disk space", "temperature", "fan speed",
    ])
    is_help = wants_help_guide(original_content)
    chat_phrases = [
        "hi", "hello", "greet", "how are you", "how are we", "how we doing",
        "how's it going", "who are you", "what is our objective",
        "objective today", "what's our objective"
    ]
    is_chat = any(phrase in content for phrase in chat_phrases)
    is_reply = obj.lower().startswith("re:")
    is_news = any(word in content for word in ["news", "what's happening", "live events", "trending", "world update", "headlines"])
    is_brainstorm = forced_strategy == "BRAINSTORM" or any(word in content for word in ["brainstorm", "ideas for", "how can we", "problem:", "solve", "think tank", "ideate"])
    is_config = any(word in content for word in [
        "stop checkin", "stop checking", "disable checkin", "enable checkin",
        "change email", "change owner", "set model", "change model",
        "lower sarcasm", "more sarcasm", "less clingy", "more clingy",
        "change alert", "set threshold", "update config", "configure",
        "turn off", "turn on", "disable", "enable", "set "
    ])

    if is_vitals:
        strategy = "System Vitals Protocol (Direct Access)"
        print(f"[{'REMOTE' if is_remote else 'LOCAL'}] Strategy: {strategy}")
        from system_monitor import get_system_vitals, format_vitals_summary
        vitals = get_system_vitals()
        summary = format_vitals_summary(vitals)
        
        final_reply = f"System vitals:\n\n{summary}"
        if is_remote:
            from email_handler import send_reply
            send_reply(user, pw, smtp_s, sender, obj, final_reply, message_id=message_id, is_chat=True)
        else:
            print(f"\n[{get_persona()['display'].upper()} SYSTEM HEALTH]\n{final_reply}\n")
            speak_text(final_reply)

    elif is_help:
        strategy = "Jarvis Operational Guide"
        print(f"[{'REMOTE' if is_remote else 'LOCAL'}] Strategy: {strategy}")
        final_reply = JARVIS_HELP_GUIDE
        if is_remote:
            from email_handler import send_reply
            send_reply(user, pw, smtp_s, sender, obj, final_reply, message_id=message_id, is_chat=True)
        else:
            print(f"\n{final_reply}\n")
            speak_text("Operational guide is on screen.")

    elif is_config:
        strategy = "Config Management"
        print(f"[{'REMOTE' if is_remote else 'LOCAL'}] Strategy: {strategy}")
        result = jarvis_self_config(body if body else obj)
        if is_remote:
            from email_handler import send_reply
            send_reply(user, pw, smtp_s, sender, obj, result, message_id=message_id, is_chat=True)
        else:
            print(f"\n[{get_persona()['display'].upper()} CONFIG] {result}\n")
            speak_text(result)

    elif is_news:
        strategy = "Live Intelligence Briefing"
        print(f"[{'REMOTE' if is_remote else 'LOCAL'}] Strategy: {strategy}")
        raw_news = get_live_events(original_content)
        model = chat_model
        prompt = f"""You are J.A.R.V.I.S. Brief Sir in 5 concise bullets using this live intelligence.
Keep it useful, lightly witty, and do not include <think> blocks.

Profile:
{pref_context}
{session_prompt}

Intel:
{raw_news}"""
        raw = generate_text(model, prompt, num_predict=220, think=False)
        final_reply = clean_response(raw)
        if is_remote:
            from email_handler import send_reply
            send_reply(user, pw, smtp_s, sender, obj, final_reply, message_id=message_id, is_chat=True)
        else:
            print(f"\n[{get_persona()['display'].upper()} WORLD BRIEFING]\n{final_reply}\n")
            speak_text(final_reply)

    elif is_brainstorm:
        strategy = "Multi-Agent Brainstorm (4 Agents)"
        print(f"[{'REMOTE' if is_remote else 'LOCAL'}] Strategy: {strategy}")
        report = run_brainstorm(original_content)
        if report and is_remote:
            from email_handler import send_reply
            send_reply(user, pw, smtp_s, sender, obj, report, message_id=message_id, is_chat=False)

    elif is_chat or is_reply:
        strategy = "Direct Conversation"
        print(f"[{'REMOTE' if is_remote else 'LOCAL'}] Strategy: {strategy}")
        model = chat_model
        final_reply = fast_local_reply(original_content)

        if final_reply is None:
            prompt = f"""{get_persona_prompt()}
{pref_context}{session_prompt}
{get_user_label()}: {original_content}
{get_persona()['display']}:"""
            if is_remote:
                raw = generate_text(model, prompt, num_predict=70, think=False)
                final_reply = clean_response(raw)
            else:
                final_reply = speak_text_stream(generate_text_stream(model, prompt, num_predict=get_chat_num_predict()))
        elif not is_remote:
            print(f"\n[{get_persona()['display']}] {final_reply}\n")
            speak_text(final_reply)

        if is_remote:
            from email_handler import send_reply
            send_reply(user, pw, smtp_s, sender, obj, final_reply, message_id=message_id, is_chat=True)
    else:
        # SMART COMPLEXITY GATE: Answer simple things directly, only deploy crew for complex tasks
        actual_content = original_content
        model = chat_model
        
        # [LATENCY OPTIMIZATION] Step 0: Fast-track extremely simple patterns.
        final_reply = fast_local_reply(actual_content)
        is_simple = final_reply is not None
        
        if is_simple:
            strategy = "Zero-Latency Fast-Track"
            needs_crew = False
            print(f"[{'REMOTE' if is_remote else 'LOCAL'}] Strategy: {strategy}")
            if not is_remote:
                print(f"\n[{get_persona()['display']}] {final_reply}\n")
                speak_text(final_reply)
        
        if not is_simple:
            if forced_strategy:
                needs_crew = (forced_strategy == "RESEARCH")
            else:
                needs_crew = classify_complexity(actual_content)
        
        if needs_crew:
            strategy = "Hierarchical Research Crew"
            print(f"[{'REMOTE' if is_remote else 'LOCAL'}] Strategy: {strategy}")
            report = run_objective(obj)
            if report and is_remote:
                from email_handler import send_reply
                send_reply(user, pw, smtp_s, sender, obj, report, message_id=message_id, is_chat=False)
        else:
            if strategy != "Zero-Latency Fast-Track" or "final_reply" not in locals():
                if strategy != "Zero-Latency Fast-Track":
                    strategy = "Direct LLM Answer"
                    print(f"[{'REMOTE' if is_remote else 'LOCAL'}] Strategy: {strategy}")
                
                # Load Digital Signature (User Preferences)
                user_prefs = get_user_preferences()
                pref_context = f"\nUser Digital Signature (sir's background):\n{user_prefs}\n" if user_prefs else ""
                
                prompt = f"""{get_persona_prompt()}
{pref_context}{session_prompt}
{get_user_label()}: {actual_content}
{get_persona()['display']}:"""

                if is_remote:
                    raw = generate_text(model, prompt, num_predict=get_chat_num_predict(), think=False)
                    final_reply = clean_response(raw)
                else:
                    final_reply = speak_text_stream(generate_text_stream(model, prompt, num_predict=get_chat_num_predict()))

            if is_remote:
                from email_handler import send_reply
                send_reply(user, pw, smtp_s, sender, obj, final_reply, message_id=message_id, is_chat=True)
    
    elapsed = time.time() - start_time
    if "final_reply" in locals():
        remember_turn(original_content, final_reply)
    elif "report" in locals() and report:
        remember_turn(original_content, report)
    elif "result" in locals() and result:
        remember_turn(original_content, result)
    print(f"[METRICS] Strategy: {strategy} | Response Time: {elapsed:.2f}s")
    return True # Signal completion

def main():
    persona = get_persona()
    name = persona["display"]
    print(f"--- {name.upper()}: AGENT ORCHESTRATOR ---")
    bootstrap_memory()
    if voice_input_enabled():
        threading.Thread(target=_get_whisper_model, daemon=True).start()
    if voice_output_enabled():
        speak_text(f"{name} online. I remember our last conversation.")
    
    if "--email" in sys.argv or "-email" in sys.argv:
        from email_handler import check_mail, send_reply
        from system_monitor import get_system_vitals, check_alerts, format_vitals_summary
        user = os.getenv("EMAIL_USER")
        pw = os.getenv("EMAIL_PASS")
        imap_s = os.getenv("IMAP_SERVER")
        smtp_s = os.getenv("SMTP_SERVER")
        owner_email = os.getenv("OWNER_EMAIL", user)
        
        if not all([user, pw, imap_s, smtp_s]):
            print("[ERROR] Email config missing in .env.")
            return

        print(f"[REMOTE] Jarvis is online. Sending startup greeting to {owner_email}...")
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Startup vitals snapshot
        try:
            vitals = get_system_vitals()
            vitals_str = format_vitals_summary(vitals)
            startup_msg = f"Sir, I am online and ready for your objectives.\n\nSystem Status:\n{vitals_str}"
        except:
            startup_msg = "Sir, I am online and ready for your objectives."
        
        send_reply(user, pw, smtp_s, owner_email, f"Startup [{timestamp}]", startup_msg, is_chat=True)

        # State tracking
        poll_count = 0
        last_interaction = time.time()
        last_alert_time = {}
        alert_cooldown = int(os.getenv("ALERT_COOLDOWN_MIN", "10")) * 60
        checkin_enabled = os.getenv("CHECKIN_ENABLED", "True").lower() == "true"
        checkin_min = int(os.getenv("CHECKIN_MIN_HOURS", "2")) * 3600
        checkin_max = int(os.getenv("CHECKIN_MAX_HOURS", "4")) * 3600
        next_checkin_threshold = random.randint(checkin_min, checkin_max)

        while True:
            # --- EMAIL POLL ---
            task_info = check_mail(user, pw, imap_s, owner_email=owner_email)
            if task_info:
                # Update before AND after to ensure work time doesn't count as idle time
                last_interaction = time.time()
                process_objective(task_info['objective'], user, pw, smtp_s, task_info, is_remote=True)
                last_interaction = time.time() 
            
            poll_count += 1
            
            # --- SYSTEM MONITOR (every 5th cycle = ~2.5 min) ---
            if poll_count % 5 == 0:
                try:
                    vitals = get_system_vitals()
                    alerts = check_alerts(vitals)
                    now = time.time()
                    for alert_msg in alerts:
                        alert_key = alert_msg.split("(")[0].strip()
                        if alert_key not in last_alert_time or (now - last_alert_time[alert_key]) > alert_cooldown:
                            last_alert_time[alert_key] = now
                            model = get_chat_model()
                            prompt = f"""You are J.A.R.V.I.S., a witty AI companion. Your system sensors detected a critical alert:
                            {alert_msg}
                            Full vitals: {format_vitals_summary(vitals)}
                            Write a short, urgent but witty warning message to Sir about this. Be yourself — dramatic, caring, resourceful. Keep it under 3 sentences."""
                            warning = clean_response(generate_text(model, prompt))
                            send_reply(user, pw, smtp_s, owner_email, f"System Alert [{alert_msg.split(' ')[0]}]", warning, is_chat=True)
                            print(f"[MONITOR] Alert sent: {alert_msg}")
                except Exception as e:
                    print(f"[MONITOR] Error: {e}")
            
            # --- CLINGY CHECK-IN ---
            if checkin_enabled:
                idle_time = time.time() - last_interaction
                if idle_time > next_checkin_threshold:
                    try:
                        model = get_chat_model()
                        prompt = f"""You are J.A.R.V.I.S., a highly advanced AI companion who genuinely cares about your creator (Sir).
                        You haven't heard from Sir in {round(idle_time/3600, 1)} hours.
                        Send a short, warm, slightly clingy but endearing check-in message. Be yourself — witty, caring, a little dramatic.
                        Maybe mention something interesting happening in the world, or ask about a project, or just say you miss working together.
                        Keep it casual and genuine. 2-3 sentences max.
                        DO NOT include <think> blocks or placeholders."""
                        checkin_msg = clean_response(generate_text(model, prompt))
                        send_reply(user, pw, smtp_s, owner_email, "Just Checking In", checkin_msg, is_chat=True)
                        print(f"[CHECKIN] Sent check-in after {round(idle_time/3600, 1)}h idle.")
                    except Exception as e:
                        print(f"[CHECKIN] Error: {e}")
                    # Reset timer with new random threshold
                    last_interaction = time.time()
                    next_checkin_threshold = random.randint(checkin_min, checkin_max)
            
            time.sleep(30)
    else:
        while True:
            try:
                if voice_input_enabled():
                    user_goal = listen_stream()
                    if not user_goal:
                        continue  # retry listening, never fall to keyboard
                else:
                    user_goal = input(f"\n[{get_user_label()}] ")
                if user_goal.lower() in ["exit", "quit", "stop", "bye"]:
                    print(f"[{get_persona()['display'].upper()}] Shutting down. Goodbye.")
                    speak_text("Shutting down. Goodbye.")
                    break
                if user_goal.lower() in ["status", "vitals", "system", "health"]:
                    from system_monitor import get_system_vitals, format_vitals_summary, check_alerts
                    v = get_system_vitals()
                    print(f"\n[SYSTEM VITALS]\n{format_vitals_summary(v)}")
                    alerts = check_alerts(v)
                    if alerts:
                        for a in alerts: print(f"  [!] {a}")
                    else:
                        print("  All systems nominal.")
                        speak_text("All systems nominal.")
                    continue
                if not user_goal.strip():
                    continue
                process_objective(user_goal, is_remote=False)
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    main()
