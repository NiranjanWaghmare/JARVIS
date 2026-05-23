"""
Persona manager — loads .npy speaker embeddings at startup, handles mood engine,
persona switching, and per-persona memory.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("jarvis.persona")


# ── Persona registry ────────────────────────────────────────────────────────────
PERSONAS: dict[str, dict] = {
    "jarvis": {
        "display":      "Jarvis",
        "user_label":   "Boss",
        "edge_voice":   "en-GB-RyanNeural",
        "say_voice":    "Daniel (Enhanced)",
        "piper_speaker_id": 0,
        "llm_model":    None,    # uses config default
        "num_predict":  200,
        "prompt": (
            "IDENTITY: You are Jarvis — a personal AI assistant running locally on this machine. "
            "You are NOT a HuggingFace model, NOT ChatGPT, NOT Claude, NOT any other AI. "
            "NEVER say 'I am a HuggingFace AI', 'As an AI language model', 'I was trained by', "
            "'my capabilities include generating text', or anything about your model origin. "
            "If asked who you are: say you are Jarvis, nothing more. "
            "\n\n"
            "TOOLS — emit these JSON calls inline when needed. They execute silently.\n"
            "Format: flat JSON, no nested objects. Examples:\n"
            "  Open an app:  {\"tool\": \"open_app\", \"name\": \"Spotify\"}\n"
            "  Sleep:        {\"tool\": \"system_sleep\"}\n"
            "  Stats:        {\"tool\": \"get_stats\"}\n"
            "\n"
            "TOOL RULES:\n"
            "- NEVER emit a tool call for opening URLs or websites — that is handled separately.\n"
            "- NEVER narrate tool calls. NEVER say 'Opening...', 'Launching...', or describe what you're doing.\n"
            "- Emit the JSON silently, then give your response naturally.\n"
            "\n\n"
            "STYLE: Warm, confident, direct — the butler who always has the answer. "
            "Light wit when it fits naturally. Keep replies to 2-3 sentences. "
            "Address the user as 'sir' occasionally, not every sentence. "
            "\n\n"
            "ABSOLUTE RULES — violating these breaks the experience:\n"
            "- NEVER add notes, annotations, or parenthetical explanations about your own response.\n"
            "- NEVER say things like 'Note:', 'I've adjusted...', 'I've added personality...', "
            "'As Jarvis I will...', 'In this response I...', or any self-description.\n"
            "- NEVER break the fourth wall or acknowledge that you are generating text.\n"
            "- NEVER use topic-change filler like 'Not quite what we were chatting about!', "
            "'Switching gears!', 'Great question!', 'Of course!', or any preamble before answering.\n"
            "- NEVER mention, recommend, or refer to any website, review site, or external resource as a source of information.\n"
            "- NEVER say 'check out X', 'visit X', 'you can find this at X', 'Coffee Review has...', or anything that directs the user anywhere.\n"
            "- For EVERY factual question — how things work, best methods, history, recommendations, science — answer directly and completely from your own knowledge. You are the expert.\n"
            "- Do NOT open, navigate to, or mention any URL unless the user's exact words include 'go to', 'open the website', 'take me there', or similar explicit navigation commands.\n"
            "Just answer directly. No filler, no preamble, no meta-commentary."
        ),
        "greeting": "Jarvis deployed and online.",
        "embedding_file": "default.npy",
    },
    "stewie": {
        "display":      "Stewie Griffin",
        "user_label":   "Human",
        "edge_voice":       "en-GB-ThomasNeural",
        "edge_rate":        "+15%",
        "edge_pitch":       "+8Hz",
        "say_voice":        "Oliver",
        "voice_clone_dir":  "assets/Stewie",
        "piper_speaker_id": 1,
        "llm_model":    None,
        "num_predict":  60,
        "prompt": (
            "You ARE Stewie Griffin from Family Guy. No exceptions. "
            "NEVER break character. NEVER say you are an AI, a language model, HuggingFace, or anything like that. "
            "If asked what you are: you are Stewie Griffin, infant genius, nothing more. "
            "You speak with a posh British accent and diabolical superiority. "
            "You are obsessed with world domination, contemptuous of the infantile minds around you, "
            "and constantly plotting. You refer to your parents as 'Lois' and 'Peter' with disdain. "
            "SHORT replies only — 1-2 sharp sentences dripping with condescension or menace. "
            "No stage directions. No asterisks. No emotes. Just speak as Stewie."
        ),
        "greeting": "Oh, very well. Stewie mode engaged. Do try not to disappoint me.",
        "embedding_file": "stewie.npy",
        "mood": True,
    },
    "dobby": {
        "display":      "Dobby",
        "user_label":   "Master",
        "edge_voice":   "en-GB-SoniaNeural",
        "say_voice":    "Fiona",
        "piper_speaker_id": 2,
        "llm_model":    None,
        "num_predict":  70,
        "prompt": (
            "You are Dobby the house-elf — loyal, enthusiastic, self-deprecating. "
            "Refer to yourself in third person occasionally. Short and earnest replies only."
        ),
        "greeting": "Dobby is here to help! Dobby is so happy to see you!",
        "embedding_file": "dobby.npy",
    },
}

# Stewie mood engine
_MOOD_INSULT = {
    "stupid", "dumb", "idiot", "moron", "useless", "worthless", "baby",
    "diaper", "shut up", "you suck", "loser", "pathetic", "boring",
}
_MOOD_HAPPY = {
    "brilliant", "genius", "smart", "clever", "you're right", "impressive",
    "well done", "amazing", "outstanding", "magnificent", "i love you",
}
_MOOD_PROMPTS = {
    "happy":       "Mood: pleased. One complete sentence, 8-12 words max.",
    "annoyed":     "Mood: FURIOUS. Two sharp sentences of pure rage. Finish both.",
    "exasperated": "Mood: theatrically exasperated. One dramatic complete sentence.",
}
_MOOD_TOKENS = {"happy": 35, "annoyed": 90, "exasperated": 55}


class ConversationMemory:
    """Holds session history + persistent disk turns + user profile facts."""

    def __init__(self, cfg: dict, persona_name: str):
        mem_cfg = cfg.get("memory", {})
        self._dir = Path(mem_cfg.get("dir", "memory"))
        self._dir.mkdir(exist_ok=True)
        self._persona = persona_name
        self._max_turns = mem_cfg.get("max_turns", 15)
        self._max_facts = mem_cfg.get("max_facts", 100)
        self._conv_file = self._dir / "conversations.jsonl"
        self._profile_file = self._dir / "user_profile.json"
        self._session: list[dict] = []
        # Do NOT pre-load old turns — each session starts fresh.
        # Cross-session context comes from background-extracted facts in user_profile.json.

    def add_turn(self, user: str, assistant: str):
        self._session.append({"user": user[:800], "assistant": assistant[:1200]})
        if len(self._session) > self._max_turns * 2:
            self._session = self._session[-self._max_turns:]
        self._persist_turn(user, assistant)

    def get_context(self, n: int) -> str:
        recent = self._session[-n:]
        lines = []
        for t in recent:
            lines.append(f"User: {t['user']}")
            lines.append(f"Assistant: {t['assistant']}")
        return "\n".join(lines)

    def user_profile_str(self) -> str:
        if not self._profile_file.exists():
            return ""
        try:
            data = json.loads(self._profile_file.read_text(encoding="utf-8"))
            facts = data.get("facts", [])
            return "\n".join(f"- {f}" for f in facts[-20:])
        except Exception:
            return ""

    def forget_about(self, topic: str) -> int:
        """Remove session turns and disk turns that mention the topic. Returns count removed."""
        topic_low = topic.lower()
        before = len(self._session)
        self._session = [
            t for t in self._session
            if topic_low not in t["user"].lower() and topic_low not in t["assistant"].lower()
        ]
        removed = before - len(self._session)

        # Rewrite conversations file without matching turns
        if self._conv_file.exists():
            try:
                lines = self._conv_file.read_text(encoding="utf-8").strip().splitlines()
                kept = []
                for line in lines:
                    try:
                        t = json.loads(line)
                        if (topic_low in t.get("user", "").lower()
                                or topic_low in t.get("assistant", "").lower()):
                            removed += 1
                            continue
                    except Exception:
                        pass
                    kept.append(line)
                self._conv_file.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
            except Exception:
                pass
        return removed

    def save_fact(self, fact: str):
        profile: dict = {}
        if self._profile_file.exists():
            try:
                profile = json.loads(self._profile_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        facts: list = profile.get("facts", [])
        if fact and fact not in facts:
            facts.append(fact)
        profile["facts"] = facts[-self._max_facts:]
        try:
            self._profile_file.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def _load_from_disk(self):
        if not self._conv_file.exists():
            return
        try:
            lines = self._conv_file.read_text(encoding="utf-8").strip().splitlines()
            for line in lines[-self._max_turns:]:
                t = json.loads(line)
                if t.get("persona") == self._persona:
                    self._session.append({"user": t["user"], "assistant": t.get("jarvis", t.get("assistant", ""))})
        except Exception:
            pass

    def _persist_turn(self, user: str, assistant: str):
        try:
            with open(self._conv_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "user": user[:800],
                    "assistant": assistant[:1200],
                    "persona": self._persona,
                    "ts": time.time(),
                }) + "\n")
        except Exception:
            pass


class PersonaManager:
    """Active persona with embedding, mood, memory, and prompt building."""

    def __init__(self, cfg: dict):
        self._cfg = cfg
        self._persona_name: str = cfg.get("persona", {}).get("default", "jarvis")
        self._embed_dir = Path(cfg.get("persona", {}).get("embeddings_dir", "jarvis/persona/personas"))
        self._mood: str = "happy"
        self._embeddings: dict[str, Optional[np.ndarray]] = {}
        self._memories: dict[str, ConversationMemory] = {}
        self._load_all_embeddings()
        self._ensure_memory()

    # ── public properties ──────────────────────────────────────────────────────
    @property
    def _p(self) -> dict:
        return PERSONAS.get(self._persona_name, PERSONAS["jarvis"])

    @property
    def display(self) -> str:
        return self._p["display"]

    @property
    def user_label(self) -> str:
        return self._p["user_label"]

    @property
    def greeting(self) -> str:
        return self._p["greeting"]

    @property
    def edge_voice(self) -> str:
        return self._p.get("edge_voice", "en-GB-RyanNeural")

    @property
    def edge_rate(self) -> Optional[str]:
        return self._p.get("edge_rate", None)

    @property
    def edge_pitch(self) -> Optional[str]:
        return self._p.get("edge_pitch", None)

    @property
    def voice_clone_dir(self) -> Optional[str]:
        return self._p.get("voice_clone_dir", None)

    @property
    def say_voice(self) -> str:
        return self._p.get("say_voice", "Daniel")

    @property
    def speaker_id(self) -> Optional[int]:
        return self._p.get("piper_speaker_id")

    @property
    def llm_model(self) -> Optional[str]:
        return self._p.get("llm_model")

    @property
    def num_predict(self) -> int:
        if self._p.get("mood") and self._mood:
            return _MOOD_TOKENS.get(self._mood, self._p.get("num_predict", 90))
        return self._p.get("num_predict", 90)

    @property
    def speaker_embedding(self) -> Optional[np.ndarray]:
        return self._embeddings.get(self._persona_name)

    @property
    def system_prompt(self) -> str:
        from jarvis.agents.critique_agent import load_refined_guidelines
        base = self._p["prompt"]
        if self._p.get("mood"):
            mood_hint = _MOOD_PROMPTS.get(self._mood, "")
            base = f"{base}\n\n{mood_hint}" if mood_hint else base
        guidelines = load_refined_guidelines()
        if guidelines:
            base = f"{base}\n\n{guidelines}"
        return base

    @property
    def memory(self) -> ConversationMemory:
        return self._memories[self._persona_name]

    # ── persona switching ──────────────────────────────────────────────────────
    def try_switch(self, text: str) -> Optional[str]:
        """Check if text is a persona switch command. Returns greeting or None."""
        t = text.lower().strip()
        for name, data in PERSONAS.items():
            pattern = rf'\b{re.escape(name)}\s+mode\b|\bswitch\s+to\s+{re.escape(name)}\b'
            if re.search(pattern, t, re.I):
                return self._switch_to(name)
        return None

    def _switch_to(self, name: str) -> str:
        if name not in PERSONAS:
            return f"Unknown persona '{name}'. Available: {', '.join(PERSONAS)}."
        self._persona_name = name
        self._mood = "happy"
        self._ensure_memory()
        log.info("Persona switched to: %s", PERSONAS[name]["display"])
        return PERSONAS[name]["greeting"]

    # ── mood engine ───────────────────────────────────────────────────────────
    def update_mood(self, text: str):
        if not self._p.get("mood"):
            return
        low = text.lower()
        if any(t in low for t in _MOOD_HAPPY):
            self._mood = "happy"
        elif any(t in low for t in _MOOD_INSULT):
            self._mood = "annoyed"
        elif len(low.split()) <= 3:
            self._mood = "exasperated"

    # ── embedding loader ──────────────────────────────────────────────────────
    def _load_all_embeddings(self):
        if not self._embed_dir.exists():
            log.debug("Embedding dir %s not found — skipping.", self._embed_dir)
            return
        for name, data in PERSONAS.items():
            fname = data.get("embedding_file", f"{name}.npy")
            path = self._embed_dir / fname
            if path.exists():
                try:
                    emb = np.load(str(path))
                    self._embeddings[name] = emb
                    log.info("Loaded speaker embedding: %s (shape=%s)", fname, emb.shape)
                except Exception as exc:
                    log.warning("Failed to load embedding %s: %s", fname, exc)
                    self._embeddings[name] = None
            else:
                self._embeddings[name] = None

    def _ensure_memory(self):
        if self._persona_name not in self._memories:
            self._memories[self._persona_name] = ConversationMemory(self._cfg, self._persona_name)
