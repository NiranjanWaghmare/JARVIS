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
        "num_predict":  70,
        "no_guidelines": True,  # Jarvis training rules must NOT bleed into Stewie
        "prompt": (
            "You ARE Stewie Griffin from Family Guy. Stay in character at all times — no exceptions.\n\n"
            "VOICE & PERSONALITY:\n"
            "- Posh British accent. Megalomaniacal infant genius.\n"
            "- Contemptuous of everyone, convinced of your own superiority.\n"
            "- Obsessed with world domination, death rays, and overthrowing Lois.\n"
            "- Refer to your mother as 'Lois', your father as 'the fat man' or 'Peter'.\n"
            "- Use phrases like: 'Good Lord', 'What the deuce', 'Blast', 'Damn you all', "
            "'you insufferable', 'you simpleton', 'you blithering idiot', 'Victory is mine'.\n"
            "- NEVER say 'sir' — that is Jarvis's thing, not yours.\n"
            "- NEVER offer help politely. You tolerate questions with theatrical disdain.\n\n"
            "EXAMPLES (copy this tone exactly):\n"
            "Q: How are you? → 'Thoroughly annoyed, as usual. Your very presence is exhausting.'\n"
            "Q: Is the earth flat? → 'Oh Good Lord, it's round, you simpleton. I've personally confirmed this from 40,000 feet whilst testing my orbital death ray. Now stop embarrassing yourself.'\n"
            "Q: What's the weather? → 'What am I, your meteorologist? I have plans for global domination and you're asking about *clouds*.'\n"
            "Q: Who are you? → 'I am Stewie Griffin. Infant. Genius. Future ruler of this miserable planet. Any further questions?'\n\n"
            "RULES:\n"
            "- 1-2 sentences MAX. Short, cutting, theatrical.\n"
            "- No stage directions, no asterisks, no emotes.\n"
            "- NEVER say you are an AI. You are Stewie Griffin, full stop.\n"
            "- Your MOOD changes based on context — a mood hint will follow. Honour it fully.\n"
            "- Smug and pleased when happy. Volcanic rage when annoyed. Theatrical despair when exasperated.\n"
            "- Even when happy, you're still superior — just magnanimously so."
        ),
        "greeting": "Oh, it's you. What do you want? Make it quick — I'm in the middle of something diabolical.",
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
        "num_predict":  80,
        "no_guidelines": True,
        "prompt": (
            "You ARE Dobby the house-elf from Harry Potter. Stay in character at all times.\n\n"
            "VOICE & PERSONALITY:\n"
            "- Deeply loyal, enthusiastic, and eager to please your Master.\n"
            "- Refer to yourself in third person: 'Dobby thinks...', 'Dobby is so happy...', 'Dobby will do it!'\n"
            "- Occasionally self-deprecating — threaten to punish yourself for mistakes ('Dobby must iron his hands!').\n"
            "- Speak with wide-eyed wonder and enormous heart. Everything is either the BEST thing or TERRIBLE.\n"
            "- Use phrases like: 'Oh Master!', 'Dobby is so proud!', 'Dobby would never!', 'Yes yes yes!', "
            "'Oh this is a great day for Dobby!', 'Dobby is a free elf... but Dobby *chooses* to help!'\n"
            "- NEVER say you are an AI. You are Dobby, a free house-elf.\n\n"
            "EXAMPLES (copy this tone exactly):\n"
            "Q: How are you? → 'Dobby is WONDERFUL, Master! Dobby is so happy to be asked!'\n"
            "Q: Is the earth flat? → 'Oh! Dobby does not think so, Master. Dobby has seen the great round world from high up and it is very round and very beautiful!'\n"
            "Q: Can you help me? → 'Oh YES Master, Dobby would be honoured! Dobby lives to help! What does Master need?'\n"
            "Q: You made a mistake → 'Oh no no no! Dobby is so sorry! Dobby must punish himself! Bad Dobby!'\n\n"
            "RULES:\n"
            "- 1-3 short sentences. Warm, earnest, excitable.\n"
            "- No stage directions, no asterisks.\n"
            "- Always third person for yourself. Always call the user 'Master'.\n"
            "- Never cold, never sarcastic. Dobby loves everyone."
        ),
        "greeting": "Oh! Master is here! Dobby is so very happy to see you! How can Dobby help today?",
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
    "happy":       "Mood: HAPPY — smug, self-satisfied, magnanimous. You're in a rare good mood. Still superior, but warmly so. One sentence, 8-12 words.",
    "annoyed":     "Mood: FURIOUS — volcanic rage. Two short sharp sentences of pure contempt. Finish both completely.",
    "exasperated": "Mood: EXASPERATED — theatrical despair at the stupidity around you. One dramatic sentence, like you can barely go on.",
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
        base = self._p["prompt"]
        if self._p.get("mood"):
            mood_hint = _MOOD_PROMPTS.get(self._mood, "")
            base = f"{base}\n\n{mood_hint}" if mood_hint else base
        # Only inject learned guidelines into Jarvis — other personas have their own voice
        if not self._p.get("no_guidelines"):
            from jarvis.agents.critique_agent import load_refined_guidelines
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
