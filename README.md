# Jarvis — Local AI Voice Assistant

A fully local, voice-first AI companion running on Apple Silicon. Persona switching, Piper TTS with speaker embeddings, streaming speech-to-text, wake-word detection, multi-agent research, and a Siri-style overlay HUD. **No cloud dependency for core features.**

### Anime episode generator (new)

Story → ~10 min open-source anime video pipeline lives in
[`src/anime_episode/`](src/anime_episode/) on **this GitHub repo** (not Hugging Face).
HF is only for downloading model weights into ComfyUI — see
[`WHERE.md`](src/anime_episode/WHERE.md) and [`MODELS.md`](src/anime_episode/MODELS.md).
Architecture: [`ARCHITECTURE.md`](src/anime_episode/ARCHITECTURE.md).

---

## Architecture

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         🔧 TOOL LAYER                                       ║
║  SYSTEM CONTROL   APP LAUNCHER   SYSTEM STATS   EMAIL AGENT   HUGGINGFACE  ║
║  (shutdown/sleep)  (open -a)     (CPU/RAM/disk)  (IMAP/SMTP)  (inference)  ║
╚══════════════════════════════════════════════════════════════════════════════╝
          ▲                               ▲
          │         function calls        │
          │                               │
╔═════════╧═══╗    ╔════════════════╗    ╔╧════════════════════════════════╗
║ EMAIL       ║    ║ TRIPLE TAP     ║    ║ PORCUPINE WAKE WORD             ║
║ TRIGGER     ║    ║ GESTURE        ║    ║ "Hey Jarvis" — ~1% CPU          ║
║ (StarMail)  ║    ║ (pynput)       ║    ║ + Cmd+Shift+J hotkey            ║
╚══════╤══════╝    ╚═══════╤════════╝    ╚══════════════╤══════════════════╝
       │                   │                             │
       └──────────────────┬┘                             │
                          ▼                             ▼
                 ┌─────────────────────────────────────────────┐
                 │           ALWAYS-ON VAD FILTER               │
                 │  webrtcvad (aggressiveness 0-3)              │
                 │  → filters silence before STT                │
                 │  → energy fallback if webrtcvad unavailable  │
                 └──────────────────┬──────────────────────────┘
                                    │ audio (float32, 16 kHz)
                                    ▼
                 ┌─────────────────────────────────────────────┐
                 │           WHISPER STT                        │
                 │  faster-whisper / small model               │
                 │  CoreML + Metal GPU acceleration (macOS)    │
                 │  Target latency: ~300ms                     │
                 └──────────────────┬──────────────────────────┘
                                    │ text
                                    ▼
                 ┌─────────────────────────────────────────────┐
                 │           LLM CORE                           │
                 │  Local Llama 3.2 via Ollama (streaming)     │
                 │  → Claude API fallback for complex queries   │
                 │  → tool dispatch via JSON function calling   │
                 │  Target first token: ~500ms                  │
                 └──────┬───────────────────────────────────────┘
                        │ token stream
                        ▼
          ┌─────────────────────────────────────┐
          │       SENTENCE CHUNKER               │
          │  splits at . ? ! boundaries          │
          │  (not word count)                    │
          └────────────────┬────────────────────┘
                           │ sentence chunks
                           ▼
          ┌─────────────────────────────────────┐
          │   ASYNC PRE-BUFFER QUEUE             │
          │  asyncio.Queue(maxsize=6)            │
          │  TTS chunk N+1 synthesised while     │
          │  chunk N is still playing            │
          └────────────────┬────────────────────┘
                           │ queued sentences
                           ▼
╔══════════════════════════╧═══════════════════════════════╗
║                    PIPER TTS                              ║
║  ONNX model, .npy speaker embeddings loaded at startup   ║
║  Personas: Dobby · Stewie Griffin · default (Jarvis)     ║
║  → edge-tts fallback (Microsoft neural voices)           ║
║  → macOS say as last resort                              ║
║  Target per-chunk: ~150ms                                ║
╚══════════════════════════╤═══════════════════════════════╝
                           │ WAV chunks
                           ▼
          ┌─────────────────────────────────────┐
          │       AUDIO OUTPUT                   │
          │  sounddevice (gapless, no gap        │
          │  between chunks)                     │
          │  → afplay fallback                   │
          └─────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│                    PERSONA EMBEDDING LAYER                                 │
│  .npy speaker embeddings loaded once at startup from jarvis/persona/       │
│  personas/stewie.npy · dobby.npy · default.npy                            │
│  Switchable at runtime: say "stewie mode" / "dobby mode" / "jarvis mode"  │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│           ✦ FULLY LOCAL INFERENCE — NO CLOUD DEPENDENCY ✦                 │
│           Apple M4 Mac Mini · 16 GB RAM · 512 GB SSD · macOS             │
└───────────────────────────────────────────────────────────────────────────┘

END-TO-END LATENCY ANALYSIS
┌──────────┬───────────────┬─────────────────┬──────────────────┬──────────────┐
│  Wake    │  STT          │  LLM First Token │  TTS per Chunk   │  TOTAL E2E   │
│  ~0ms    │  ~300ms       │  ~500ms          │  ~150ms          │  < 1.5s      │
└──────────┴───────────────┴─────────────────┴──────────────────┴──────────────┘
```

> The diagram above reflects the v2 async pipeline in [`jarvis/`](jarvis/). The legacy
> monolithic `run_jarvis.py` remains functional alongside it.

---

## Features

- **Persona system** — switch between Jarvis and Stewie Griffin mid-conversation (`stewie mode` / `jarvis mode`). Stewie has a mood engine that reacts to insults, compliments, and stupid questions.
- **Voice cloning** — XTTS v2 clones Stewie's voice from reference audio in `assets/Stewie/`. More clips = better quality.
- **Streaming STT** — faster-whisper runs locally with real-time VAD. No Google API, no cloud, responds as soon as you finish speaking.
- **Streaming LLM → TTS pipeline** — Ollama streams tokens directly into the XTTS synthesiser sentence by sentence. First audio plays before the full response is generated.
- **Pipelined synthesis** — silence trim of chunk N runs in parallel with synthesis of chunk N+1 via a thread pool.
- **Persistent memory** — conversation history and personal facts are saved across sessions.
- **Multi-agent research** — CrewAI crew for deep objectives, brainstorming with 4 specialist agents, live news via Serper.
- **Email mode** — polls IMAP, replies via SMTP, sends system alerts and check-ins.
- **Self-configuration** — change settings in plain English (`set CPU alert to 90%`, `disable checkins`).
- **Siri-style overlay HUD** — luminous blue orb that peeks up from the bottom of the screen (matching macOS Siri positioning), with distinct animations per state: idle dot → listening rings → thinking flicker → speaking beats.
- **Audio feedback** — `assets/ping.wav` plays on wake detection and when ending a session with "bye".
- **Thinking sound** — subtle low-volume ping loop plays while the LLM is generating a response so you know Jarvis is working.
- **Self-wake prevention** — Jarvis no longer re-triggers on its own TTS audio; wake event is cleared after each response plays.
- **Natural sleep phrases** — say "bye", "goodbye", "see you later", "that's all", "go to sleep", or "thanks bye" to end the session.

---

## Project Layout

### v2 modular structure (`jarvis/`)

```
jarvis/
├── main.py                    # Async entry point — wires all listeners + pipeline
├── config.yaml                # All settings (models, thresholds, paths, email)
├── pipeline/
│   ├── vad.py                 # webrtcvad silence filter
│   ├── stt.py                 # faster-whisper STT (CoreML/Metal)
│   ├── llm.py                 # Ollama streaming + Claude fallback + tool dispatch
│   ├── chunker.py             # Sentence boundary splitter
│   ├── tts.py                 # Piper TTS + .npy embeddings + edge-tts fallback
│   └── audio.py               # sounddevice gapless playback
├── tools/
│   ├── system.py              # shutdown, sleep, open_app, kill, stats
│   └── email_agent.py         # send, read, alerts, daily digest
├── persona/
│   ├── loader.py              # PersonaManager + ConversationMemory + mood engine
│   └── personas/              # stewie.npy  dobby.npy  default.npy
├── wake/
│   ├── porcupine.py           # Porcupine "Hey Jarvis" wake word
│   └── tap.py                 # Triple-tap (pynput) + Cmd+Shift+J hotkey
└── overlay/
    └── overlay.py             # Siri-style luminous blue orb HUD (idle/listening/thinking/speaking)
```

### Legacy v1 files (still functional)

```
Jarvis/
├── run_jarvis.py              # Monolithic orchestrator (v1 — preserved)
├── tts_server.py              # XTTS v2 voice-cloning subprocess server
├── email_handler.py           # IMAP / SMTP helpers
├── system_monitor.py          # psutil vitals
├── assets/Stewie/             # XTTS reference WAV clips
├── assets/ping.wav            # UI sound for wake detection and session end
├── memory/                    # Persistent conversation + user profile
├── src/my_log_agent/          # CrewAI multi-agent crew
└── .env                       # Legacy config (v1 reads this; v2 uses config.yaml)
```

---

## Requirements

- **Python 3.10–3.13** for the main environment (`uv` managed)
- **Python 3.11** venv at `tts_env/` for XTTS voice cloning
- **Ollama** running locally with at least one model
- **ffmpeg** — for silence trimming between TTS chunks (`brew install ffmpeg`)
- Optional: Serper API key for live news, email credentials for remote mode

---

## Installation

### 1. Main environment

```bash
uv sync
```

### 2. XTTS voice cloning environment (one-time)

```bash
python3.11 -m venv tts_env
tts_env/bin/pip install TTS
```

### 3. Pull an Ollama model

```bash
ollama pull llama3.2:3b      # recommended for fast chat
ollama pull lfm2.5-thinking  # for deep research / analysis
```

### 4. Add Stewie voice samples

Drop `.mp3` or `.wav` files into `assets/Stewie/`. Run this to convert any MP3s to WAV (required by XTTS):

```bash
cd "assets/Stewie"
for f in *.mp3; do ffmpeg -y -i "$f" -ar 22050 -ac 1 "${f%.mp3}.wav"; done
```

10–15 clean clips of 5–15 seconds each is the sweet spot. More than 20 gives diminishing returns.

---

## Configuration (`.env`)

```env
# Models
CHAT_MODEL=ollama/llama3.2:3b
ANALYSIS_MODEL=ollama/lfm2.5-thinking:latest

# Voice output
VOICE_OUTPUT_ENABLED=false        # set true or use --voice flag
VOICE_TTS_COMMAND=edge-tts        # fallback TTS when XTTS is not active
VOICE_TTS_VOICE=en-US-AndrewNeural

# Voice input (faster-whisper)
VOICE_INPUT_ENABLED=false         # set true or use --voice flag
WHISPER_MODEL_SIZE=base           # tiny / base / small / medium / large
VOICE_ENERGY_THRESHOLD=0.005      # mic sensitivity — raise if false triggers
VOICE_PAUSE_THRESHOLD=0.7         # seconds of silence before processing (0.5–1.0)
VOICE_PHRASE_TIME_LIMIT=12        # max recording length in seconds

# LLM tuning
OLLAMA_TEMP=0.8
OLLAMA_NUM_PREDICT=90
OLLAMA_KEEP_ALIVE=10m

# Email mode
EMAIL_USER=you@gmail.com
EMAIL_PASS=your_app_password
IMAP_SERVER=imap.gmail.com
SMTP_SERVER=smtp.gmail.com
OWNER_EMAIL=you@example.com

# Live news
SERPER_API_KEY=your_key

# System alerts
CPU_ALERT=80
RAM_ALERT=80
DISK_ALERT=80
ALERT_COOLDOWN_MIN=20

# Check-ins (email mode)
CHECKIN_ENABLED=false
CHECKIN_MIN_HOURS=2
CHECKIN_MAX_HOURS=4
```

---

## Running

### v2 (modular — recommended)

```bash
# Install dependencies
pip install -r requirements.txt

# Voice mode (wake word + overlay HUD)
python -m jarvis.main --voice

# Text-only REPL
python -m jarvis.main --text

# Email daemon
python -m jarvis.main --email

# Test the overlay animation standalone
python jarvis/overlay/overlay.py
```

**First-run checklist for v2:**
1. Edit `jarvis/config.yaml` — set your Piper model path, Porcupine access key
2. Copy credentials to `.env`: `EMAIL_USER`, `EMAIL_PASS`, `OWNER_EMAIL`
3. Download a Piper voice model: `en_US-ryan-medium.onnx` → `jarvis/persona/models/`
4. Get a free Porcupine key at [console.picovoice.ai](https://console.picovoice.ai/) and set `wake.porcupine_access_key`

### v1 legacy (still works)

```bash
uv run python run_jarvis.py --voice       # voice mode
uv run python run_jarvis.py               # text mode
uv run python run_jarvis.py --email       # email daemon
uv run python run_jarvis.py --hf          # HuggingFace cloud models
```

On first run, faster-whisper downloads the selected model (~74 MB for `base`). XTTS loads in ~30 seconds the first time.

---

## Voice Pipeline

```
Microphone
  └─ sounddevice (real-time blocks)
       └─ VAD (RMS energy gate)
            └─ faster-whisper (local Whisper transcription)
                 └─ Ollama streaming generation
                      └─ Sentence buffer
                           └─ XTTS v2 synthesis  ─── Silence trim (ffmpeg)
                                └─ afplay (macOS)
                                     └─ [next sentence already synthesised]
```

The first audio chunk plays as soon as the first sentence is generated — you do not wait for the full response.

---

## Persona Commands

| Say this | Effect |
|----------|--------|
| `stewie mode` | Switch to Stewie Griffin |
| `dobby mode` | Switch to Dobby the house-elf |
| `jarvis mode` | Switch back to Jarvis |
| `what persona` / `which mode` | Ask who is active |

Stewie's mood engine:
- Starts **happy** (smug, magnanimous)
- Flips to **annoyed** if you insult or demean him
- Flips to **exasperated** if you ask something very short or banal
- Returns to **happy** if you compliment him

Dobby:
- Loyal, enthusiastic, refers to himself in third person
- Calls you "Master" — deeply eager to help
- No mood engine — Dobby is always full of love and energy

---

## In-conversation Commands

| Input | What happens |
|-------|-------------|
| `status` / `vitals` / `health` | System CPU, RAM, disk report |
| `help` | Full capability guide |
| `brainstorm ...` | 4-agent brainstorm (Optimist, Critic, Engineer, Wildcard) |
| `news ...` | Live news briefing via Serper |
| `research ...` / `deep dive ...` | Multi-agent research crew |
| `set ...` | Natural-language config change |
| `exit` / `quit` | Shutdown |

---

## Adding More Voice Clips

1. Drop `.mp3` files into `assets/Stewie/`
2. Convert to WAV: `ffmpeg -y -i clip.mp3 -ar 22050 -ac 1 clip.wav`
3. Restart — clips are auto-discovered on startup

Tips for best quality:
- Use clean audio with no background music
- Aim for 8–15 seconds per clip
- Vary emotion: angry, sarcastic, cheerful, theatrical
- 10–15 good clips beats 30 noisy ones

---

## Overlay HUD

The overlay is a macOS-native Cocoa window (PyObjC) that renders a luminous blue orb peeking up from the bottom of the screen — matching the macOS Siri aesthetic.

**States and animations:**

| State | Visual |
|-------|--------|
| `IDLE` | Tiny dim dot — almost invisible |
| `LISTENING` | Expanding concentric rings radiating outward |
| `THINKING` | Rapid brightness flicker while the LLM generates |
| `SPEAKING` | Four beat-rings that pulse outward in sync with speech |

**Implementation notes:**
- Runs in a separate `multiprocessing.Process` — completely isolated from the main async loop
- State changes are sent via `multiprocessing.Queue`; the overlay polls the queue in a background thread using `queue.get(timeout=0.5)` (not `queue.empty()`, which is unreliable across process boundaries)
- Window is positioned at y=0 (screen bottom), with the orb center near the bottom edge so only the upper arc of the orb is visible — identical to how macOS Siri appears
- Window level 1000 (NSScreenSaverWindowLevel) keeps it on top of all other windows and spaces
- Automatically dies when the parent process exits (parent-death detection via `os.getppid()`)
- Graceful shutdown: `overlay.stop()` sends a `"quit"` message; falls back to `SIGKILL` if the process doesn't exit within 2 seconds

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| XTTS not loading | Check `tts_env/bin/python` exists; run `python3.11 -m venv tts_env && tts_env/bin/pip install TTS` |
| No mic detected | System Settings → Privacy → Microphone → enable Terminal |
| Voice too quiet / not triggering | Lower `VOICE_ENERGY_THRESHOLD` in `.env` (try `0.003`) |
| Sentences cut off | Raise `VOICE_PAUSE_THRESHOLD` (try `0.9`) |
| Ollama not responding | Run `ollama serve` and confirm model is pulled |
| Email mode silent | Check IMAP/SMTP credentials and app password in `.env` |
| News fetch fails | Confirm `SERPER_API_KEY` is set |
| Overlay not visible | Confirm you are in `--voice` mode; the HUD only shows in voice pipeline sessions |
| Overlay stays on screen after Ctrl+C | Fixed — parent-death detection kills the overlay process automatically |
| Jarvis re-triggers on its own voice | Fixed — wake event is cleared after each TTS response plays |
| "Bye" not stopping the session | Fixed — natural language sleep phrases are matched with a regex pattern |
| No ping sound on wake | Ensure `assets/ping.wav` exists; it's played via `afplay` (not through the TTS system) |
