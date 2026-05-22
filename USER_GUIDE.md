# Jarvis — User Guide

---

## Starting Jarvis

Before launching, make sure Ollama is running in a separate terminal:

```bash
ollama serve
```

Then start Jarvis in one of three modes:

| Command | Mode | Use when |
|---------|------|----------|
| `python -m jarvis.main --voice` | Full voice mode | You want to speak to Jarvis with wake word + overlay HUD |
| `python -m jarvis.main --text` | Text-only REPL | No microphone, or just want to type |
| `python -m jarvis.main --email` | Email daemon | Running Jarvis headless / remotely via email |

> Run from the project root (`/path/to/Jarvis`). All three modes share the same LLM, memory, and persona system.

---

## Waking Jarvis

In voice mode, Jarvis waits silently until triggered. Three ways to wake it:

| Method | How |
|--------|-----|
| Voice | Say **"Hey Jarvis"** |
| Keyboard | Triple-tap **T** quickly (within 1 second) |
| Hotkey | **Cmd + Shift + J** from anywhere |

The overlay ring pulses when Jarvis is listening. Speak your request, then pause — it stops recording when you go quiet.

---

## Talking to Jarvis

Just speak or type naturally. No keywords required for general conversation.

```
You:    What's the capital of Iceland?
Jarvis: Reykjavik. It's also the world's northernmost capital city.

You:    Tell me something interesting about it.
Jarvis: Reykjavik runs almost entirely on geothermal energy...
```

Jarvis remembers the last 6 turns of your conversation and builds a profile of you over time (your name, preferences, things you've mentioned).

---

## System Control

Jarvis can control your Mac. Destructive actions (shutdown, kill) always ask for verbal confirmation first.

| Say this | What happens |
|----------|-------------|
| `open Spotify` | Opens Spotify |
| `open Terminal` | Opens Terminal |
| `open [any app name]` | Launches that application |
| `kill Chrome` | Asks confirmation, then kills Chrome |
| `kill [process name]` | Terminates that process |
| `sleep` | Puts the Mac to sleep immediately |
| `restart` | Asks confirmation, then restarts |
| `shutdown` / `power off` | Asks confirmation, then shuts down |

---

## System Stats

| Say this | What you get |
|----------|-------------|
| `vitals` / `health` / `status` | CPU %, RAM %, disk usage, uptime, top 3 processes |
| `show me the top processes` | Top 10 processes ranked by CPU usage |
| `how's the CPU` / `RAM usage` | Included in the vitals readout |

Example output:
```
CPU:    14.2%
RAM:    61.3%  (9.8/16.0 GB)
Disk:   43.0%  (220/512 GB)
Uptime: 2 days, 4:31:12
Top:    Xcode(8%), Slack(3%), Chrome(2%)
```

---

## Research & Brainstorming

### Brainstorm
Spins up 4 specialist agents — Optimist, Critic, Engineer, Wildcard — and returns a structured breakdown.

```
brainstorm a subscription model for a local coffee shop
brainstorm how to reduce latency in a Python web server
```

### Deep Research
Full multi-agent research crew. Takes longer but goes deeper.

```
research the history of large language models
deep dive into fusion energy progress in 2024
```

### Live News
Fetches real-time articles via Serper. Requires `SERPER_API_KEY` in `.env`.

```
news about climate policy
latest news on Apple
```

---

## Personas

Switch Jarvis's personality mid-conversation at any time.

| Say this | Who you get |
|----------|------------|
| `jarvis mode` | Default — calm, precise, British |
| `stewie mode` | Stewie Griffin — theatrical, sarcastic, occasionally withering |
| `dobby mode` | Dobby — eager, earnest, slightly anxious |
| `what persona are you` | Tells you who is currently active |

### Stewie's Mood Engine

Stewie starts happy but his mood shifts based on how you treat him:

| You do this | Stewie becomes |
|-------------|----------------|
| Insult or demean him | Annoyed — sharp, clipped responses |
| Compliment him | Happy — warm, magnanimous |
| Ask something very short or dull | Exasperated — theatrical sighs |

---

## Memory

Jarvis remembers things across sessions.

- **Conversation history** — last 6 turns used as context for every response
- **User profile** — facts you've mentioned ("I'm a developer", "I prefer dark mode") are saved and referenced in future conversations
- **Persona isolation** — Stewie and Jarvis maintain separate conversation histories

You don't need any special command. Just talk normally and Jarvis builds the profile automatically.

---

## Email Mode

Run with `--email` to start the email daemon. Jarvis polls your inbox every 30 seconds.

**What it does:**
- Reads emails from your `OWNER_EMAIL` address only
- Replies automatically using the LLM
- Sends a **daily digest** at 8am with system stats and a summary
- Fires **alerts** if CPU, RAM, or disk cross their thresholds

**Trigger Jarvis remotely by emailing yourself:**
```
Subject: open safari
Subject: what are the system stats
Subject: shutdown
```

Jarvis replies with the result.

---

## Overlay HUD

A small animated overlay appears in voice mode showing what Jarvis is doing.

| Animation | State | Meaning |
|-----------|-------|---------|
| Pulsing concentric rings | Listening | Waiting for your voice |
| Spinning dots | Thinking | LLM is generating |
| Equaliser bars | Speaking | Playing the response |
| No animation | Idle | Waiting to be woken |

---

## Tips

**First audio plays fast.** Jarvis streams tokens into the TTS engine sentence by sentence, so you hear the first sentence before the full response is generated.

**Interrupt anytime.** Wake Jarvis again (triple-tap or hotkey) while it's speaking to cut it off and ask something new.

**Long responses are chunked.** Each sentence is synthesised and played individually with no gap between them.

**Piper voice sounds best.** If you hear a robotic fallback voice, check that the Piper model file is in `jarvis/persona/models/` and the path in `config.yaml` matches.

**Adjust sensitivity if it mishears you.** In `jarvis/config.yaml`:
```yaml
vad:
  energy_floor: 0.002     # lower = more sensitive mic pickup
  pause_threshold_sec: 0.9  # higher = waits longer before cutting off
```
