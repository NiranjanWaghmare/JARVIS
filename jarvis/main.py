"""
Jarvis v2 — async entry point.

Run modes:
  python -m jarvis.main              # interactive voice+text loop
  python -m jarvis.main --voice      # force voice I/O
  python -m jarvis.main --text       # force text-only
  python -m jarvis.main --email      # remote email daemon
"""

import asyncio
import logging
import random
import re
import signal
import sys
import threading
from pathlib import Path

import yaml
from dotenv import load_dotenv

# ── bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()

_CFG_PATH   = Path(__file__).parent / "config.yaml"
_ASSETS_DIR = Path(__file__).parent.parent / "assets"
_PING_WAV   = str(_ASSETS_DIR / "ping.wav")
with open(_CFG_PATH) as _f:
    CFG: dict = yaml.safe_load(_f)

_log_cfg = CFG.get("logging", {})
logging.basicConfig(
    level=getattr(logging, _log_cfg.get("level", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_log_cfg.get("log_file", "jarvis.log"), encoding="utf-8"),
    ],
)
log = logging.getLogger("jarvis.main")

# ── lazy imports (each module degrades gracefully if deps missing) ─────────────
from jarvis.pipeline.vad import VADRecorder
from jarvis.pipeline.stt import STTEngine
from jarvis.pipeline.llm import LLMCore
from jarvis.pipeline.chunker import SentenceChunker
from jarvis.pipeline.tts import TTSEngine
from jarvis.pipeline.audio import AudioPlayer
from jarvis.persona.loader import PersonaManager
from jarvis.tools.system import SystemTools, get_system_stats
from jarvis.tools.email_agent import EmailAgent
from jarvis.agents.browser_agent import BrowserAgent, BROWSER_TRIGGERS
from jarvis.agents.critique_agent import background_tasks as _bg_tasks, RUN_EVERY_N_TURNS
from jarvis.overlay.overlay import OverlayController, IDLE, SESSION, LISTENING, THINKING, SPEAKING
from jarvis.tools.audio_device import AudioDeviceManager, patch_vad_for_device_switching


# ── voice pipeline ─────────────────────────────────────────────────────────────
async def run_voice_pipeline(
    text: str,
    persona: PersonaManager,
    llm: LLMCore,
    tts: TTSEngine,
    player: AudioPlayer,
    overlay: OverlayController | None = None,
) -> str:
    """LLM → TTS → playback.

    Fast path (Piper installed): streams LLM tokens into sentence chunks and
    synthesises each chunk while the next is being generated.

    Batch path (edge-tts / say): collects the full LLM response first, then
    calls TTS once — guarantees a single consistent voice for the whole reply.
    """
    loop = asyncio.get_event_loop()

    if overlay:
        overlay.set_state(THINKING)
    _start_thinking_sound()

    # ── Batch path (edge-tts / say) ────────────────────────────────────────────
    if not tts.fast:
        tokens: list[str] = []
        async for token in llm.stream(text, persona):
            tokens.append(token)
        reply = "".join(tokens)
        print(f"\n[{persona.display}] {reply}\n")
        _stop_thinking_sound()
        if overlay:
            overlay.set_state(SPEAKING)
        wav = await loop.run_in_executor(None, tts.synthesize, reply, persona)
        if wav:
            await loop.run_in_executor(None, player.play, wav)
        if overlay:
            overlay.set_state(IDLE)
        return reply

    # ── Streaming path (Piper) ─────────────────────────────────────────────────
    sentence_q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=6)
    wav_q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=3)
    full_tokens: list[str] = []

    async def _llm_stage():
        chunker = SentenceChunker()
        async for token in llm.stream(text, persona):
            full_tokens.append(token)
            for sentence in chunker.push(token):
                await sentence_q.put(sentence)
        for sentence in chunker.flush():
            await sentence_q.put(sentence)
        print(f"\n[{persona.display}] {''.join(full_tokens)}\n")
        await sentence_q.put(None)

    async def _tts_stage():
        while True:
            sentence = await sentence_q.get()
            if sentence is None:
                await wav_q.put(None)
                return
            try:
                wav_path = await loop.run_in_executor(None, tts.synthesize, sentence, persona)
                if wav_path:
                    await wav_q.put(wav_path)
            except Exception as exc:
                log.warning("TTS failed for chunk: %s", exc)

    async def _play_stage():
        first = True
        while True:
            wav = await wav_q.get()
            if wav is None:
                return
            if first:
                _stop_thinking_sound()
                if overlay:
                    overlay.set_state(SPEAKING)
                first = False
            await loop.run_in_executor(None, player.play, wav)

    await asyncio.gather(_llm_stage(), _tts_stage(), _play_stage())
    if overlay:
        overlay.set_state(IDLE)
    return "".join(full_tokens)


# ── single-turn processing ─────────────────────────────────────────────────────
async def process_turn(
    text: str,
    *,
    persona: PersonaManager,
    llm: LLMCore,
    tts: TTSEngine,
    player: AudioPlayer,
    email_agent: EmailAgent | None,
    sys_tools: SystemTools,
    browser: BrowserAgent | None = None,
    voice_out: bool,
    overlay: OverlayController | None = None,
    confirm_cb=None,
) -> str:
    text = text.strip()
    if not text:
        return ""

    log.info("User: %s", text)
    persona.update_mood(text)

    # Forget command — "forget about X" / "forget the X conversation"
    _forget_m = re.search(
        r'\bforget\b(?:\s+about)?\s+(?:the\s+)?(.+?)(?:\s+conversation)?$',
        text, re.I
    )
    if _forget_m:
        topic = _forget_m.group(1).strip().rstrip(".,!?")
        removed = persona.memory.forget_about(topic)
        reply = (
            f"Done. I've cleared {removed} message{'s' if removed != 1 else ''} "
            f"about '{topic}' from my memory."
            if removed else
            f"I don't have anything about '{topic}' in my memory."
        )
        if voice_out:
            wav = await asyncio.get_event_loop().run_in_executor(None, tts.synthesize, reply, persona)
            if wav:
                player.play(wav)
        return reply

    # Persona switch commands
    switch = persona.try_switch(text)
    if switch:
        reply = switch
        if voice_out:
            wav = await asyncio.get_event_loop().run_in_executor(None, tts.synthesize, reply, persona)
            if wav:
                player.play(wav)
        return reply

    # System tool dispatch — pass recent session context so agents can resolve "take me there" etc.
    tool_reply = await sys_tools.dispatch(
        text,
        confirm_cb=confirm_cb or _confirm_verbal,
        memory_context=persona.memory.get_context(3),
    )
    if tool_reply is not None:
        if voice_out and tool_reply:
            wav = await asyncio.get_event_loop().run_in_executor(None, tts.synthesize, tool_reply, persona)
            if wav:
                player.play(wav)
        return tool_reply

    # Browser interaction — detected before hitting the main LLM
    if browser and BROWSER_TRIGGERS.search(text):
        if overlay:
            overlay.set_state(THINKING)
        context = persona.memory.get_context(3)
        reply = await browser.execute_task(text, context, llm)
        if reply:
            print(f"\n[{persona.display}] {reply}\n")
            if voice_out:
                if overlay:
                    overlay.set_state(SPEAKING)
                wav = await asyncio.get_event_loop().run_in_executor(None, tts.synthesize, reply, persona)
                if wav:
                    await asyncio.get_event_loop().run_in_executor(None, player.play, wav)
        persona.memory.add_turn(text, reply)
        asyncio.create_task(_bg_tasks(text, reply, llm, persona))
        return reply

    # Main LLM path
    if voice_out:
        reply = await run_voice_pipeline(text, persona, llm, tts, player, overlay=overlay)
    else:
        chunks = []
        async for token in llm.stream(text, persona):
            chunks.append(token)
        reply = "".join(chunks)
        print(f"\n[{persona.display}] {reply}\n")

    persona.memory.add_turn(text, reply)
    # Background tasks fire every Nth turn to avoid Ollama contention.
    # They run sequentially inside _bg_tasks, delayed by STARTUP_DELAY_SEC.
    global _turn_counter
    _turn_counter += 1
    if _turn_counter % RUN_EVERY_N_TURNS == 0:
        asyncio.create_task(_bg_tasks(text, reply, llm, persona))
    return reply


def _play_sound_fx(path: str):
    """Play a WAV file non-blocking and without deleting it (afplay on macOS)."""
    import subprocess
    try:
        subprocess.Popen(
            ["afplay", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


_thinking_stop = threading.Event()

def _start_thinking_sound():
    """Soft repeating ping while the LLM is generating a response."""
    _thinking_stop.clear()
    def _loop():
        import subprocess
        while not _thinking_stop.wait(timeout=2.2):
            try:
                subprocess.Popen(
                    ["afplay", "-v", "0.18", _PING_WAV],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                break
    threading.Thread(target=_loop, daemon=True, name="thinking-sound").start()

def _stop_thinking_sound():
    _thinking_stop.set()


_turn_counter: int = 0   # throttle background LLM tasks to every Nth turn


async def _confirm_verbal(prompt: str) -> bool:
    """Text-mode confirmation — reads from stdin."""
    print(f"\n[CONFIRM] {prompt} (yes/no): ", end="", flush=True)
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(None, input, "")
    return answer.strip().lower() in {"yes", "y", "ok", "confirm", "sure"}


_YES = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "confirm",
        "do it", "go ahead", "go for it", "affirmative"}
_NO  = {"no", "nope", "cancel", "stop", "don't", "abort", "negative"}


def _make_voice_confirm(tts, player, vad, stt, persona, overlay=None):
    """Returns a voice confirmation callback that speaks the prompt and listens for yes/no."""
    async def _voice_confirm(prompt: str) -> bool:
        loop = asyncio.get_event_loop()
        action = prompt.rstrip("?").strip().lower()
        question = f"Should I {action}? Can you please confirm, Sir."
        print(f"\n[CONFIRM] {question}")
        wav = await loop.run_in_executor(None, tts.synthesize, question, persona)
        if wav:
            await loop.run_in_executor(None, player.play, wav)
        if overlay:
            overlay.set_state(LISTENING)
        audio = await loop.run_in_executor(None, vad.record)
        if overlay:
            overlay.set_state(THINKING)
        if audio is None or len(audio) == 0:
            return False
        text = await loop.run_in_executor(None, stt.transcribe, audio)
        if not text:
            return False
        words = set(text.lower().split())
        print(f"[CONFIRM heard] {text}")
        if words & _YES:
            return True
        if words & _NO:
            return False
        return False   # unrecognised — treat as cancel
    return _voice_confirm


# ── wake-trigger listener ──────────────────────────────────────────────────────
async def _start_wake_listeners(wake_event: asyncio.Event):
    """Start OpenWakeWord, Porcupine, triple-tap, and hotkey listeners in daemon threads."""
    import threading
    from jarvis.wake.openwakeword_wake import OpenWakeWordWake
    from jarvis.wake.porcupine import PorcupineWake
    from jarvis.wake.tap import TapWake

    loop = asyncio.get_event_loop()

    def _fire():
        loop.call_soon_threadsafe(wake_event.set)

    # OpenWakeWord — always-on "hey Jarvis", no API key needed
    oww = OpenWakeWordWake(CFG, on_wake=_fire)
    if oww.available:
        threading.Thread(target=oww.run, daemon=True, name="oww-wake").start()
        log.info("OpenWakeWord wake listener active — say 'hey Jarvis'.")
    else:
        log.warning("OpenWakeWord not available — install: uv pip install openwakeword")

    # Porcupine — higher accuracy, requires API key
    porcupine = PorcupineWake(CFG, on_wake=_fire)
    if porcupine.available:
        threading.Thread(target=porcupine.run, daemon=True).start()
        log.info("Porcupine wake word active.")

    tap = TapWake(CFG, on_wake=_fire)
    threading.Thread(target=tap.run, daemon=True).start()
    log.info("Triple-tap and hotkey listeners active.")


# Phrases that end an active session and send Jarvis back to sleep
_SLEEP_PHRASES = {
    "bye", "bye for now", "goodbye", "good bye", "see you", "see you later",
    "that's all", "thats all", "go to sleep", "sleep", "thanks bye",
    "stop listening", "later", "peace out",
}

# For natural sign-offs that include filler words, search anywhere in short utterances
_SLEEP_PATTERN = re.compile(
    r'\b(bye|goodbye|good\s+bye|see\s+you(\s+later)?|'
    r"that'?s\s+all|go\s+to\s+sleep|thanks\s+bye|"
    r'stop\s+listening|peace\s+out)\b',
    re.IGNORECASE,
)

def _is_sleep_phrase(text: str) -> bool:
    t = text.lower().strip().rstrip(".,!?")
    if t in _SLEEP_PHRASES or any(t.startswith(p) for p in _SLEEP_PHRASES):
        return True
    # "All right. Bye, Jarvis." / "Ok bye" / "alright see you" — short sign-offs
    if len(t.split()) <= 8:
        return bool(_SLEEP_PATTERN.search(t))
    return False

# Phrases that fully shut down the process
_QUIT_PHRASES = {"exit", "quit", "shutdown", "shut down"}


# ── interactive voice loop ─────────────────────────────────────────────────────
async def voice_loop(
    persona: PersonaManager,
    llm: LLMCore,
    tts: TTSEngine,
    player: AudioPlayer,
    vad: VADRecorder,
    stt: STTEngine,
    sys_tools: SystemTools,
    email_agent: EmailAgent | None,
    browser: BrowserAgent | None = None,
    overlay: OverlayController | None = None,
):
    wake_event = asyncio.Event()
    await _start_wake_listeners(wake_event)

    loop = asyncio.get_event_loop()
    typed_input: list[str] = []  # shared: stdin watcher deposits typed text here

    def _stdin_watcher():
        print("(Type a message + Enter, or press Enter alone to speak via mic)\n")
        while True:
            try:
                line = sys.stdin.readline()
                if line is None:
                    break
                line = line.strip()
                if line:
                    typed_input.clear()
                    typed_input.append(line)
                loop.call_soon_threadsafe(wake_event.set)
            except Exception:
                break

    import threading
    threading.Thread(target=_stdin_watcher, daemon=True).start()

    # Greet
    greeting = persona.greeting
    log.info("[%s] %s", persona.display, greeting)
    print(f"\n[{persona.display}] {greeting}\n")
    if overlay:
        overlay.set_state(SPEAKING)
    wav = await loop.run_in_executor(None, tts.synthesize, greeting, persona)
    if wav:
        await loop.run_in_executor(None, player.play, wav)
    if overlay:
        overlay.set_state(IDLE)
    # Prevent greeting audio from re-triggering wake word detector
    await asyncio.sleep(1.0)
    wake_event.clear()

    in_session = False  # True while a conversation is active (no wake word needed)

    while True:
        if not in_session:
            log.info("Sleeping — waiting for wake trigger…")
            await wake_event.wait()
            wake_event.clear()

            # Ping + verbal acknowledgment so the wake feels connected
            _play_sound_fx(_PING_WAV)
            _WAKE_ACKS = [
                "Yes, sir?",
                "How can I help?",
                "At your service.",
                "I'm listening.",
                "What can I do for you?",
                "Go ahead, sir.",
            ]
            ack = random.choice(_WAKE_ACKS)
            print(f"[{persona.display}] {ack}")
            if overlay:
                overlay.set_state(SPEAKING)
            _ack_wav = await loop.run_in_executor(None, tts.synthesize, ack, persona)
            if _ack_wav:
                await loop.run_in_executor(None, player.play, _ack_wav)

            # Flush stray keystrokes (e.g. 'ttt' from triple-tap) from stdin
            try:
                import termios
                termios.tcflush(sys.stdin, termios.TCIFLUSH)
            except Exception:
                pass

        # Typed input takes priority — no mic needed
        if typed_input:
            text = typed_input.pop(0)
            print(f"[You] {text}")
        else:
            if overlay:
                overlay.set_state(LISTENING)
            if not in_session:
                log.info("Wake detected — recording…")
                in_session = True  # session starts on first wake
            print("[Listening…]")
            audio = await loop.run_in_executor(None, vad.record)
            if overlay:
                overlay.set_state(THINKING)  # stay visible while STT transcribes

            if audio is None or len(audio) == 0:
                print("[No audio captured]")
                continue

            text = await loop.run_in_executor(None, stt.transcribe, audio)
            if not text:
                print("[Could not understand — please repeat]")
                if overlay:
                    overlay.set_state(SESSION if in_session else IDLE)
                continue
            print(f"[Heard] {text}")

        # Full shutdown
        if text.lower().strip() in _QUIT_PHRASES:
            farewell = "Shutting down. Goodbye."
            print(f"[{persona.display}] {farewell}")
            _play_sound_fx(_PING_WAV)
            if overlay:
                overlay.set_state(SPEAKING)
            wav = await loop.run_in_executor(None, tts.synthesize, farewell, persona)
            if wav:
                await loop.run_in_executor(None, player.play, wav)
            break

        # Sleep phrase — end session, go back to waiting for wake word
        if _is_sleep_phrase(text):
            reply = "Okay, going to sleep."
            print(f"[{persona.display}] {reply}")
            _play_sound_fx(_PING_WAV)
            if overlay:
                overlay.set_state(SPEAKING)
            wav = await loop.run_in_executor(None, tts.synthesize, reply, persona)
            if wav:
                await loop.run_in_executor(None, player.play, wav)
            if overlay:
                overlay.set_state(IDLE)
            in_session = False
            log.info("Session ended — returning to sleep.")
            # Flush any wake events triggered by Jarvis's own voice
            await asyncio.sleep(1.2)
            wake_event.clear()
            continue

        # Start / continue session
        in_session = True
        if overlay:
            overlay.set_state(SESSION)
        await process_turn(
            text,
            persona=persona,
            llm=llm,
            tts=tts,
            player=player,
            email_agent=email_agent,
            sys_tools=sys_tools,
            browser=browser,
            voice_out=True,
            overlay=overlay,
            confirm_cb=_make_voice_confirm(tts, player, vad, stt, persona, overlay),
        )
        # After reply, show session-ready indicator while waiting for next input
        if overlay:
            overlay.set_state(SESSION)
        # Cooldown so the mic doesn't pick up speaker echo from Jarvis's reply
        await asyncio.sleep(0.8)


# ── text-only interactive loop ─────────────────────────────────────────────────
async def text_loop(
    persona: PersonaManager,
    llm: LLMCore,
    tts: TTSEngine,
    player: AudioPlayer,
    sys_tools: SystemTools,
    email_agent: EmailAgent | None,
    browser: BrowserAgent | None = None,
):
    loop = asyncio.get_event_loop()
    print(f"\n--- {persona.display.upper()} ONLINE ---")
    print("Type your message. 'exit' to quit.\n")

    while True:
        try:
            user_input = await loop.run_in_executor(None, input, f"[{persona.user_label}] ")
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in {"exit", "quit", "stop"}:
            print(f"[{persona.display}] Goodbye.")
            break

        await process_turn(
            user_input,
            persona=persona,
            llm=llm,
            tts=tts,
            player=player,
            email_agent=email_agent,
            sys_tools=sys_tools,
            browser=browser,
            voice_out=False,
        )


# ── email daemon loop ──────────────────────────────────────────────────────────
async def email_daemon(
    persona: PersonaManager,
    llm: LLMCore,
    email_agent: EmailAgent,
    sys_tools: SystemTools,
):
    log.info("Email daemon started. Polling every %ss.", CFG["email"]["poll_interval_sec"])
    poll_count = 0

    # Startup greeting
    from jarvis.tools.system import get_system_stats
    stats = get_system_stats()
    startup_body = f"Jarvis online.\n\nSystem Status:\n{stats}"
    await asyncio.get_event_loop().run_in_executor(
        None, email_agent.send, email_agent.owner_email, "Jarvis Startup", startup_body
    )

    while True:
        poll_count += 1
        task = await asyncio.get_event_loop().run_in_executor(None, email_agent.check_inbox)
        if task:
            log.info("Email task received: %s", task["objective"])
            reply = await process_turn(
                task["body"] or task["objective"],
                persona=persona,
                llm=llm,
                tts=None,
                player=None,
                email_agent=email_agent,
                sys_tools=sys_tools,
                voice_out=False,
            )
            if reply:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    email_agent.reply,
                    task["sender"],
                    task["objective"],
                    reply,
                    task.get("message_id"),
                )

        # System monitor every 5 cycles
        if poll_count % CFG["monitor"]["check_interval_cycles"] == 0:
            await asyncio.get_event_loop().run_in_executor(None, email_agent.check_alerts, llm)

        await asyncio.sleep(CFG["email"]["poll_interval_sec"])


# ── main ───────────────────────────────────────────────────────────────────────
async def _main():
    args = sys.argv[1:]
    mode_email = "--email" in args or "-email" in args
    mode_voice = "--voice" in args
    mode_text  = "--text"  in args

    # Default: text mode. Use --voice explicitly to enable wake-word pipeline.
    if not mode_email and not mode_voice and not mode_text:
        mode_text = True

    log.info("Initialising Jarvis v2…")

    persona = PersonaManager(CFG)
    llm     = LLMCore(CFG)
    tts     = TTSEngine(CFG)
    player  = AudioPlayer(CFG)
    sys_tools = SystemTools(CFG)

    # Give SystemTools access to LLM so agents can do intelligent resolution
    sys_tools.set_llm(llm)

    # Wire system tools into LLM's JSON tool-call registry.
    # NOTE: open_url is intentionally NOT registered here — the LLM must never
    # proactively open websites. URL opening only happens via the dispatch layer
    # when the user explicitly requests it ("take me there", "open that").
    llm.register_tool("open_app", sys_tools.open_app)
    llm.register_tool("system_sleep", sys_tools.system_sleep)
    llm.register_tool("system_shutdown", sys_tools.system_shutdown)
    llm.register_tool("system_restart", sys_tools.system_restart)
    llm.register_tool("get_stats", lambda: get_system_stats())
    llm.register_tool("set_reminder", sys_tools.set_reminder)

    # Browser agent — lazy-started on first use
    browser = BrowserAgent()

    # Email agent (always available for process_turn)
    email_agent: EmailAgent | None = None
    try:
        email_agent = EmailAgent(CFG)
    except Exception as exc:
        log.warning("Email agent unavailable: %s", exc)

    if mode_email:
        if email_agent is None:
            log.error("Email credentials not configured. Set EMAIL_USER/EMAIL_PASS in .env")
            return
        await email_daemon(persona, llm, email_agent, sys_tools)
        return

    if mode_voice:
        try:
            vad = VADRecorder(CFG)
            stt = STTEngine(CFG)
            # Pre-warm Whisper in background so first query has no load delay
            import threading as _t
            _t.Thread(target=stt._load, daemon=True, name="whisper-prewarm").start()
            overlay = OverlayController(CFG)
            overlay.start()
            _overlay_ref[0] = overlay
            # Patch VAD for runtime device switching, then expose AudioDeviceManager
            patch_vad_for_device_switching(vad)
            audio_mgr = AudioDeviceManager(vad, player)
            sys_tools.set_audio_manager(audio_mgr)
            await voice_loop(persona, llm, tts, player, vad, stt, sys_tools, email_agent, browser=browser, overlay=overlay)
            overlay.stop()
            _overlay_ref[0] = None
        except ImportError as exc:
            log.warning("Voice deps missing (%s) — falling back to text mode.", exc)
            mode_text = True
        finally:
            if browser.is_open:
                await browser.stop()

    if mode_text or not mode_voice:
        await text_loop(persona, llm, tts, player, sys_tools, email_agent, browser=browser)
        if browser.is_open:
            await browser.stop()


_overlay_ref: list = [None]  # set in _main() so _shutdown can clean it up


def main():
    import os
    import threading

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _called = [0]

    def _kill_overlay():
        ov = _overlay_ref[0]
        if ov is not None:
            try:
                ov.stop()
            except Exception:
                pass

    def _shutdown(sig, frame):
        _called[0] += 1
        if _called[0] == 1:
            log.info("Shutting down… (Ctrl+C again to force quit)")
            for task in asyncio.all_tasks(loop):
                task.cancel()
            loop.call_soon_threadsafe(loop.stop)
            def _force_exit():
                _kill_overlay()
                os._exit(0)
            # Force-exit after 1.5 s if still alive (audio threads can block cleanup)
            threading.Timer(1.5, _force_exit).start()
        else:
            log.info("Force quit.")
            _kill_overlay()
            os._exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        loop.run_until_complete(_main())
    except (KeyboardInterrupt, asyncio.CancelledError, RuntimeError):
        pass
    finally:
        try:
            loop.close()
        except Exception:
            pass
        log.info("Jarvis offline.")


if __name__ == "__main__":
    main()
