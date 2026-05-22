"""
TTS Engine — Piper TTS with .npy speaker embeddings, pre-buffer queue,
with edge-tts and macOS `say` as fallback tiers.

Piper synthesis model path is set in config.yaml → tts.piper_model.
Speaker embeddings are .npy files in jarvis/persona/personas/.
"""

import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from jarvis.persona.loader import PersonaManager

log = logging.getLogger("jarvis.tts")

_PIPER_OK = False
try:
    from piper import PiperVoice  # type: ignore
    _PIPER_OK = True
except ImportError:
    log.warning("piper-tts not installed — using edge-tts/say fallback. "
                "Install: pip install piper-tts")

_FFMPEG_TRIM = (
    "silenceremove=start_periods=1:start_silence=0.02:start_threshold=-30dB"
    ",areverse"
    ",silenceremove=start_periods=1:start_silence=0.02:start_threshold=-30dB"
    ",areverse"
)

# max chars Piper handles well in a single call
_PIPER_MAX = 500


# Path to the Python interpreter that has TTS/XTTS installed
_XTTS_PYTHON = str(Path(__file__).parents[2] / "tts_env" / "bin" / "python")
_XTTS_SERVER = str(Path(__file__).parents[2] / "tts_server.py")


class _XTTSClient:
    """Persistent subprocess wrapper for tts_server.py (XTTS voice cloning)."""

    def __init__(self, ref_wavs: list[str]):
        self._ref_wavs = ref_wavs
        self._proc: Optional[subprocess.Popen] = None

    def _start(self):
        if self._proc and self._proc.poll() is None:
            return  # already running
        log.info("Starting XTTS server with %d reference clips…", len(self._ref_wavs))
        self._proc = subprocess.Popen(
            [_XTTS_PYTHON, _XTTS_SERVER] + self._ref_wavs,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        # Wait for READY signal (may take ~10-20s on first run to load model)
        for line in self._proc.stdout:
            line = line.strip()
            log.debug("XTTS server: %s", line)
            if line == "READY":
                log.info("XTTS server ready.")
                return
        raise RuntimeError("XTTS server failed to start.")

    def synthesize(self, text: str) -> Optional[str]:
        try:
            self._start()
        except Exception as exc:
            log.error("XTTS server start failed: %s", exc)
            return None

        out = tempfile.mktemp(suffix=".wav")
        req = json.dumps({"text": text, "output": out})
        try:
            self._proc.stdin.write(req + "\n")
            self._proc.stdin.flush()
            resp_line = self._proc.stdout.readline()
            resp = json.loads(resp_line)
            if resp.get("status") == "ok":
                return resp["file"]
            log.warning("XTTS error: %s", resp.get("message"))
        except Exception as exc:
            log.error("XTTS synthesis failed: %s", exc)
            self._proc = None  # force restart next time
        return None

    def stop(self):
        if self._proc:
            try:
                self._proc.stdin.close()
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None


class TTSEngine:
    def __init__(self, cfg: dict):
        tts_cfg = cfg.get("tts", {})
        self.engine: str         = tts_cfg.get("engine", "piper")
        self.piper_model: str    = tts_cfg.get("piper_model", "")
        self.piper_config: str   = tts_cfg.get("piper_config", "")
        self.edge_voice: str     = tts_cfg.get("edge_tts_voice", "en-GB-RyanNeural")
        self.edge_rate: str      = tts_cfg.get("edge_tts_rate", "-5%")
        self.trim_silence: bool  = tts_cfg.get("silence_trim", True)
        self.ffmpeg: str         = tts_cfg.get("ffmpeg", "ffmpeg")

        self._piper_voice: Optional["PiperVoice"] = None  # lazy-loaded once
        self._xtts_clients: dict[str, _XTTSClient] = {}  # keyed by persona name

    @property
    def fast(self) -> bool:
        """True when synthesis is local and fast enough to pipeline with LLM streaming."""
        return self.engine == "piper" and _PIPER_OK

    # ── public ─────────────────────────────────────────────────────────────────
    def synthesize(self, text: str, persona: "PersonaManager") -> Optional[str]:
        """
        Synthesize text for the given persona.
        Returns path to a WAV/MP3 file, or None on failure.
        """
        text = _clean_for_speech(text)
        if not text:
            return None

        # Voice cloning path — real voice samples via XTTS
        clone_dir = getattr(persona, "voice_clone_dir", None)
        if clone_dir:
            wav = self._xtts_synth(text, persona, clone_dir)
            if wav:
                return wav
            log.warning("XTTS voice cloning failed — falling back to edge-tts.")

        # Route by engine preference (with automatic fallback)
        if self.engine == "piper" and _PIPER_OK:
            wav = self._piper_synth(text, persona)
            if wav:
                return self._maybe_trim(wav)
            log.warning("Piper failed — trying edge-tts.")

        wav = self._edge_tts(text, persona)
        if wav:
            return wav

        return self._say_fallback(text, persona)

    # ── XTTS voice cloning ─────────────────────────────────────────────────────
    def _xtts_synth(self, text: str, persona: "PersonaManager", clone_dir: str) -> Optional[str]:
        name = persona.display
        if name not in self._xtts_clients:
            ref_wavs = sorted(Path(clone_dir).glob("*.wav"))
            if not ref_wavs:
                log.warning("No wav files found in voice_clone_dir: %s", clone_dir)
                return None
            self._xtts_clients[name] = _XTTSClient([str(p) for p in ref_wavs])
        return self._xtts_clients[name].synthesize(text)

    # ── Piper TTS ──────────────────────────────────────────────────────────────
    def _load_piper(self):
        if self._piper_voice is not None:
            return
        model_path = Path(self.piper_model)
        if not model_path.exists():
            raise FileNotFoundError(f"Piper model not found: {model_path}")
        config_path = self.piper_config or str(model_path) + ".json"
        log.info("Loading Piper model: %s", model_path)
        self._piper_voice = PiperVoice.load(str(model_path), config_path=config_path, use_cuda=False)

    def _piper_synth(self, text: str, persona: "PersonaManager") -> Optional[str]:
        try:
            self._load_piper()
        except Exception as exc:
            log.warning("Piper load failed: %s", exc)
            return None

        try:
            import wave
            out = tempfile.mktemp(suffix=".wav")
            embedding: Optional[np.ndarray] = persona.speaker_embedding

            with wave.open(out, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self._piper_voice.config.sample_rate)

                synth_kwargs = {}
                if embedding is not None:
                    # Multi-speaker Piper: pass speaker embedding
                    speaker_id = getattr(persona, "speaker_id", None)
                    if speaker_id is not None:
                        synth_kwargs["speaker_id"] = speaker_id
                    # Some Piper builds accept speaker_embedding directly
                    try:
                        synth_kwargs["speaker_embedding"] = embedding
                    except Exception:
                        pass

                for audio_bytes in self._piper_voice.synthesize_stream_raw(text[:_PIPER_MAX], **synth_kwargs):
                    wf.writeframes(audio_bytes)

            return out
        except Exception as exc:
            log.error("Piper synthesis error: %s", exc)
            return None

    # ── edge-tts ───────────────────────────────────────────────────────────────
    def _edge_tts(self, text: str, persona: "PersonaManager") -> Optional[str]:
        voice = persona.edge_voice or self.edge_voice
        rate  = getattr(persona, "edge_rate",  None) or self.edge_rate
        pitch = getattr(persona, "edge_pitch", None)
        try:
            out = tempfile.mktemp(suffix=".mp3")
            args = ["edge-tts", "--voice", voice, "--text", text[:1200], "--write-media", out]
            if rate:
                args += [f"--rate={rate}"]
            if pitch:
                args += [f"--pitch={pitch}"]
            r = subprocess.run(args, capture_output=True, timeout=15)
            if r.returncode == 0 and Path(out).exists():
                return out
            log.warning("edge-tts failed (rc=%d): %s", r.returncode, r.stderr.decode()[:200])
        except FileNotFoundError:
            log.warning("edge-tts not found — install: pip install edge-tts")
        except Exception as exc:
            log.error("edge-tts error: %s", exc)
        return None

    # ── macOS say ──────────────────────────────────────────────────────────────
    def _say_fallback(self, text: str, persona: "PersonaManager") -> Optional[str]:
        try:
            out = tempfile.mktemp(suffix=".aiff")
            voice = getattr(persona, "say_voice", "Daniel")
            subprocess.run(
                ["say", "-v", voice, "-o", out, text[:1000]],
                capture_output=True, timeout=20, check=True,
            )
            return out
        except Exception as exc:
            log.error("say fallback failed: %s", exc)
        return None

    # ── silence trim ───────────────────────────────────────────────────────────
    def _maybe_trim(self, wav_path: str) -> str:
        if not self.trim_silence:
            return wav_path
        try:
            out = tempfile.mktemp(suffix=".wav")
            r = subprocess.run(
                [self.ffmpeg, "-y", "-i", wav_path, "-af", _FFMPEG_TRIM, out],
                capture_output=True, timeout=10,
            )
            if r.returncode == 0 and Path(out).exists():
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass
                return out
        except Exception:
            pass
        return wav_path


# ── text cleaning ──────────────────────────────────────────────────────────────
_MD_LINKS   = re.compile(r"\[([^\]]*)\]\([^)]*\)")          # [label](url) → label
_MD_FENCES  = re.compile(r"```[\s\S]*?```|`[^`\n]+`")       # code fences / inline code
_MD_HEADING = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_MD_BOLD    = re.compile(r"[\*_]{1,3}")
_MD_QUOTE   = re.compile(r"^>\s*", re.MULTILINE)
_URLS       = re.compile(r"https?://\S+")
_BRACKETS   = re.compile(r"\[[A-Z][A-Z\s'&]+\]")            # [JARVIS] [SIR'S PROFILE]
_DASHES     = re.compile(r"-{2,}")
_PARENS     = re.compile(r"\([^)]*\)")
_MULTI_DOT  = re.compile(r"\.(\s*\.)+")
_WHITESPACE = re.compile(r"\s{2,}")
_JARVIS_DOT = re.compile(r"J\.A\.R\.V\.I\.S\.?")


def _clean_for_speech(text: str) -> str:
    text = _JARVIS_DOT.sub("Jarvis", text)  # do first so dots don't confuse later patterns
    text = _MD_LINKS.sub(r"\1", text)        # [label](url) → label
    text = _MD_FENCES.sub("", text)
    text = _MD_HEADING.sub("", text)
    text = _MD_BOLD.sub("", text)
    text = _MD_QUOTE.sub("", text)
    text = _URLS.sub("", text)
    text = _BRACKETS.sub("", text)
    text = _DASHES.sub(" ", text)
    text = _PARENS.sub("", text)
    text = re.sub(r"\n{2,}", ". ", text)
    text = text.replace("\n", " ")
    text = _MULTI_DOT.sub(".", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()
