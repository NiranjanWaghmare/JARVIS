#!/usr/bin/env python3
"""
Persistent XTTS2 voice cloning server.
Pre-computes speaker latents from all reference clips, then synthesises via
direct model.inference() — skips per-call speaker encoding for 10-20x speedup.
MPS (Apple Silicon) is used when available; tensors and model stay on same device.
"""
import sys
import os
import io
import json
import contextlib
import tempfile

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["COQUI_TOS_AGREED"] = "1"

import warnings
warnings.filterwarnings("ignore")

import torch

def _best_device() -> str:
    try:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def _move(obj, device):
    if hasattr(obj, "to"):
        try:
            return obj.to(device)
        except Exception:
            pass
    return obj

def main():
    ref_files = sys.argv[1:] if len(sys.argv) > 1 else ["stewie_ref.wav"]
    ref_files = [f for f in ref_files if os.path.exists(f)]

    if not ref_files:
        print(json.dumps({"status": "error", "message": "No valid reference audio files found"}), flush=True)
        sys.exit(1)

    try:
        from TTS.api import TTS
        import soundfile as sf
    except ImportError as e:
        print(json.dumps({"status": "error", "message": f"TTS not installed: {e}"}), flush=True)
        sys.exit(1)

    device = _best_device()
    print(f"LOADING:{device}", flush=True)

    # Load on CPU first (TTS library always initialises to CPU)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2",
                      progress_bar=False,
                      gpu=(device == "cuda"))
        try:
            tts.synthesizer.tts_model.config.use_language_d_vector_file = False
        except (TypeError, AttributeError):
            pass
    except Exception as e:
        print(json.dumps({"status": "error", "message": f"Model load failed: {e}"}), flush=True)
        sys.exit(1)

    model = tts.synthesizer.tts_model

    # ── Pre-compute speaker latents on CPU first ──────────────────────────────
    # Do this BEFORE moving to MPS so we know it works; we'll move them together.
    try:
        cfg = model.config
        latent_kwargs = {"audio_path": ref_files}
        for k in ("gpt_cond_len", "gpt_cond_chunk_len", "max_ref_len", "sound_norm_refs"):
            if hasattr(cfg, k):
                latent_kwargs[k] = getattr(cfg, k)
        with contextlib.redirect_stdout(io.StringIO()):
            gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(**latent_kwargs)
        use_direct = True
        print("LATENTS:ok", flush=True)
    except Exception as e:
        sys.__stdout__.write(f"# Latent precompute failed ({e}), will use tts_to_file\n")
        sys.__stdout__.flush()
        gpt_cond_latent = speaker_embedding = None
        use_direct = False

    # ── Move model AND latents to target device together ─────────────────────
    if device == "mps":
        try:
            model = model.to("mps")
            tts.synthesizer.tts_model = model
            if gpt_cond_latent is not None:
                gpt_cond_latent  = _move(gpt_cond_latent,  "mps")
                speaker_embedding = _move(speaker_embedding, "mps")
            print("DEVICE:mps", flush=True)
        except Exception as e:
            sys.__stdout__.write(f"# MPS move failed ({e}), staying on CPU\n")
            sys.__stdout__.flush()
            device = "cpu"

    sample_rate = 24000
    try:
        sample_rate = model.config.audio.sample_rate
    except Exception:
        pass

    ref_audio = ref_files if len(ref_files) > 1 else ref_files[0]
    print("READY", flush=True)

    def _synthesize(text: str, output: str) -> bool:
        nonlocal device, model, gpt_cond_latent, speaker_embedding, use_direct
        try:
            if use_direct:
                with contextlib.redirect_stdout(io.StringIO()):
                    out = model.inference(
                        text=text,
                        language="en",
                        gpt_cond_latent=gpt_cond_latent,
                        speaker_embedding=speaker_embedding,
                        temperature=0.7,
                        enable_text_splitting=False,
                    )
                sf.write(output, out["wav"], sample_rate)
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    tts.tts_to_file(text=text, speaker_wav=ref_audio,
                                    language="en", file_path=output)
            return True
        except (RuntimeError, Exception) as e:
            err = str(e)
            if device == "mps" and ("mps" in err.lower() or "metal" in err.lower()):
                # MPS op unsupported — permanently fall back to CPU
                sys.__stdout__.write("# MPS synthesis error, falling back to CPU\n")
                sys.__stdout__.flush()
                device = "cpu"
                model = model.to("cpu")
                tts.synthesizer.tts_model = model
                gpt_cond_latent  = _move(gpt_cond_latent,  "cpu")
                speaker_embedding = _move(speaker_embedding, "cpu")
                return _synthesize(text, output)  # retry on CPU
            elif use_direct:
                # model.inference() API mismatch — fall back to tts_to_file permanently
                sys.__stdout__.write(f"# model.inference() failed ({err[:80]}), switching to tts_to_file\n")
                sys.__stdout__.flush()
                use_direct = False
                return _synthesize(text, output)
            raise

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req  = json.loads(line)
            text = req.get("text", "").strip()
            out  = req.get("output", tempfile.mktemp(suffix=".wav"))

            if not text:
                sys.__stdout__.write(json.dumps({"status": "error", "message": "empty text"}) + "\n")
                sys.__stdout__.flush()
                continue

            try:
                ok = _synthesize(text, out)
                if ok:
                    sys.__stdout__.write(json.dumps({"status": "ok", "file": out}) + "\n")
                else:
                    sys.__stdout__.write(json.dumps({"status": "error", "message": "synthesis returned False"}) + "\n")
            except Exception as e:
                sys.__stdout__.write(json.dumps({"status": "error", "message": str(e)}) + "\n")
            sys.__stdout__.flush()

        except Exception as e:
            sys.__stdout__.write(json.dumps({"status": "error", "message": str(e)}) + "\n")
            sys.__stdout__.flush()

if __name__ == "__main__":
    main()
