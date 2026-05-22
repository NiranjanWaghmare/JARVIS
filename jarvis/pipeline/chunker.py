"""
Sentence Chunker — splits an incoming token stream at sentence boundaries.

Splits ONLY at . ? ! boundaries (not word count), so TTS gets natural phrases.
"""

import re
from typing import Iterator

_BOUNDARY = re.compile(r'(?<=[.?!])\s+')
_ABBREVS  = re.compile(r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e|Inc|Ltd|Corp)\.$', re.I)


class SentenceChunker:
    """Stateful chunker: call push() with each new token, collect emitted sentences."""

    def __init__(self, min_chars: int = 12):
        self._buf = ""
        self._min_chars = min_chars  # don't emit tiny fragments

    def push(self, token: str) -> list[str]:
        """Add one token. Returns a list of complete sentences ready for TTS (may be empty)."""
        self._buf += token
        return self._drain()

    def flush(self) -> list[str]:
        """Call after the token stream ends. Returns any remaining buffer content."""
        out = []
        if self._buf.strip():
            out.append(self._buf.strip())
        self._buf = ""
        return out

    # ── private ────────────────────────────────────────────────────────────────
    def _drain(self) -> list[str]:
        results = []
        while True:
            m = _BOUNDARY.search(self._buf)
            if not m:
                break
            candidate = self._buf[:m.start() + 1].strip()
            rest       = self._buf[m.end():]

            # Don't break after abbreviations (Dr. Smith → keep together)
            if _ABBREVS.search(candidate):
                self._buf = rest
                continue

            if len(candidate) >= self._min_chars:
                results.append(candidate)
                self._buf = rest
            else:
                # Sentence too short — absorb into next
                self._buf = candidate + " " + rest

        return results


def split_into_sentences(text: str) -> list[str]:
    """Utility: split a complete text string into sentence chunks."""
    chunker = SentenceChunker()
    results = []
    # Feed word by word to simulate streaming
    words = text.split(" ")
    for i, word in enumerate(words):
        token = word + (" " if i < len(words) - 1 else "")
        results.extend(chunker.push(token))
    results.extend(chunker.flush())
    return results
