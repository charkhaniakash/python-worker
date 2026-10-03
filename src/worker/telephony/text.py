"""TTS-safe text sanitizer.

Emoji / pictograph characters cause choppy or paused synthesis on many TTS
providers — strip them and collapse whitespace before handing text to TTS.
"""

from __future__ import annotations

import re

_EMOJI_PATTERN = re.compile(
    "["
    "\U0001f000-\U0001faff"
    "☀-➿"
    "⬀-⯿"
    "︀-️"
    "\U0001f1e6-\U0001f1ff"
    "‍"
    "]",
    flags=re.UNICODE,
)

_WHITESPACE = re.compile(r"\s+")


def sanitize_for_tts(text: str | None) -> str:
    if not text:
        return ""
    cleaned = _EMOJI_PATTERN.sub("", text)
    return _WHITESPACE.sub(" ", cleaned).strip()
