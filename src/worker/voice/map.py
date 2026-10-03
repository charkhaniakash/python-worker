"""
Google Cloud voice name resolution for Chirp3 HD streaming TTS.

Chirp3 HD voice names follow the format: <locale>-Chirp3-HD-<VoiceName>
e.g. "en-IN-Chirp3-HD-Kore", "hi-IN-Chirp3-HD-Orus"

The livekit google.TTS plugin auto-detects chirp_3 model when "chirp" appears
in the voice name, which enables gRPC streaming. This module translates:
  - Legacy Google Cloud TTS voice ids (e.g. "en-IN-Neural2-B") → Chirp3 HD
  - Bare Gemini voice names (e.g. "Kore") → Chirp3 HD with locale prefix
  - Already-valid Chirp3 HD names → pass through unchanged

All voice data (the 30 prebuilt voice names, masculine set, locale map)
lives in `config/defaults.yaml` under `gemini_voices` — this module is pure
logic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from ..logging import ConvLog
from ..settings import get_settings

VoiceGender = Literal["female", "male", "neutral"]

_LEGACY_GOOGLE_CLOUD_VOICE = re.compile(
    r"^[a-z]{2,3}-[A-Z]{2}-"
    r"(?:Neural2|Wavenet|Standard|Studio|News|Polyglot|Chirp3?|Journey|Casual)"
    r"-[A-Z0-9]+$",
    flags=re.IGNORECASE,
)

_LEGACY_SUFFIX = re.compile(
    r"-(?:Neural2|Wavenet|Standard|Studio|News|Polyglot|Chirp3?|Journey|Casual)-([A-Z0-9]+)$",
    flags=re.IGNORECASE,
)

# Chirp3 HD voice name format: <locale>-Chirp3-HD-<VoiceName>
_CHIRP3_HD_VOICE = re.compile(
    r"^([a-z]{2,3}-[A-Z]{2})-Chirp3-HD-(\w+)$",
    flags=re.IGNORECASE,
)


@dataclass
class ResolvedGeminiVoice:
    voice: str           # Full Chirp3 HD name e.g. "en-IN-Chirp3-HD-Kore"
    gender: VoiceGender = "female"
    mapped_from: str | None = None
    locale: str = "en-IN"  # BCP-47 locale for the language_code param


def is_gemini_tts_voice(voice: str) -> bool:
    voices = set(get_settings().runtime.gemini_voices.all)
    return voice in voices


def is_legacy_google_cloud_tts_voice(voice: str) -> bool:
    trimmed = voice.strip()
    if not trimmed or is_gemini_tts_voice(trimmed):
        return False
    return bool(_LEGACY_GOOGLE_CLOUD_VOICE.match(trimmed))


def _canonical_gemini_voice(voice: str) -> str | None:
    if is_gemini_tts_voice(voice):
        return voice
    target = voice.strip().lower()
    for v in get_settings().runtime.gemini_voices.all:
        if v.lower() == target:
            return v
    return None


def _infer_gender_from_legacy(voice: str) -> VoiceGender:
    match = _LEGACY_SUFFIX.search(voice)
    if not match:
        return "neutral"
    suffix = match.group(1).upper()
    if suffix in {"A", "C", "E", "G", "I", "FEMALE", "F"}:
        return "female"
    if suffix in {"B", "D", "H", "J", "MALE", "M"}:
        return "male"
    return "neutral"


def _locale_key_from_legacy_voice(voice: str) -> str:
    parts = voice.strip().split("-")
    if len(parts) >= 2:
        return f"{parts[0].lower()}-{parts[1].lower()}"
    return parts[0].lower() if parts else "en"


def _locale_to_bcp47(locale_key: str) -> str:
    """Convert our internal locale key (e.g. 'en-in') to proper BCP-47 (e.g. 'en-IN')."""
    parts = locale_key.split("-")
    if len(parts) == 2:
        return f"{parts[0].lower()}-{parts[1].upper()}"
    return locale_key


def _map_locale_gender_to_gemini(locale_key: str, gender: VoiceGender) -> str:
    """Return the bare Gemini voice name (e.g. 'Kore') for the given locale+gender."""
    voices_cfg = get_settings().runtime.gemini_voices
    default_voice = voices_cfg.locale_map.get("en")
    lang = locale_key.split("-")[0] if "-" in locale_key else locale_key
    mapping = (
        voices_cfg.locale_map.get(locale_key)
        or voices_cfg.locale_map.get(lang)
        or default_voice
    )
    fallback = get_settings().runtime.tts.google.default_voice
    if not mapping:
        return fallback
    if gender == "female":
        return mapping.female
    if gender == "male":
        return mapping.male
    return mapping.default


def _to_chirp3_hd_name(bare_voice: str, locale_bcp47: str) -> str:
    """Format a bare voice name into Chirp3 HD format: <locale>-Chirp3-HD-<Voice>."""
    # Normalise locale to Title-case region: en-IN, hi-IN etc.
    parts = locale_bcp47.split("-")
    if len(parts) == 2:
        locale_norm = f"{parts[0].lower()}-{parts[1].upper()}"
    else:
        locale_norm = locale_bcp47
    # Capitalise voice name for the HD format (Kore, Orus etc.)
    voice_title = bare_voice.strip().capitalize()
    return f"{locale_norm}-Chirp3-HD-{voice_title}"


def resolve_gemini_tts_voice(raw_voice: str | None, *, conv: ConvLog | None = None) -> ResolvedGeminiVoice:
    """
    Translate any Google voice id into a Chirp3 HD voice name for streaming.

    Output format: "<locale>-Chirp3-HD-<VoiceName>"  e.g. "en-IN-Chirp3-HD-Kore"

    The livekit google.TTS plugin sees "chirp" in the voice name, auto-selects
    chirp_3 model, and enables gRPC streaming — which is what we need.
    Never raises — unknown values fall back to the configured default.
    """
    default_bare = get_settings().runtime.tts.google.default_voice  # e.g. "Kore"
    masculine = set(get_settings().runtime.gemini_voices.masculine)

    trimmed = (raw_voice or "").strip()
    if not trimmed:
        voice = _to_chirp3_hd_name(default_bare, "en-IN")
        return ResolvedGeminiVoice(voice=voice, gender="female", locale="en-IN")

    # Already a valid Chirp3 HD name — pass through
    chirp3_match = _CHIRP3_HD_VOICE.match(trimmed)
    if chirp3_match:
        locale_bcp47 = _locale_to_bcp47(chirp3_match.group(1))
        bare = chirp3_match.group(2)
        gender: VoiceGender = "male" if bare.capitalize() in masculine else "female"
        return ResolvedGeminiVoice(voice=trimmed, gender=gender, locale=locale_bcp47)

    # Bare Gemini voice name e.g. "Kore", "Orus"
    canonical = _canonical_gemini_voice(trimmed)
    if canonical:
        gender = "male" if canonical in masculine else "female"
        # Default locale for bare names — en-IN for Indian context
        locale_bcp47 = "en-IN"
        voice = _to_chirp3_hd_name(canonical, locale_bcp47)
        return ResolvedGeminiVoice(voice=voice, gender=gender, locale=locale_bcp47)

    # Legacy Google Cloud TTS voice e.g. "en-IN-Neural2-B"
    if is_legacy_google_cloud_tts_voice(trimmed):
        locale_key = _locale_key_from_legacy_voice(trimmed)
        locale_bcp47 = _locale_to_bcp47(locale_key)
        gender = _infer_gender_from_legacy(trimmed)
        bare = _map_locale_gender_to_gemini(locale_key, gender)
        voice = _to_chirp3_hd_name(bare, locale_bcp47)
        msg = (
            f"Mapped legacy Google Cloud voice '{trimmed}' → Chirp3 HD "
            f"'{voice}' ({locale_bcp47}, {gender})"
        )
        if conv is not None:
            conv.line("WARN", "TTS", msg)
        else:
            print(f"[TTS] {msg}")
        return ResolvedGeminiVoice(voice=voice, gender=gender, mapped_from=trimmed, locale=locale_bcp47)

    # Unknown — fall back to default
    voice = _to_chirp3_hd_name(default_bare, "en-IN")
    msg = f"Google voice '{trimmed}' is unrecognised — using default '{voice}'"
    if conv is not None:
        conv.line("WARN", "TTS", msg)
    else:
        print(f"[TTS] {msg}")
    return ResolvedGeminiVoice(voice=voice, gender="female", mapped_from=trimmed, locale="en-IN")
