"""
Translate raw LiveKit metric events into cost-tracking provider keys + log lines.

Field-name reference (livekit-agents 1.6.x, verified against the installed
plugin — do not "correct" these to camelCase, they are snake_case in the
Python SDK and durations come in **seconds** on Python, not milliseconds):
  STTMetrics : type, label, audio_duration, duration, streamed, ...
  LLMMetrics : type, label, ttft, duration, prompt_tokens, completion_tokens, ...
  TTSMetrics : type, label, ttfb, duration, audio_duration, characters_count, ...
  EOUMetrics : type, end_of_utterance_delay, transcription_delay, ...
  VADMetrics : type, idle_time, inference_count, inference_duration_total, ...
"""

from __future__ import annotations

from typing import Any

from ..settings import get_settings


def stt_provider_key() -> str:
    return get_settings().runtime.cost_tracking.stt_provider_key


def tts_label_to_provider_key(label: str | None, fallback: str) -> str:
    prefixes = get_settings().runtime.cost_tracking.tts_provider_key_by_label_prefix
    if label:
        for prefix, key in prefixes.items():
            if label.startswith(prefix):
                return key
    return fallback


def fallback_tts_provider_key(provider: str) -> str:
    if provider == "google":
        return "google_tts"
    if provider == "open_ai":
        return "openai_tts"
    return "elevenlabs"


def _sec_to_ms(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(float(value) * 1000)
    except (TypeError, ValueError):
        return 0


def _int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def stt_audio_ms(m: Any) -> int:
    return _sec_to_ms(getattr(m, "audio_duration", None))


def llm_tokens(m: Any) -> tuple[int, int]:
    return _int(getattr(m, "prompt_tokens", None)), _int(getattr(m, "completion_tokens", None))


def tts_characters(m: Any) -> int:
    return _int(getattr(m, "characters_count", None))


def format_pipeline_metrics(m: Any) -> str | None:
    """One-line summary of a metrics event, or None to skip."""
    m_type = getattr(m, "type", None)
    if m_type == "stt_metrics":
        return f"STT audio={_sec_to_ms(getattr(m, 'audio_duration', None))}ms streamed={getattr(m, 'streamed', None)}"
    if m_type == "llm_metrics":
        return (
            f"LLM ttft={_sec_to_ms(getattr(m, 'ttft', None))}ms "
            f"out={_int(getattr(m, 'completion_tokens', None))}tok "
            f"in={_int(getattr(m, 'prompt_tokens', None))}tok "
            f"{_sec_to_ms(getattr(m, 'duration', None))}ms"
        )
    if m_type == "tts_metrics":
        return (
            f"TTS ttfb={_sec_to_ms(getattr(m, 'ttfb', None))}ms "
            f"speech={_sec_to_ms(getattr(m, 'audio_duration', None))}ms "
            f"chars={_int(getattr(m, 'characters_count', None))}"
        )
    if m_type == "eou_metrics":
        return (
            f"EOU transcript_delay={_sec_to_ms(getattr(m, 'transcription_delay', None))}ms "
            f"eou_delay={_sec_to_ms(getattr(m, 'end_of_utterance_delay', None))}ms"
        )
    if m_type == "vad_metrics":
        return (
            f"VAD inferences={_int(getattr(m, 'inference_count', None))} "
            f"idle={_sec_to_ms(getattr(m, 'idle_time', None))}ms"
        )
    return None
