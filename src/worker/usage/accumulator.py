"""
Per-call raw usage accumulator.

STT is bucketed by a single provider key (never changes mid-call).
LLM is bucketed by a single provider key (never changes mid-call).
TTS uses a single provider for the whole call too (no mid-call swapping), but
we still bucket by the label reported on each metric event for robustness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class _LlmTokens:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class CallUsageAccumulator:
    _stt: dict[str, int] = field(default_factory=dict)              # provider → audio_duration_ms
    _llm: dict[str, _LlmTokens] = field(default_factory=dict)       # provider → tokens
    _tts: dict[str, int] = field(default_factory=dict)              # provider → characters_count

    def add_stt(self, provider: str, audio_duration_ms: int) -> None:
        self._stt[provider] = self._stt.get(provider, 0) + audio_duration_ms

    def add_llm(self, provider: str, prompt_tokens: int, completion_tokens: int) -> None:
        cur = self._llm.get(provider) or _LlmTokens()
        cur.prompt_tokens += prompt_tokens
        cur.completion_tokens += completion_tokens
        self._llm[provider] = cur

    def add_tts(self, provider: str, characters_count: int) -> None:
        self._tts[provider] = self._tts.get(provider, 0) + characters_count

    def to_payload(self, call_log_id: str, duration_seconds: int) -> dict[str, Any]:
        return {
            "callLogId": call_log_id,
            "durationSeconds": duration_seconds,
            "stt": [
                {"provider": p, "audioDurationMs": ms} for p, ms in self._stt.items()
            ],
            "llm": [
                {
                    "provider": p,
                    "promptTokens": tok.prompt_tokens,
                    "completionTokens": tok.completion_tokens,
                }
                for p, tok in self._llm.items()
            ],
            "tts": [
                {"provider": p, "charactersCount": chars} for p, chars in self._tts.items()
            ],
        }
