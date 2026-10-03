"""Compact, turn-aware conversation logger."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from .settings import get_settings

ConvLevel = Literal["INFO", "USER", "AGENT", "WARN", "ERROR"]

_ICONS: dict[ConvLevel, str] = {
    "INFO": "·",
    "USER": "👤",
    "AGENT": "🤖",
    "WARN": "⚠",
    "ERROR": "✗",
}


def _clock() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]


def truncate_for_log(text: str | None, max_chars: int | None = None) -> str:
    if not text:
        return "(empty)"
    if max_chars is None:
        max_chars = get_settings().runtime.logging.truncate_chars
    one_line = " ".join(text.split()).strip()
    if len(one_line) <= max_chars:
        return one_line
    return f"{one_line[:max_chars]}… [{len(one_line)} chars]"


@dataclass
class ConvLog:
    """
    Turn-aware structured logger. Use `bump_turn()` when a final user
    utterance arrives; every subsequent log line is stamped with that turn.
    """

    session_id: str
    _turn: int = field(default=0, init=False, repr=False)

    @classmethod
    def create(cls, room_name: str | None, job_id: str | None) -> "ConvLog":
        sid = (room_name or job_id or "session")[:40]
        return cls(session_id=sid)

    @property
    def turn(self) -> int:
        return self._turn

    def bump_turn(self) -> int:
        self._turn += 1
        return self._turn

    def line(
        self,
        level: ConvLevel,
        category: str,
        message: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        turn_tag = f" T{self._turn}" if self._turn > 0 else ""
        extra_str = ""
        if extra:
            # Filter None values so log lines don't get cluttered with noise
            filtered = {k: v for k, v in extra.items() if v is not None}
            if filtered:
                extra_str = f" {json.dumps(filtered, default=str, ensure_ascii=False)}"
        print(
            f"[{_clock()}] [{self.session_id}{turn_tag}] "
            f"[{level}/{category}] {_ICONS[level]} {message}{extra_str}",
            flush=True,
        )
