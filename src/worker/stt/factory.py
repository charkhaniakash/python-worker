"""STT factory. Deepgram is the only provider today; the interface stays open
so we can add more providers without touching entrypoint code."""

from __future__ import annotations

import inspect
from typing import Any

from livekit.plugins import deepgram

from ..settings import get_settings

# livekit-plugins-deepgram.STT does not accept utterance_end_ms (that's a
# Deepgram REST/utterance_end query that this plugin never wired). Passing it
# TypeErrors on some versions and is silently ignored on others.
_STT_SKIP_KEYS = frozenset({"utterance_end_ms"})


def build_stt(
    language: str,
    *,
    is_telephony: bool,
    model: str | None = None,
    keywords: list[tuple[str, float]] | None = None,
) -> deepgram.STT:
    s = get_settings().runtime.stt
    sample_rate = (
        s.sample_rate_hz["telephony"] if is_telephony else s.sample_rate_hz["non_telephony"]
    )
    kwargs: dict[str, Any] = {
        "model": model or s.model,
        "language": language,
        "sample_rate": sample_rate,
        "detect_language": s.detect_language,
    }
    for key in (
        "endpointing_ms",
        "interim_results",
        "smart_format",
        "punctuate",
        "filler_words",
        "no_delay",
        "numerals",
    ):
        val = getattr(s, key, None)
        if val is not None:
            kwargs[key] = val

    keyterms = [t.strip() for t in (s.keyterms or []) if t and t.strip()]
    if keyterms:
        kwargs["keyterms"] = keyterms
    if keywords:
        kwargs["keywords"] = keywords

    accepted = set(inspect.signature(deepgram.STT.__init__).parameters)
    accepted.discard("self")
    skip = set(_STT_SKIP_KEYS)
    # nova-2 accepts utterance_end_ms; older plugins TypeError — keep skip
    # unless this install actually has the kwarg.
    if "utterance_end_ms" in accepted and (model or s.model) == "nova-2":
        skip.discard("utterance_end_ms")
        val = getattr(s, "utterance_end_ms", None)
        if val is not None:
            kwargs["utterance_end_ms"] = val
    kwargs = {k: v for k, v in kwargs.items() if k in accepted and k not in skip}
    return deepgram.STT(**kwargs)
