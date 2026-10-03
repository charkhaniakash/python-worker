"""
Call-hangup helpers for the telephony pipeline.

Two things live here:

  1. `is_goodbye_intent(transcript)` — detect whether a final user transcript
     is an end-of-call request ("bye", "hang up", "end the call", …). Used by
     the entrypoint's transcript handler to trigger `end_call` when the caller
     asks to disconnect.

  2. `end_call(session, room, identity, farewell, conv, reason)` — play a
     short, non-interruptible farewell then hang up the SIP leg by removing
     the SIP participant via LiveKit RoomService. RoomService.RemoveParticipant
     is what causes LiveKit to send SIP BYE to the trunk; without it, calling
     `Room.disconnect()` alone only removes the agent from the room and the
     caller's phone stays connected.

`hangup_sip_participant` is exposed separately for callers that want the raw
kick without the farewell playback (e.g. teardown paths).
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any

from livekit import api

from ..logging import ConvLog


# ---------------------------------------------------------------------------
# Goodbye-intent detection
# ---------------------------------------------------------------------------

# Whole short utterances that end the call outright. Matched against the
# normalized transcript (lowercased, punctuation stripped). Keep entries
# short and unambiguous — anything that also appears mid-sentence naturally
# ("say bye to the noise") belongs in `_GOODBYE_PHRASES` instead so the
# 5-word cap protects it.
_GOODBYE_WORDS: frozenset[str] = frozenset({
    # English
    "bye",
    "goodbye",
    "bye bye",
    "byebye",
    "ok bye",
    "okay bye",
    "thanks bye",
    "thank you bye",
    "that's all",
    "thats all",
    "that is all",
    "i am done",
    "im done",
    "we are done",
    # Hindi (romanized)
    "alvida",
    "khatam",
    "bas",
    "bas bas",
    "theek hai bye",
    "ok bas",
})

# Single tokens that, if the whole utterance consists ONLY of these words
# (in any repetition/order, ≤ _MAX_WHOLE_UTTERANCE_WORDS), end the call.
# Handles "bye bye bye", "goodbye bye", "ok ok bye bye" without needing to
# enumerate every combination. Keep this to tokens that are unambiguously
# goodbye-only — never add filler words like "the" or "please" here.
_GOODBYE_TOKENS: frozenset[str] = frozenset({
    "bye",
    "byebye",
    "goodbye",
    "alvida",
    "khatam",
    "ok",
    "okay",
    "thanks",
    "thank",
    "you",   # only combines with "thank" — the token-only path still needs
             # at least one strong goodbye token to trip (see is_goodbye_intent)
})
# Tokens above that are "strong" — at least one must appear for the
# token-only rule to fire. This prevents "ok ok ok" or "thank you" alone
# from ending the call.
_STRONG_GOODBYE_TOKENS: frozenset[str] = frozenset({
    "bye", "byebye", "goodbye", "alvida", "khatam",
})

# Multi-word phrases that end the call even inside a longer utterance.
# Reserved for phrases whose only reasonable meaning is "please disconnect".
_GOODBYE_PHRASES: tuple[str, ...] = (
    "end the call",
    "end call",
    "hang up",
    "hangup",
    "cut the call",
    "cut call",
    "disconnect the call",
    "close the call",
    "call cut karo",
    "call band karo",
    "phone cut karo",
)

_MAX_WHOLE_UTTERANCE_WORDS = 5
_STRIP_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)

# Trailing-goodbye rule: a longer sentence that ENDS with one of these
# tokens is treated as an end-of-call intent ("Alright thanks a lot, bye.",
# "OK great, goodbye!"). Real callers rarely say a bare "bye" — they
# usually wrap it in pleasantries. Kept to strong tokens only so a random
# mid-conversation mention doesn't fire; still an accepted small false-
# positive risk on sentences like "he said bye" — rare on real calls.
_TRAILING_GOODBYE_RE = re.compile(
    r"\b(bye|byebye|goodbye|alvida|khatam)\s*$",
    re.IGNORECASE,
)


def _normalize(transcript: str) -> str:
    return _STRIP_PUNCT_RE.sub(" ", transcript.lower()).strip()


def is_goodbye_intent(transcript: str) -> bool:
    """
    Return True if `transcript` looks like an end-of-call request.

    Matching rules (in order):
      1. Exact match: normalized utterance is in `_GOODBYE_WORDS` AND has
         ≤ _MAX_WHOLE_UTTERANCE_WORDS words. Prevents "say bye to the noise"
         (7 words) from ending the call.
      2. Token-only: utterance has ≤ _MAX_WHOLE_UTTERANCE_WORDS words AND
         every word is a `_GOODBYE_TOKENS` entry AND at least one is a
         `_STRONG_GOODBYE_TOKENS` entry. Handles STT repetitions like
         "bye bye bye" and "ok ok bye" without enumerating combinations.
      3. Phrase substring: any `_GOODBYE_PHRASES` entry appears as a
         whole-word substring. These phrases only mean "disconnect", so
         they fire at any length.
      4. Trailing goodbye: the utterance ENDS with a strong goodbye token
         ("…thanks a lot, bye.", "…great, goodbye!"). Handles the common
         case of a caller closing a longer sentence with "bye".
    """
    if not transcript:
        return False
    norm = _normalize(transcript)
    if not norm:
        return False
    tokens = norm.split()
    if 1 <= len(tokens) <= _MAX_WHOLE_UTTERANCE_WORDS:
        if norm in _GOODBYE_WORDS:
            return True
        if all(t in _GOODBYE_TOKENS for t in tokens) and any(
            t in _STRONG_GOODBYE_TOKENS for t in tokens
        ):
            return True
    if any(
        re.search(rf"\b{re.escape(p)}\b", norm) is not None
        for p in _GOODBYE_PHRASES
    ):
        return True
    return _TRAILING_GOODBYE_RE.search(norm) is not None


# ---------------------------------------------------------------------------
# SIP hangup
# ---------------------------------------------------------------------------


def _http_url_from_env() -> str:
    return (
        os.environ.get("LIVEKIT_URL", "")
        .replace("wss://", "https://")
        .replace("ws://", "http://")
    )


async def hangup_sip_participant(
    room_name: str, participant_identity: str, conv: ConvLog
) -> None:
    """
    Kick the SIP participant so LiveKit sends BYE to the trunk and the
    caller's phone actually disconnects. Fail-open: errors are logged, never
    raised — a failed hangup should not prevent the rest of teardown.
    """
    api_key = os.environ.get("LIVEKIT_API_KEY", "")
    api_secret = os.environ.get("LIVEKIT_API_SECRET", "")
    http_url = _http_url_from_env()

    if not (http_url and api_key and api_secret):
        conv.line("WARN", "SESSION", "Cannot hang up SIP: LiveKit credentials missing")
        return

    try:
        lk = api.LiveKitAPI(http_url, api_key, api_secret)
        try:
            await lk.room.remove_participant(
                api.RoomParticipantIdentity(
                    room=room_name,
                    identity=participant_identity,
                )
            )
            conv.line(
                "INFO",
                "SESSION",
                "SIP participant removed — phone disconnecting",
                {"identity": participant_identity},
            )
        finally:
            await lk.aclose()
    except Exception as e:  # noqa: BLE001
        conv.line("WARN", "SESSION", f"SIP hangup via RoomService failed: {e}")


async def _play_and_wait(session: Any, text: str, conv: ConvLog) -> None:
    """Play `text` non-interruptibly and await playback completion."""
    if not text:
        return
    try:
        handle = session.say(text, allow_interruptions=False)
        wait_fn = getattr(handle, "wait_for_playout", None)
        if callable(wait_fn):
            result = wait_fn()
            if asyncio.iscoroutine(result):
                await result
    except Exception as e:  # noqa: BLE001
        conv.line("WARN", "SESSION", f"Farewell TTS failed: {e}")


async def end_call(
    session: Any,
    room_name: str,
    participant_identity: str,
    farewell: str,
    conv: ConvLog,
    *,
    reason: str,
) -> None:
    """
    Play a short farewell, then hang up the SIP leg. Used by both the
    idle-timeout watchdog and the user-goodbye handler so the two paths
    stay consistent.
    """
    conv.line("INFO", "SESSION", f"Ending call ({reason}) — playing farewell")
    await _play_and_wait(session, farewell, conv)
    await hangup_sip_participant(room_name, participant_identity, conv)
