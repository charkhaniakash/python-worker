"""
SIP call lifecycle helpers — waiting for the callee to answer, waiting for the
room to fully close, and waiting for `RoomIO` to be ready before the greeting
is spoken.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from livekit import rtc

from ..logging import ConvLog


async def wait_for_sip_answer(
    room: rtc.Room,
    participant: Any,
    conv: ConvLog,
    *,
    timeout_ms: int,
    poll_interval_ms: int,
) -> bool:
    """
    Outbound SIP legs join in `dialing` state while the phone rings. Block until
    `sip.callStatus == "active"` (answered), or short-circuit false if the
    callee rejects, hangs up, or the timer expires.

    Returns True on answer, False on any negative outcome.
    """
    tracked_attrs: dict[str, str] = dict(getattr(participant, "attributes", None) or {})

    def status() -> str | None:
        return tracked_attrs.get("sip.callStatus")

    initial = status()
    if initial is None:  # not a SIP leg
        return True
    if initial == "active":
        return True

    conv.line("INFO", "SESSION", f"Ringing — waiting for answer (status={initial})…")

    future: asyncio.Future[bool] = asyncio.get_event_loop().create_future()

    def resolve(answered: bool, reason: str) -> None:
        if future.done():
            return
        conv.line("INFO", "SESSION", f"Answer wait resolved: {reason}")
        future.set_result(answered)

    identity = getattr(participant, "identity", None)

    def on_attributes_changed(changed_attrs: dict[str, str], p: Any) -> None:
        if getattr(p, "identity", None) != identity:
            return
        # Merge because the SDK gives us only the delta.
        current = getattr(p, "attributes", None) or {}
        tracked_attrs.update(current)
        tracked_attrs.update(changed_attrs)
        new_status = changed_attrs.get("sip.callStatus")
        if new_status:
            conv.line("INFO", "SESSION", f"SIP status changed: {new_status}")
        _check()

    def on_track_published(_publication: Any, p: Any) -> None:
        if getattr(p, "identity", None) == identity:
            resolve(True, "remote media track published")

    def _check() -> None:
        s = status()
        if s == "active":
            resolve(True, "answered")
        elif s in {"hangup", "disconnected"}:
            resolve(False, f"callee {s}")

    room.on("participant_attributes_changed", on_attributes_changed)
    room.on("track_published", on_track_published)

    start = time.monotonic()
    poll_seconds = poll_interval_ms / 1000.0
    try:
        while not future.done():
            _check()
            if future.done():
                break
            if (time.monotonic() - start) * 1000.0 > timeout_ms:
                resolve(False, "ring timeout")
                break
            try:
                return await asyncio.wait_for(
                    asyncio.shield(future),
                    timeout=poll_seconds,
                )
            except asyncio.TimeoutError:
                continue
        return future.result()
    finally:
        room.off("participant_attributes_changed", on_attributes_changed)
        room.off("track_published", on_track_published)


async def wait_for_call_end(room: rtc.Room, conv: ConvLog) -> None:
    """Resolve when the remote side leaves or the room disconnects."""

    future: asyncio.Future[None] = asyncio.get_event_loop().create_future()

    def done(reason: str) -> None:
        if future.done():
            return
        conv.line("INFO", "SESSION", f"Call ended ({reason})")
        future.set_result(None)

    def on_participant_disconnected(_p: Any) -> None:
        if len(room.remote_participants) == 0:
            done("participant-left")

    def on_room_disconnected(*_args: Any, **_kwargs: Any) -> None:
        done("room-disconnected")

    if len(room.remote_participants) == 0:
        conv.line("INFO", "SESSION", "Call ended (no-participants)")
        return

    room.on("participant_disconnected", on_participant_disconnected)
    room.on("disconnected", on_room_disconnected)
    try:
        await future
    finally:
        room.off("participant_disconnected", on_participant_disconnected)
        room.off("disconnected", on_room_disconnected)


async def with_timeout(coro: "asyncio.Future[Any]", timeout_ms: int, label: str) -> Any:
    """Fail fast — raise TimeoutError with a descriptive label instead of hanging."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_ms / 1000.0)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"{label} timed out after {timeout_ms}ms") from exc
