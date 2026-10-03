"""Inbound call-log creation and per-call usage reporting."""

from __future__ import annotations

from typing import Any

from ..logging import ConvLog
from ..settings import get_settings
from ..telephony.signaling import with_timeout
from .client import get_backend_client


async def create_inbound_call_log(
    *,
    app_id: str,
    partner_id: str,
    from_number: str | None,
    to_number: str | None,
    call_sid: str | None,
    conv: ConvLog,
) -> str | None:
    """Ask the backend to create a CallLog row for an inbound call. Returns callLogId or None."""
    s = get_settings()
    client = get_backend_client()
    try:
        resp = await with_timeout(
            client.post_json(
                "/webhooks/livekit/call-started",
                json_body={
                    "appId": app_id,
                    "partnerId": partner_id,
                    "fromNumber": from_number,
                    "toNumber": to_number,
                    "callSid": call_sid,
                },
                timeout_ms=s.runtime.backend.call_started.timeout_ms,
            ),
            timeout_ms=s.runtime.backend.call_started.timeout_ms,
            label="call-started",
        )
        if not resp.is_success:
            conv.line("WARN", "COST", f"call-started → {resp.status_code}")
            return None
        data: dict[str, Any] = resp.json()
        if data.get("success") and data.get("callLogId"):
            call_log_id = str(data["callLogId"])
            conv.line("INFO", "COST", "Inbound CallLog created", {"callLogId": call_log_id})
            return call_log_id
        return None
    except Exception as e:
        conv.line(
            "WARN",
            "COST",
            f"call-started failed (cost won't be tracked for this call): {e}",
        )
        return None


async def report_call_usage(payload: dict[str, Any], conv: ConvLog) -> None:
    """Fire-and-forget usage report — must never block or break call teardown."""
    s = get_settings()
    client = get_backend_client()
    try:
        resp = await with_timeout(
            client.post_json(
                "/webhooks/livekit/call-usage",
                json_body=payload,
                timeout_ms=s.runtime.backend.usage_report.timeout_ms,
            ),
            timeout_ms=s.runtime.backend.usage_report.timeout_ms,
            label="call-usage report",
        )
        if resp.is_success:
            conv.line("INFO", "COST", "Call usage reported for cost calculation")
        else:
            conv.line("WARN", "COST", f"call-usage report → {resp.status_code}")
    except Exception as e:
        conv.line("WARN", "COST", f"call-usage report failed: {e}")
