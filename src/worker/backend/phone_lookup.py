"""Resolve appId / partnerId / personaId from a called phone number (inbound SIP)."""

from __future__ import annotations

from pydantic import BaseModel

from ..logging import ConvLog
from .client import get_backend_client


class PhoneLookupResult(BaseModel):
    success: bool
    appId: str | None = None
    partnerId: str | None = None
    personaId: str | None = None


async def fetch_app_info_by_number(phone_number: str, conv: ConvLog) -> PhoneLookupResult | None:
    client = get_backend_client()
    try:
        resp = await client.get(
            "/webhooks/livekit/phone-lookup",
            params={"number": phone_number},
        )
        if not resp.is_success:
            conv.line("WARN", "CONFIG", f"phone-lookup → {resp.status_code}")
            return None
        return PhoneLookupResult.model_validate(resp.json())
    except Exception as e:
        conv.line("
        return None
