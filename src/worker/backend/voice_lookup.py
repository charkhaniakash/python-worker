"""Resolve a voice name into its provider + provider-specific voice id."""

from __future__ import annotations

from pydantic import BaseModel

from ..logging import ConvLog
from .client import get_backend_client


class VoiceLookupResult(BaseModel):
    provider: str
    voice_id: str
    gender: str | None = None


class _VoiceApiMetadata(BaseModel):
    voice_id: str | None = None
    region: str | None = None


class _VoiceApiResponse(BaseModel):
    id: str
    provider: str
    name: str
    display_name: str | None = None
    gender: str | None = None
    metadata: _VoiceApiMetadata | None = None


async def fetch_voice_details(voice_name: str, conv: ConvLog) -> VoiceLookupResult | None:
    client = get_backend_client()
    try:
        resp = await client.get("/voice/by-name", params={"name": voice_name})
        if not resp.is_success:
            conv.line("WARN", "TTS", f"Voice API returned {resp.status_code} for '{voice_name}'")
            return None
        data = _VoiceApiResponse.model_validate(resp.json())
        if data.provider == "open_ai":
            return VoiceLookupResult(provider="open_ai", voice_id=data.name, gender=data.gender)
        if data.provider == "elevenlabs":
            if data.metadata and data.metadata.voice_id:
                return VoiceLookupResult(provider="elevenlabs", voice_id=data.metadata.voice_id, gender=data.gender)
            conv.line("WARN", "TTS", f"ElevenLabs voice '{voice_name}' has no metadata.voiceId")
            return None
        conv.line("WARN", "TTS", f"Unknown provider '{data.provider}' for voice '{voice_name}'")
        return None
    except Exception as e:  # noqa: BLE001
        conv.line("ERROR", "TTS", f"Error fetching voice details for '{voice_name}': {e}")
        return None
