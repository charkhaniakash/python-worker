"""Persona API — fetch the persona definition for a given personaId."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from pydantic import BaseModel, Field

from ..logging import ConvLog
from ..settings import get_settings
from .client import get_backend_client


class PersonaData(BaseModel):
    """Raw persona payload — every field optional; consumers apply their own defaults."""

    model_config = {"populate_by_name": True, "extra": "allow"}

    id: str | None = None
    name: str | None = None
    llm_provider: str | None = Field(default=None, alias="llmProvider")
    llm_model: str | None = Field(default=None, alias="llmModel")
    llm_model_id: str | None = Field(default=None, alias="llmModelId")
    tts_provider: str | None = Field(default=None, alias="ttsProvider")
    tts_voice: str | None = Field(default=None, alias="ttsVoice")
    tts_voice_id: str | None = Field(default=None, alias="ttsVoiceId")
    voice: str | None = None
    language_preference: list[str] | None = Field(default=None, alias="languagePreference")
    default_language: str | None = Field(default=None, alias="defaultLanguage")
    system_prompt: str | None = Field(default=None, alias="systemPrompt")
    greeting_message: str | None = Field(default=None, alias="greetingMessage")
    temperature: float | str | int | None = None


async def fetch_persona(
    persona_id: str,
    *,
    ids: dict[str, str | None],
    conv: ConvLog,
) -> PersonaData | None:
    """
    Fetch a persona with configurable retry + timeout. Returns None on total
    failure — caller falls back to default config.
    """
    s = get_settings()
    client = get_backend_client()
    cfg = s.runtime.backend.persona_fetch
    path = f"/personas/{persona_id}"

    headers: dict[str, str] = {"Accept": "application/json"}
    for k, alias in (
        ("partnerId", "x-partner-id"),
        ("appId", "x-app-id"),
        ("userId", "x-user-id"),
    ):
        value = ids.get(k)
        if value:
            headers[alias] = value

    conv.line("INFO", "CONFIG", f"Fetching persona {persona_id}", {
        "url": f"{s.backend_url}{path}",
        "headerKeys": list(headers.keys()),
        "hasPartnerId": bool(ids.get("partnerId")),
        "hasAppId": bool(ids.get("appId")),
        "hasUserId": bool(ids.get("userId")),
    })

    payload: dict[str, Any] | None = None
    for attempt in range(1, cfg.attempts + 1):
        started = time.monotonic()
        try:
            resp = await client.get(
                path,
                headers=headers,
                timeout_ms=cfg.per_attempt_timeout_ms,
            )
            latency_ms = int((time.monotonic() - started) * 1000)
            conv.line("INFO", "CONFIG", f"Persona fetch attempt {attempt} completed", {
                "status": resp.status_code,
                "ok": resp.is_success,
                "latencyMs": latency_ms,
                "contentType": resp.headers.get("content-type") or "unknown",
            })
            if resp.is_success:
                payload = resp.json()
                break
            conv.line("WARN", "CONFIG", f"Persona fetch attempt {attempt} → {resp.status_code}")
        except Exception as e:  # noqa: BLE001
            conv.line("WARN", "CONFIG", f"Persona fetch attempt {attempt} failed: {e}", {
                "latencyMs": int((time.monotonic() - started) * 1000),
            })
        if attempt < cfg.attempts:
            await asyncio.sleep((cfg.backoff_base_ms * attempt) / 1000.0)

    if payload is None:
        conv.line("ERROR", "CONFIG", "Persona fetch failed after retries")
        return None

    # The backend may return a list [persona] or the persona directly.
    if isinstance(payload, list):
        payload = payload[0] if payload else None
        if payload is None:
            conv.line("ERROR", "CONFIG", "Persona API returned empty list")
            return None

    conv.line("INFO", "CONFIG", "Persona API response parsed", {
        "shape": "list" if isinstance(payload, list) else type(payload).__name__,
    })

    data = PersonaData.model_validate(payload)
    conv.line("INFO", "CONFIG", "Persona API fields", {
        "id": data.id or persona_id,
        "name": data.name,
        "llmProvider": data.llm_provider,
        "llmModel": data.llm_model,
        "llmModelId": data.llm_model_id,
        "ttsProvider": data.tts_provider,
        "ttsVoice": data.tts_voice,
        "ttsVoiceId": data.tts_voice_id,
        "languagePreference": data.language_preference or [],
        "defaultLanguage": data.default_language,
        "hasSystemPrompt": bool(data.system_prompt),
        "systemPromptChars": len(data.system_prompt or ""),
        "hasGreeting": bool(data.greeting_message),
        "greetingChars": len(data.greeting_message or ""),
    })
    return data
