"""
Build the LLM instance used by AgentSession.

Three shapes are possible:
  1. `BharatGptChatLLM`                — if appId+partnerId present and chat enabled
  2. `BharatGptChatLLM` + Gemini via `FallbackAdapter` — for auto-recovery on 5xx
  3. `google.LLM` plugin instance      — direct Gemini call, no chat SSE
  4. inference model id string         — LiveKit hosts everything else
"""

from __future__ import annotations

from typing import Any

from livekit.agents import llm as agents_llm
from livekit.plugins import google as google_plugin

from ..logging import ConvLog
from ..settings import get_settings
from .bharatgpt_chat import BharatGptChatLLM, BharatGptChatOptions, ChatDonePayload


def _should_use_chat(app_id: str | None, partner_id: str | None) -> bool:
    if not app_id or not partner_id:
        return False
    settings = get_settings()
    env_flag = settings.env.bharatgpt_chat_enabled()
    if env_flag is False:
        print("[CHAT] BharatGPT chat SSE disabled via USE_BHARATGPT_CHAT=false")
        return False
    return settings.runtime.chat.enabled


def _build_google_llm(model_id: str, temperature: float) -> google_plugin.LLM | None:
    if not model_id.startswith("google/"):
        return None
    google_model = model_id.split("/", 1)[1]
    return google_plugin.LLM(model=google_model, temperature=temperature)


def build_session_llm(
    *,
    resolved_model_id: str,
    temperature: float,
    chat_language: str,
    ids: dict[str, str | None],
    chat_session_id: str,
    conv: ConvLog,
    use_internet: bool | None = None,
) -> agents_llm.LLM | str:
    settings = get_settings()
    app_id = ids.get("appId")
    partner_id = ids.get("partnerId")

    fallback = _build_google_llm(resolved_model_id, temperature)

    if _should_use_chat(app_id, partner_id):
        chat_url = settings.chat_api_url
        assert app_id is not None
        assert partner_id is not None

        def _on_request_start() -> None:
            conv.line("INFO", "CHAT", "Chat SSE request started (waiting on backend)")

        def _on_reasoning(step: str, source: str | None) -> None:
            conv.line("INFO", "CHAT", step, {"source": source})

        def _on_done(payload: ChatDonePayload) -> None:
            conv.line(
                "INFO",
                "CHAT",
                "Response complete",
                {"source": payload.source, "latencyMs": payload.latencyMs},
            )

        chat_cfg = settings.runtime.chat
        chat_llm = BharatGptChatLLM(
            BharatGptChatOptions(
                chat_url=chat_url,
                session_id=chat_session_id,
                app_id=app_id,
                partner_id=partner_id,
                user_id=ids.get("userId"),
                persona_id=ids.get("personaId"),
                language=chat_language,
                timezone=chat_cfg.default_timezone,
                internal_api_key=settings.env.internal_api_key,
                attempt_timeout_seconds=chat_cfg.attempt_timeout_seconds,
                llm_response=chat_cfg.llm_response,
                use_internet=use_internet,
                input_type=chat_cfg.input_type,
                channel=chat_cfg.channel,
                on_request_start=_on_request_start,
                on_reasoning=_on_reasoning,
                on_done=_on_done,
            )
        )

        if fallback is not None:
            conv.line(
                "INFO",
                "PIPELINE",
                "Using BharatGPT chat SSE with Gemini fallback",
                {"chatUrl": chat_url, "sessionId": chat_session_id},
            )
            adapter_cls = getattr(agents_llm, "FallbackAdapter", None)
            if adapter_cls is not None:
                return adapter_cls(  # type: ignore[no-any-return]
                    llm=[chat_llm, fallback],
                    attempt_timeout=settings.runtime.chat.attempt_timeout_seconds,
                    max_retry_per_llm=0,
                )
        conv.line(
            "INFO",
            "PIPELINE",
            "Using BharatGPT chat SSE",
            {"chatUrl": chat_url, "sessionId": chat_session_id},
        )
        return chat_llm

    if fallback is not None:
        conv.line(
            "INFO",
            "PIPELINE",
            f"Using Google LLM plugin directly ({resolved_model_id.split('/', 1)[1]})",
        )
        return fallback

    return resolved_model_id


def describe_llm(instance: Any) -> str:
    if isinstance(instance, str):
        return "inference-string"
    if isinstance(instance, BharatGptChatLLM):
        return "bharatgpt-chat"
    adapter_cls = getattr(agents_llm, "FallbackAdapter", None)
    if adapter_cls is not None and isinstance(instance, adapter_cls):
        return "bharatgpt-chat+fallback"
    return "google-plugin"
