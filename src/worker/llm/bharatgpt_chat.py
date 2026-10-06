"""
Custom LiveKit LLM that streams from BharatGPT's SSE chat backend.

The endpoint speaks a small event stream:
  event: reasoning   → status ping (not spoken)
  event: token       → { "delta": "..." }  incremental assistant text
  event: done        → { "finalAnswer": "...", ... }  end-of-turn payload

The LiveKit `llm.LLM` contract expects us to push `ChatChunk` items onto the
stream's queue as text arrives, then terminate. If no tokens were produced
we raise `APIConnectionError(retryable=True)` so a fallback adapter can try
the next LLM.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
from livekit.agents import (
    APIConnectionError,
    APIStatusError,
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    llm,
)


@dataclass
class ChatDonePayload:
    id: str | None = None
    finalAnswer: str | None = None
    source: str | None = None
    latencyMs: int | None = None
    sessionId: str | None = None


@dataclass
class BharatGptChatOptions:
    chat_url: str
    session_id: str
    app_id: str
    partner_id: str
    user_id: str | None = None
    persona_id: str | None = None
    language: str = "en"
    timezone: str = "Asia/Calcutta"
    internal_api_key: str | None = None
    attempt_timeout_seconds: float = 30.0
    llm_response: bool = True
    use_internet: bool | None = None
    input_type: str = "voice"
    channel: str = "telephony"
    on_request_start: Callable[[], None] | None = field(default=None)
    on_reasoning: Callable[[str, str | None], None] | None = field(default=None)
    on_done: Callable[[ChatDonePayload], None] | None = field(default=None)


def _extract_latest_user_message(chat_ctx: llm.ChatContext) -> str | None:
    for item in reversed(list(chat_ctx.items)):
        if getattr(item, "type", None) != "message":
            continue
        role = getattr(item, "role", None)
        if role != "user":
            continue
        text = getattr(item, "text_content", None) or getattr(item, "textContent", None)
        if callable(text):
            text = text()
        if isinstance(text, str) and text.strip():
            return text.strip()
    return None


def _localize_language_text(raw: str, language: str) -> str:
    """
    The /chat backend may return persona fallback/greeting messages as a
    language-keyed JSON dict, e.g. {"en": "Hello", "kn": "ನಮಸ್ಕಾರ", "hi": "नमस्ते"}.
    When it does, resolve to the string for the requested language (fallback:
    base code, 'en', then the first key) so TTS never reads the raw dict in all
    configured languages. Any other text passes through unchanged.
    """
    text = (raw or "").strip()
    if not text.startswith("{"):
        return raw
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return raw
    if not isinstance(parsed, dict):
        return raw
    lang = (language or "").lower()
    base = lang.split("-")[0]
    for key in (lang, base, "en"):
        val = parsed.get(key)
        if isinstance(val, str) and val.strip():
            return val
    for val in parsed.values():
        if isinstance(val, str) and val.strip():
            return val
    return raw


def _parse_sse_block(block: str) -> tuple[str, str] | None:
    event = "message"
    data_lines: list[str] = []
    for line in block.splitlines():
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if not data_lines:
        return None
    return event, "\n".join(data_lines)


async def _iter_sse_events(response: httpx.Response) -> AsyncIterator[tuple[str, str]]:
    buffer = ""
    async for chunk in response.aiter_text():
        if not chunk:
            continue
        buffer += chunk
        while True:
            boundary = buffer.find("\n\n")
            if boundary == -1:
                break
            block = buffer[:boundary].strip()
            buffer = buffer[boundary + 2:]
            if block:
                parsed = _parse_sse_block(block)
                if parsed:
                    yield parsed
    tail = buffer.strip()
    if tail:
        parsed = _parse_sse_block(tail)
        if parsed:
            yield parsed


class BharatGptChatLLM(llm.LLM):
    """LiveKit `llm.LLM` shim over the BharatGPT SSE chat endpoint."""

    def __init__(self, opts: BharatGptChatOptions) -> None:
        super().__init__()
        self._opts = opts

    @property
    def model(self) -> str:  # type: ignore[override]
        return "agentic_builder/chat"

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[Any] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: bool | None = None,
        tool_choice: Any = None,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> "BharatGptChatStream":
        del tools, parallel_tool_calls, tool_choice, extra_kwargs
        return BharatGptChatStream(
            self,
            chat_ctx=chat_ctx,
            conn_options=conn_options,
            opts=self._opts,
        )


class BharatGptChatStream(llm.LLMStream):
    def __init__(
        self,
        llm_instance: BharatGptChatLLM,
        *,
        chat_ctx: llm.ChatContext,
        conn_options: APIConnectOptions,
        opts: BharatGptChatOptions,
    ) -> None:
        super().__init__(llm_instance, chat_ctx=chat_ctx, tools=[], conn_options=conn_options)
        self._opts = opts

    async def _run(self) -> None:
        message = _extract_latest_user_message(self._chat_ctx)
        if not message:
            raise APIConnectionError(
                "No user message found in chat context",
                retryable=False,
            )

        headers: dict[str, str] = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "x-app-id": self._opts.app_id,
            "x-partner-id": self._opts.partner_id,
        }
        if self._opts.user_id:
            headers["x-user-id"] = self._opts.user_id
        if self._opts.persona_id:
            headers["x-persona-id"] = self._opts.persona_id
        if self._opts.internal_api_key:
            headers["x-internal-api-key"] = self._opts.internal_api_key

        body = {
            "message": message,
            "language": self._opts.language,
            "sessionId": self._opts.session_id,
            "llmResponse": self._opts.llm_response,
            "useInternet": self._opts.use_internet if self._opts.use_internet is not None else False,
            "inputType": self._opts.input_type,
            "timezone": self._opts.timezone,
        }
        if self._opts.channel:
            body["channel"] = self._opts.channel

        chunk_id = str(uuid.uuid4())
        sent_any_token = False
        final_answer_text = ""
        _json_buf = ""  # accumulates token deltas that look like a JSON dict

        timeout = httpx.Timeout(
            self._opts.attempt_timeout_seconds,
            connect=self._opts.attempt_timeout_seconds,
        )

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                if self._opts.on_request_start is not None:
                    self._opts.on_request_start()
                
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream(
                        "POST",
                        self._opts.chat_url,
                        headers=headers,
                        json=body,
                    ) as response:
                        if response.status_code >= 400:
                            err_body = await response.aread()
                            raw = err_body.decode(errors="replace")[:500] if err_body else None
                            raise APIStatusError(
                                f"BharatGPT chat API returned {response.status_code}",
                                status_code=response.status_code,
                                body={"raw": raw} if raw else None,
                                retryable=response.status_code >= 500 or response.status_code == 429,
                            )

                        async for event_name, data in _iter_sse_events(response):
                            if event_name == "reasoning":
                                step = data
                                source: str | None = None
                                try:
                                    payload = json.loads(data)
                                    if isinstance(payload, dict):
                                        step = payload.get("step") or data
                                        source = payload.get("source")
                                except json.JSONDecodeError:
                                    pass
                                if self._opts.on_reasoning is not None:
                                    self._opts.on_reasoning(step, source)
                                continue

                            if event_name == "token":
                                try:
                                    payload = json.loads(data)
                                except json.JSONDecodeError:
                                    continue
                                delta = payload.get("delta") if isinstance(payload, dict) else None
                                if not delta:
                                    continue

                                # --- Multi-language JSON buffering ---
                                # The /chat backend sometimes returns a language-keyed
                                # JSON dict like {"en":"...", "hi":"..."} streamed as
                                # individual token deltas.  Each fragment isn't valid
                                # JSON on its own, so _localize_language_text fails on
                                # each piece and the raw JSON leaks to TTS.
                                #
                                # Strategy: if the accumulated text starts with '{',
                                # keep buffering until we can parse it. Once parsed,
                                # localize and flush.  If a delta arrives that doesn't
                                # look like JSON continuation, flush the buffer as-is
                                # and send the new delta normally.
                                _json_buf += delta

                                # Try to localize the *full* buffer so far.
                                localized = _localize_language_text(
                                    _json_buf, self._opts.language
                                )

                                if localized != _json_buf:
                                    # Successfully extracted a language string from
                                    # the accumulated JSON — flush the localized text.
                                    sent_any_token = True
                                    final_answer_text += localized
                                    self._event_ch.send_nowait(
                                        llm.ChatChunk(
                                            id=chunk_id,
                                            delta=llm.ChoiceDelta(
                                                role="assistant", content=localized,
                                            ),
                                        )
                                    )
                                    _json_buf = ""
                                elif not _json_buf.lstrip().startswith("{"):
                                    # Not a JSON object at all — flush immediately.
                                    sent_any_token = True
                                    final_answer_text += _json_buf
                                    self._event_ch.send_nowait(
                                        llm.ChatChunk(
                                            id=chunk_id,
                                            delta=llm.ChoiceDelta(
                                                role="assistant", content=_json_buf,
                                            ),
                                        )
                                    )
                                    _json_buf = ""
                                # else: still accumulating a potential JSON dict — wait.
                                continue

                            if event_name == "done":
                                # --- Flush any remaining JSON buffer ---
                                if _json_buf:
                                    flushed = _localize_language_text(
                                        _json_buf, self._opts.language
                                    )
                                    if flushed:
                                        sent_any_token = True
                                        final_answer_text += flushed
                                        self._event_ch.send_nowait( 
                                            llm.ChatChunk(
                                                id=chunk_id,
                                                delta=llm.ChoiceDelta(
                                                    role="assistant", content=flushed,
                                                ),
                                            )
                                        )
                                    _json_buf = ""

                                try:
                                    payload = json.loads(data) if data else {}
                                except json.JSONDecodeError:
                                    payload = {}
                                if not isinstance(payload, dict):
                                    payload = {}
                                done_evt = ChatDonePayload(
                                    id=payload.get("id"),
                                    finalAnswer=payload.get("finalAnswer"),
                                    source=payload.get("source"),
                                    latencyMs=payload.get("latencyMs"),
                                    sessionId=payload.get("sessionId"),
                                )
                                if self._opts.on_done is not None:
                                    self._opts.on_done(done_evt)

                                answer = _localize_language_text(
                                    (done_evt.finalAnswer or "").strip(),
                                    self._opts.language,
                                )
                                if answer and not sent_any_token:
                                    self._event_ch.send_nowait(
                                        llm.ChatChunk(
                                            id=chunk_id,
                                            delta=llm.ChoiceDelta(role="assistant", content=answer),
                                        )
                                    )
                                    sent_any_token = True
                                    final_answer_text = answer
                                elif answer and len(answer) > len(final_answer_text):
                                    remainder = answer[len(final_answer_text):]
                                    if remainder:
                                        self._event_ch.send_nowait( 
                                            llm.ChatChunk(
                                                id=chunk_id,
                                                delta=llm.ChoiceDelta(
                                                    role="assistant", content=remainder
                                                ),
                                            )
                                        )
                                    final_answer_text = answer
                                break

                if not sent_any_token:
                    raise APIConnectionError(
                        "BharatGPT chat API returned no assistant tokens",
                        retryable=True,
                    )
                break

            except (httpx.TimeoutException, httpx.HTTPError, APIStatusError, APIConnectionError) as exc:
                retryable = True
                if isinstance(exc, (APIStatusError, APIConnectionError)):
                    retryable = getattr(exc, "retryable", True)

                if not sent_any_token and retryable and attempt < max_attempts:
                    backoff_delay = 1.0 * attempt
                    await asyncio.sleep(backoff_delay)
                    continue
                else: 
                    if isinstance(exc, httpx.TimeoutException):
                        raise APIConnectionError(
                            f"BharatGPT chat request timed out: {exc}",
                            retryable=False if sent_any_token else True,
                        ) from exc
                    elif isinstance(exc, httpx.HTTPError):
                        raise APIConnectionError(
                            f"BharatGPT chat request failed: {exc}",
                            retryable=False if sent_any_token else True,
                        ) from exc
                    else:
                        raise exc
"