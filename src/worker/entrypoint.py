"""
Worker entrypoint — one job per SIP participant.

Flow:
  1. Connect to the room, wait for the participant to join.
  2. Resolve appId/partnerId/personaId from attributes → metadata → phone lookup.
  3. Create the inbound CallLog (or read the outbound one from attributes).
  4. Build AgentConfig from persona (falls back to defaults on failure).
  5. For outbound: block until the callee answers (or bail out on rejection).
  6. Build STT/LLM/TTS strictly from the persona's selected provider — a
     Google-TTS persona never touches ElevenLabs (providers are never mixed).
  7. Start AgentSession, speak greeting, block until the call ends.
  8. Report per-call usage to the backend.
  9. Force-exit the job subprocess so the worker recycles for the next call.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from livekit.agents import (
    Agent,
    AgentSession,
    EndpointingOptions,
    InterruptionOptions,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    TurnHandlingOptions,
    metrics,
)
from livekit.agents.voice.background_audio import (
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
)
from livekit.plugins import noise_cancellation, silero

from .agent import TelephonyAssistant
from .backend.call_log import create_inbound_call_log, report_call_usage
from .backend.phone_lookup import fetch_app_info_by_number
from .config_builder import (
    AgentConfig,
    apply_outbound_call_opening,
    build_config_from_persona,
    default_agent_config,
    resolve_chat_language,
    validate_config,
)
from .llm.factory import build_session_llm, describe_llm
from .llm.model_resolver import llm_provider_key
from .logging import ConvLog, truncate_for_log
from .settings import get_settings

# Built-in ambient clips shipped with livekit-plugins — names map to
# config/defaults.yaml `ambience.sound`.
_AMBIENCE_CLIPS: dict[str, BuiltinAudioClip] = {
    "crowded_room": BuiltinAudioClip.CROWDED_ROOM,
    "office_ambience": BuiltinAudioClip.OFFICE_AMBIENCE,
    "city_ambience": BuiltinAudioClip.CITY_AMBIENCE,
    "forest_ambience": BuiltinAudioClip.FOREST_AMBIENCE,
    "hold_music": BuiltinAudioClip.HOLD_MUSIC,
    "keyboard_typing": BuiltinAudioClip.KEYBOARD_TYPING,
    "keyboard_typing2": BuiltinAudioClip.KEYBOARD_TYPING2,
}
from .stt.factory import build_stt
from .telephony import (
    is_outbound_leg,
    is_telephony_participant,
    sanitize_for_tts,
    wait_for_call_end,
    wait_for_sip_answer,
)
from .telephony.signaling import with_timeout
from .tts.factory import ResolvedTtsConfig, build_tts_from_config
from .usage import CallUsageAccumulator, format_pipeline_metrics, stt_provider_key, tts_label_to_provider_key
from .usage.metrics_bridge import (
    fallback_tts_provider_key,
    llm_tokens,
    stt_audio_ms,
    tts_characters,
)
from .voice.map import resolve_gemini_tts_voice


# ---------------------------------------------------------------------------
# Prewarm
# ---------------------------------------------------------------------------


def prewarm(proc: JobProcess) -> None:
    s = get_settings().runtime.vad
    kwargs: dict[str, Any] = {
        "min_speech_duration": s.min_speech_duration,
        "min_silence_duration": s.min_silence_duration,
        "activation_threshold": s.activation_threshold,
    }
    if s.deactivation_threshold is not None:
        kwargs["deactivation_threshold"] = s.deactivation_threshold
    if s.prefix_padding_duration is not None:
        kwargs["prefix_padding_duration"] = s.prefix_padding_duration
    if s.sample_rate is not None:
        kwargs["sample_rate"] = s.sample_rate
    proc.userdata["vad"] = silero.VAD.load(**kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_ids_from_participant(participant: Any) -> dict[str, str | None]:
    attrs = getattr(participant, "attributes", None) or {}
    ids: dict[str, str | None] = {
        "partnerId": attrs.get("partner_id"),
        "personaId": attrs.get("persona_id"),
        "appId": attrs.get("app_id"),
        "userId": attrs.get("user_id"),
    }
    metadata = getattr(participant, "metadata", None)
    if metadata:
        try:
            md = json.loads(metadata)
            ids["partnerId"] = ids["partnerId"] or md.get("partnerId") or md.get("partner_id")
            ids["personaId"] = ids["personaId"] or md.get("personaId") or md.get("persona_id")
            ids["appId"] = ids["appId"] or md.get("appId") or md.get("app_id")
            ids["userId"] = ids["userId"] or md.get("userId") or md.get("user_id")
        except json.JSONDecodeError:
            pass
    return ids


def _extract_call_log_id(participant: Any) -> str | None:
    attrs = getattr(participant, "attributes", None) or {}
    if attrs.get("sip.callLogId"):
        return attrs["sip.callLogId"]
    metadata = getattr(participant, "metadata", None)
    if metadata:
        try:
            md = json.loads(metadata)
            return md.get("call_log_id") or md.get("callLogId")
        except json.JSONDecodeError:
            return None
    return None


def _teardown_and_exit(conv: ConvLog, reason: str, code: int = 0) -> None:
    delay_ms = get_settings().runtime.teardown.exit_delay_ms
    conv.line(
        "INFO",
        "SESSION",
        f"Teardown ({reason}) — exiting job subprocess to release worker",
    )
    # Give logs / shutdown callbacks a brief moment to flush, then hard-exit so
    # a slow or crashing native teardown can't wedge the worker for future calls.
    loop = asyncio.get_event_loop()
    loop.call_later(delay_ms / 1000.0, lambda: __import__("os")._exit(code))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


async def entrypoint(ctx: JobContext) -> None:
    settings = get_settings()
    conv = ConvLog.create(ctx.room.name, ctx.job.id if ctx.job else None)
    
    # Log agent dispatch metadata FIRST
    agent_metadata = {}
    if ctx.job and hasattr(ctx.job, 'agent_dispatch'):
        dispatch = getattr(ctx.job, 'agent_dispatch', None)
        if dispatch and hasattr(dispatch, 'metadata'):
            metadata_str = getattr(dispatch, 'metadata', '') or ''
            if metadata_str:
                try:
                    agent_metadata = json.loads(metadata_str)
                    conv.line("INFO", "SESSION", "Agent dispatch metadata", agent_metadata)
                except json.JSONDecodeError:
                    pass
    
    conv.line("INFO", "SESSION", "Telephony job started", {
        "room": ctx.room.name,
        "jobId": ctx.job.id if ctx.job else None,
        "hasAgentMetadata": bool(agent_metadata),
    })

    await ctx.connect()
    conv.line("INFO", "SESSION", "Connected to room — waiting for caller…")

    participant = await ctx.wait_for_participant()
    is_telephony = is_telephony_participant(participant)
    is_outbound = is_outbound_leg(participant)
    conv.line("INFO", "SESSION", "Participant joined", {
        "identity": participant.identity,
        "telephony": is_telephony,
        "direction": "outbound" if is_outbound else "inbound",
        "callStatus": (participant.attributes or {}).get("sip.callStatus"),
    })

    # ── Hold music — play immediately while the agent pipeline loads ──
    # tone (playDialtone=true on createSipParticipant).
    hold_music_player: BackgroundAudioPlayer | None = None
    if is_telephony and not is_outbound:
        try:
            hold_music_player = BackgroundAudioPlayer(
                ambient_sound=AudioConfig(
                    BuiltinAudioClip.HOLD_MUSIC,
                    volume=0.4,
                    fade_in=0.5,
                ),
            )
            await hold_music_player.start(room=ctx.room)
            conv.line("INFO", "AUDIO", "Hold music started — agent pipeline loading")

        except Exception as e:  # noqa: BLE001
            conv.line("WARN", "AUDIO", f"Hold music unavailable: {e}")
            hold_music_player = None

    # --- Resolve routing identity ---
    ids = _extract_ids_from_participant(participant)
    
    # ALSO try to get IDs from agent dispatch metadata (sent by backend)
    if agent_metadata:
        ids["appId"] = ids["appId"] or agent_metadata.get("appId") or agent_metadata.get("app_id")
        ids["partnerId"] = ids["partnerId"] or agent_metadata.get("partnerId") or agent_metadata.get("partner_id")
        ids["personaId"] = ids["personaId"] or agent_metadata.get("personaId") or agent_metadata.get("persona_id")
    
    conv.line("INFO", "SESSION", "Initial IDs extracted", {
        "ids": ids,
        "attributes": dict(participant.attributes or {}),
        "metadata": participant.metadata,
    })
    
    if not (ids.get("appId") and ids.get("partnerId") and ids.get("personaId")):
        attrs = participant.attributes or {}
        called = (
            attrs.get("sip.trunkPhoneNumber")
            or attrs.get("calledNumber")
            or attrs.get("sip.phoneNumber")
        )
        conv.line("INFO", "SESSION", "Inbound number resolution", {
            "identity": participant.identity,
            "sip.trunkPhoneNumber": attrs.get("sip.trunkPhoneNumber"),
            "calledNumber": attrs.get("calledNumber"),
            "sip.phoneNumber": attrs.get("sip.phoneNumber"),
            "usedCalled": called,
        })
        if called:
            info = await fetch_app_info_by_number(called, conv)
            if info and info.success:
                ids["appId"] = ids["appId"] or info.appId
                ids["partnerId"] = ids["partnerId"] or info.partnerId
                ids["personaId"] = ids["personaId"] or info.personaId
                conv.line("INFO", "SESSION", "Resolved via phone lookup", ids)

    conv.line("INFO", "SESSION", "Final Call context", ids)

    # --- Resolve callLogId (for cost tracking) ---
    call_log_id = _extract_call_log_id(participant)
    if not call_log_id and not is_outbound and ids.get("appId") and ids.get("partnerId"):
        attrs = participant.attributes or {}
        call_log_id = await create_inbound_call_log(
            app_id=ids["appId"],  # type: ignore[arg-type]
            partner_id=ids["partnerId"],  # type: ignore[arg-type]
            from_number=attrs.get("sip.phoneNumber"),
            to_number=attrs.get("sip.trunkPhoneNumber"),
            call_sid=attrs.get("sip.callID"),
            conv=conv,
        )

    chat_session_id = str(uuid.uuid4())
    use_chat_backend = (
        settings.runtime.chat.enabled
        and settings.env.bharatgpt_chat_enabled() is not False
        and bool(ids.get("appId") and ids.get("partnerId"))
    )
    if use_chat_backend:
        conv.line("INFO", "SESSION", "BharatGPT chat session", {
            "chatSessionId": chat_session_id,
            "chatUrl": settings.chat_api_url,
        })

    # --- Build pipeline config ---
    config: AgentConfig | None = None
    if ids.get("personaId"):
        try:
            config = await build_config_from_persona(
                ids["personaId"],  # type: ignore[arg-type]
                ids=ids,
                conv=conv,
            )
        except Exception as e:  # noqa: BLE001
            conv.line("ERROR", "CONFIG", f"Persona config build failed: {e}")
    if config is None:
        config = default_agent_config(conv)

    if is_outbound:
        apply_outbound_call_opening(config)

    validate_config(config, conv)

    # --- Build TTS + kick off greeting pre-synthesis BEFORE the outbound
    # SIP-answer wait (not after) — this is what actually makes "ringing phase
    # = free time" true. `config` is already fully resolved at this point, so
    # nothing here depends on the callee having answered yet.
    # ONE TTS instance for the entire call, strictly from the persona's selected
    # provider — a Google-TTS persona uses Google for the greeting too (no
    # ElevenLabs greeting, no mid-call swap).
    tts: Any
    if config.tts.provider == "google":
        resolved = resolve_gemini_tts_voice(config.tts.voice_id, conv=conv)
        tts = build_tts_from_config(
            ResolvedTtsConfig(
                provider="google",
                voice_id=resolved.voice,
                model=config.tts.model,
            )
        )
    else:
        tts = build_tts_from_config(config.tts)

    # Pre-synthesize the greeting now — tts is built, and for outbound calls
    # we're about to wait for SIP answer (ringing phase = free time). For
    # inbound calls the session setup takes ~100ms which also overlaps nicely.
    # Result: audio frames are ready by the time session.say() is called,
    # eliminating the TTS TTFB the caller would otherwise hear as silence
    # after pickup — this task is created BEFORE the outbound wait below so
    # it genuinely runs concurrently with the ringing tone, not after it.
    greeting = sanitize_for_tts(config.initial_message)
    greeting_tts = tts  # greeting uses the same selected-provider TTS as the conversation

    async def _presynthesise_greeting() -> list[Any]:
        """Synthesize greeting audio frames in the background. Returns [] on failure."""
        try:
            stream = greeting_tts.synthesize(greeting)
            frames: list[Any] = []
            async for ev in stream:
                frame = getattr(ev, "frame", ev)
                frames.append(frame)
            conv.line("INFO", "TTS", f"Greeting pre-synthesized ({len(frames)} frames ready)")
            
            # Stop hold music as soon as greeting audio is ready — caller hears the
            # greeting immediately instead of waiting for pipeline to fully initialize
            nonlocal hold_music_player
            if hold_music_player is not None:
                try:
                    await hold_music_player.aclose()
                    conv.line("INFO", "AUDIO", "Hold music stopped — greeting audio ready")
                except Exception:  # noqa: BLE001
                    pass
                hold_music_player = None
            
            return frames
        except Exception as e:  # noqa: BLE001
            conv.line("WARN", "TTS", f"Greeting pre-synthesis failed, will synthesize on demand: {e}")
            return []

    presyn_task: asyncio.Task[list[Any]] = asyncio.create_task(_presynthesise_greeting())

    # --- Outbound: wait for the callee to actually answer ---
    if is_outbound:
        answered = await wait_for_sip_answer(
            ctx.room,
            participant,
            conv,
            timeout_ms=settings.runtime.session.sip_answer_timeout_ms,
            poll_interval_ms=settings.runtime.session.sip_answer_poll_interval_ms,
        )
        if not answered:
            conv.line("WARN", "SESSION", "Call not answered — ending job without greeting")
            presyn_task.cancel()
            ctx.shutdown(reason="call-not-answered")
            return

    # --- Build STT / LLM ---
    stt = build_stt(config.stt.language, is_telephony=is_telephony)

    llm_instance = build_session_llm(
        resolved_model_id=config.llm.model,
        temperature=config.llm.temperature,
        chat_language=resolve_chat_language(config),
        ids=ids,
        chat_session_id=chat_session_id,
        conv=conv,
    )

    # Greeting uses the exact same voice as the conversation (no mixing).
    active_greeting_voice_id = config.tts.voice_id
    sample_rate = (
        settings.runtime.stt.sample_rate_hz["telephony"]
        if is_telephony
        else settings.runtime.stt.sample_rate_hz["non_telephony"]
    )
    conv.line("INFO", "PIPELINE", "Voice pipeline ready", {
        "agent": config.agent_name,
        "stt": f"{config.stt.model}/{config.stt.language}@{sample_rate}Hz",
        "llm": "bharatgpt-chat-sse" if use_chat_backend else config.llm.model,
        "tts": f"{config.tts.provider}/{active_greeting_voice_id}",
        "telephony": is_telephony,
    })

    conv.line("INFO", "SESSION", "Creating AgentSession", {
        "llmType": describe_llm(llm_instance),
        "hasVad": bool(ctx.proc.userdata.get("vad")),
        "turnDetection": "vad",
    })

    session_cfg = settings.runtime.session
    session = AgentSession(
        llm=llm_instance,
        stt=stt,
        tts=tts,
        vad=ctx.proc.userdata["vad"],
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            endpointing=EndpointingOptions(
                mode="fixed",
                min_delay=session_cfg.min_endpointing_delay_ms / 1000.0,
                max_delay=session_cfg.max_endpointing_delay_ms / 1000.0,
            ),
            interruption=InterruptionOptions(
                enabled=session_cfg.allow_interruptions,
                mode=session_cfg.interruption_mode,
                discard_audio_if_uninterruptible=session_cfg.discard_audio_if_uninterruptible,
                min_duration=session_cfg.min_interruption_duration_ms / 1000.0,
                min_words=session_cfg.min_interruption_words,
            ),
        ),
    )
    conv.line("INFO", "SESSION", "AgentSession created")

    # --- Metrics + state ---
    usage_collector = metrics.UsageCollector()
    usage_acc = CallUsageAccumulator()
    call_started_at = time.monotonic()

    def _on_metrics_collected(ev: Any) -> None:
        m = getattr(ev, "metrics", ev)
        try:
            usage_collector.collect(m)
        except Exception:  # noqa: BLE001
            pass
        m_type = getattr(m, "type", None)
        if m_type == "stt_metrics":
            usage_acc.add_stt(stt_provider_key(), stt_audio_ms(m))
        elif m_type == "llm_metrics":
            prompt, completion = llm_tokens(m)
            usage_acc.add_llm(llm_provider_key(config.llm.model), prompt, completion)
        elif m_type == "tts_metrics":
            label = getattr(m, "label", None)
            provider_key = tts_label_to_provider_key(label, fallback_tts_provider_key(config.tts.provider))
            usage_acc.add_tts(provider_key, tts_characters(m))
        summary = format_pipeline_metrics(m)
        if summary:
            conv.line("INFO", "METRICS", summary)

    def _on_error(ev: Any) -> None:
        err = getattr(ev, "error", ev)
        conv.line("ERROR", "PIPELINE", "Agent session error", {
            "error": str(err),
        })

    def _on_user_state_changed(ev: Any) -> None:
        conv.line(
            "INFO",
            "VOICE",
            f"User state: {getattr(ev, 'old_state', '?')} → {getattr(ev, 'new_state', '?')}",
        )

    def _on_user_input_transcribed(ev: Any) -> None:
        is_final = getattr(ev, "is_final", False)
        transcript = getattr(ev, "transcript", "") or ""
        if not is_final:
            if settings.verbose_conv_log and transcript:
                conv.line("INFO", "STT", f"Interim: {truncate_for_log(transcript, 120)}")
            return
        conv.bump_turn()
        conv.line(
            "USER",
            "VOICE",
            truncate_for_log(transcript),
            {"lang": getattr(ev, "language", None) or "unknown"},
        )

    def _on_conversation_item_added(ev: Any) -> None:
        item = getattr(ev, "item", None)
        if item is None:
            return
        role = getattr(item, "role", None)
        if role == "assistant":
            text = getattr(item, "text_content", None) or getattr(item, "textContent", None) or ""
            if callable(text):
                text = text()
            conv.line("AGENT", "VOICE", truncate_for_log(text))

    session.on("metrics_collected", _on_metrics_collected)
    session.on("error", _on_error)
    session.on("user_state_changed", _on_user_state_changed)
    session.on("user_input_transcribed", _on_user_input_transcribed)
    session.on("conversation_item_added", _on_conversation_item_added)

    async def _log_usage_summary() -> None:
        try:
            summary_obj = usage_collector.get_summary()
        except Exception as e:  # noqa: BLE001
            conv.line("WARN", "SESSION", f"Usage summary unavailable: {e}")
            return
        conv.line("INFO", "SESSION", "Usage summary", {"summary": str(summary_obj)})

    ctx.add_shutdown_callback(_log_usage_summary)

    # --- Start the session ---
    try:
        # RoomInputOptions is a dataclass — pass optional fields via constructor
        # rather than mutating, so its type checks (and any future frozen
        # variant) keep working.
        room_input_kwargs: dict[str, Any] = {"participant_identity": participant.identity}
        if is_telephony and settings.telephony_nc_enabled:
            room_input_kwargs["noise_cancellation"] = noise_cancellation.BVCTelephony()
        input_opts = RoomInputOptions(**room_input_kwargs)

        output_opts: RoomOutputOptions | None = None
        if is_telephony:
            output_opts = RoomOutputOptions(
                audio_sample_rate=session_cfg.telephony_output_sample_rate_hz,
                audio_num_channels=session_cfg.telephony_output_num_channels,
            )

        active_agent = TelephonyAssistant(config, use_chat_backend=use_chat_backend)
        start_coro = session.start(
            agent=active_agent,
            room=ctx.room,
            room_input_options=input_opts,
            room_output_options=output_opts,
        )
        await with_timeout(
            asyncio.ensure_future(start_coro),
            session_cfg.start_timeout_ms,
            "session.start()",
        )
        conv.line("INFO", "SESSION", "AgentSession started", {"participant": participant.identity})
        conv.line("INFO", "SESSION", "══════ Conversation live ══════")

        # --- Background ambience (e.g. agent sounds like it's in a crowded room) ---
        ambience = settings.runtime.ambience
        ambience_player: BackgroundAudioPlayer | None = None
        if is_telephony and ambience.enabled:
            clip = _AMBIENCE_CLIPS.get(ambience.sound.lower())
            if clip is None:
                conv.line(
                    "WARN",
                    "AUDIO",
                    f"Unknown ambience sound '{ambience.sound}' — skipping background audio",
                )
            else:
                try:
                    ambience_player = BackgroundAudioPlayer(
                        ambient_sound=AudioConfig(clip, volume=ambience.volume, fade_in=1.0),
                    )
                    await ambience_player.start(room=ctx.room)
                    conv.line("INFO", "AUDIO", "Background ambience playing", {
                        "sound": ambience.sound,
                        "volume": ambience.volume,
                    })
                except Exception as e:  # noqa: BLE001
                    conv.line("WARN", "AUDIO", f"Background ambience unavailable: {e}")
                    ambience_player = None

        conv.line("AGENT", "VOICE", truncate_for_log(greeting), {
            "type": "greeting",
            "direction": "outbound" if is_outbound else "inbound",
        })

        async def _play_greeting() -> None:
            try:
                # Use pre-synthesized frames if available — zero TTFB for the caller.
                # Falls back to text if pre-synthesis failed or was cancelled.
                presynth_frames = await presyn_task
                if presynth_frames:
                    async def _frames_iter() -> Any:
                        for f in presynth_frames:
                            yield f
                    handle = session.say(greeting, audio=_frames_iter(), allow_interruptions=True)
                else:
                    # Fallback: stop hold music now if pre-synthesis failed
                    nonlocal hold_music_player
                    if hold_music_player is not None:
                        try:
                            await hold_music_player.aclose()
                            conv.line("INFO", "AUDIO", "Hold music stopped — greeting synthesizing on-demand")
                        except Exception:  # noqa: BLE001
                            pass
                        hold_music_player = None
                    
                    handle = session.say(greeting, allow_interruptions=True)
                wait_fn = getattr(handle, "wait_for_playout", None)
                if callable(wait_fn):
                    result = wait_fn()
                    if asyncio.iscoroutine(result):
                        await result
                conv.line("INFO", "TTS", "Greeting playback finished")
            except Exception as e:  # noqa: BLE001
                conv.line("ERROR", "TTS", f"Greeting failed: {e}")

        asyncio.create_task(_play_greeting())

        await wait_for_call_end(ctx.room, conv)

        if hold_music_player is not None:
            try:
                await hold_music_player.aclose()
            except Exception:  # noqa: BLE001
                pass

        if ambience_player is not None:
            try:
                await ambience_player.aclose()
            except Exception:  # noqa: BLE001
                pass

        if call_log_id:
            duration_seconds = int(time.monotonic() - call_started_at)
            await report_call_usage(
                usage_acc.to_payload(call_log_id, duration_seconds),
                conv,
            )
        else:
            conv.line("WARN", "COST", "No callLogId for this call — cost not tracked")

        _teardown_and_exit(conv, "call-ended")
    except Exception as err:  # noqa: BLE001
        conv.line("ERROR", "SESSION", f"Session failed: {err}")
        try:
            await session.aclose()
        except Exception:  # noqa: BLE001
            pass
        _teardown_and_exit(conv, "error", 1)
        raise
