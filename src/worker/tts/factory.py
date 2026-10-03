"""
TTS resolution + factories.

The persona API returns provider/voice hints; we normalise those into a
`ResolvedTtsConfig` (still cheap to construct), then instantiate the actual
plugin via `build_tts_from_config`.

BharatGPT-labelled voices route to Google Chirp 3 TTS (streaming).

Google TTS uses `google.TTS` with `model_name="chirp_3"` and
`use_streaming=True` — this delivers real-time audio chunks to the caller
as synthesis happens. The alternative `google.beta.GeminiTTS` buffers the
entire audio response before sending it, causing 8-12s of silence.

Google TTS credentials are **always** fetched from Secret Manager
(`GOOGLE_CREDENTIALS_SECRET`). On dev machines without Application Default
Credentials, the local SA key at `src/worker/credientials/google.json` is
used only to authenticate the Secret Manager client — it is never used as
the TTS credential itself.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from livekit.plugins import elevenlabs, google, openai
from google.cloud import texttospeech as gtts

from .fish_audio import FishAudioTTS
from ..backend.persona import PersonaData
from ..backend.voice_lookup import fetch_voice_details
from ..logging import ConvLog
from ..llm.model_resolver import is_bharatgpt
from ..settings import get_settings
from ..voice.map import _map_locale_gender_to_gemini, resolve_gemini_tts_voice

# Relative path from repo root — used ONLY to bootstrap the Secret Manager
# client on dev machines without ADC. Never used as TTS credentials directly.
_LOCAL_SA_KEY_REL = Path("src/worker/credientials/google.json")


@lru_cache(maxsize=1)
def _load_google_credentials() -> dict[str, Any]:
    """
    Load Google service-account credentials for TTS.

    Two modes (controlled by ``GOOGLE_CREDENTIALS_USE_LOCAL``):

    * **false / unset (default — production & recommended dev)**:
      Fetches the SA key JSON from Google Secret Manager using the resource
      name in ``GOOGLE_CREDENTIALS_SECRET``.  If Application Default
      Credentials are not available (e.g. plain dev laptop), the local SA key
      at ``src/worker/credientials/google.json`` is used *only* to
      authenticate the Secret Manager call — TTS credentials still come from
      the secret.

    * **true (local dev escape-hatch)**:
      Loads the SA key directly from the local file.  Not recommended — prefer
      Secret Manager even in dev.

    Raises ``RuntimeError`` on any failure with an actionable message.
    """
    settings = get_settings()
    env = settings.env
    local_key_path = (settings.repo_root / _LOCAL_SA_KEY_REL).resolve()

    # ── Escape-hatch: GOOGLE_CREDENTIALS_USE_LOCAL=true ──────────────
    if env.use_local_google_credentials():
        print("[TTS] GOOGLE_CREDENTIALS_USE_LOCAL=true — using local SA key directly (dev mode)")
        return _load_local_key(local_key_path)

    # ── Default path: credentials from Secret Manager ────────────────
    secret_name = (env.google_credentials_secret or "").strip()
    if not secret_name:
        raise RuntimeError(
            "Google TTS: GOOGLE_CREDENTIALS_SECRET is not set and "
            "GOOGLE_CREDENTIALS_USE_LOCAL is not 'true'. "
            "Set GOOGLE_CREDENTIALS_SECRET to the Secret Manager resource name "
            "(e.g. 'projects/948143540104/secrets/key_file_json/versions/latest') "
            "in your .env or environment."
        )

    # Try with default credentials (ADC / Workload Identity) first.
    adc_err: Exception | None = None
    try:
        return _fetch_secret(secret_name)
    except Exception as exc:  # noqa: BLE001
        adc_err = exc  # Save — Python deletes `exc` after the except block.

    # ADC unavailable — try bootstrapping the SM client with the local SA key.
    if local_key_path.exists() and local_key_path.stat().st_size > 2:
        print(
            "[TTS] No Application Default Credentials — bootstrapping "
            f"Secret Manager client with local SA key ({local_key_path})"
        )
        try:
            return _fetch_secret(secret_name, credentials_file=str(local_key_path))
        except Exception as bootstrap_err:  # noqa: BLE001
            raise RuntimeError(
                f"Google TTS: failed to fetch credentials from Secret Manager "
                f"({secret_name}) even with local SA key bootstrap.\n"
                f"  ADC error: {adc_err}\n"
                f"  Bootstrap error: {bootstrap_err}\n"
                f"Verify the SA has roles/secretmanager.secretAccessor on the secret."
            ) from bootstrap_err

    # No ADC and no usable local key — fail with guidance.
    raise RuntimeError(
        f"Google TTS: cannot authenticate to Secret Manager ({secret_name}).\n"
        f"  ADC error: {adc_err}\n"
        f"Fix (pick one):\n"
        f"  • Production: attach Workload Identity or set GOOGLE_APPLICATION_CREDENTIALS.\n"
        f"  • Dev: run 'gcloud auth application-default login' with an account that has "
        f"roles/secretmanager.secretAccessor on the secret.\n"
        f"  • Dev: place a SA key at {_LOCAL_SA_KEY_REL} to bootstrap the SM client."
    )


def _load_local_key(path: Path) -> dict[str, Any]:
    """Load a service-account JSON from a local file. Raises on any problem."""
    if not path.exists():
        raise RuntimeError(
            f"Google TTS: GOOGLE_CREDENTIALS_USE_LOCAL=true but local key not found "
            f"at {path}. Place a valid service-account JSON there or switch to "
            f"Secret Manager (set GOOGLE_CREDENTIALS_USE_LOCAL=false)."
        )
    if path.stat().st_size < 3:
        raise RuntimeError(
            f"Google TTS: local key file is empty or invalid ({path}). "
            f"Restore a valid service-account JSON or switch to Secret Manager."
        )
    with path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    if "private_key" not in data:
        raise RuntimeError(
            f"Google TTS: local key file at {path} does not look like a "
            f"service-account JSON (missing 'private_key'). "
            f"Restore the correct file or switch to Secret Manager."
        )
    print(f"[TTS] Loaded Google credentials from local file: {path}")
    return data


def _fetch_secret(secret_name: str, credentials_file: str | None = None) -> dict[str, Any]:
    """Fetch and parse a Secret Manager secret version.

    When *credentials_file* is provided, it is used to authenticate the
    Secret Manager client (dev machines without ADC).  The returned
    credentials are always from Secret Manager, not the local file.
    """
    from google.cloud import secretmanager  # type: ignore[import-untyped]

    if credentials_file:
        from google.oauth2 import service_account  # type: ignore[import-untyped]

        creds_obj = service_account.Credentials.from_service_account_file(
            credentials_file,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        client = secretmanager.SecretManagerServiceClient(credentials=creds_obj)
    else:
        client = secretmanager.SecretManagerServiceClient()

    response = client.access_secret_version(request={"name": secret_name})
    payload = response.payload.data.decode("utf-8")
    creds: dict[str, Any] = json.loads(payload)
    print(f"[TTS] ✓ Loaded Google TTS credentials from Secret Manager: {secret_name}")
    return creds


Provider = Literal["elevenlabs", "google", "open_ai", "fish_audio"]


@dataclass
class ResolvedTtsConfig:
    provider: Provider
    voice_id: str
    model: str | None = None
    language: str | None = None


@dataclass
class PersonaLanguageSettings:
    language_preference: list[str]
    default_language: str | None
    primary_language: str | None
    stt_language: str
    stt_detect_language: bool


def _map_stt_language(code: str) -> str:
    s = get_settings().runtime.stt
    normalized = code.lower().strip()
    return s.language_defaults.get(normalized, normalized)


def language_settings_from_persona(persona: PersonaData) -> PersonaLanguageSettings:
    lp = [str(x).strip() for x in (persona.language_preference or []) if str(x).strip()]
    default_language = persona.default_language.strip() if persona.default_language else None
    primary = lp[0] if lp else default_language
    multi = len(lp) > 1
    return PersonaLanguageSettings(
        language_preference=lp,
        default_language=default_language,
        primary_language=primary,
        # Deepgram's 'multi' hallucinates heavily without detect_language.
        # Force the STT to use the primary language to stabilize recognition.
        stt_language=_map_stt_language(primary or "en"),
        stt_detect_language=False,
    )


def pick_elevenlabs_model(_language_preference: list[str]) -> str:
    # Multilingual model — quality over lowest latency (see YAML comment).
    return get_settings().runtime.tts.elevenlabs.model


def unsupported_elevenlabs_languages(language_preference: list[str]) -> list[str]:
    supported = set(get_settings().runtime.tts.elevenlabs.supported_languages)
    return [
        lang for lang in language_preference
        if lang.lower().split("-")[0] not in supported
    ]


async def resolve_tts_config(
    persona: PersonaData,
    lang: PersonaLanguageSettings,
    conv: ConvLog,
) -> ResolvedTtsConfig:
    settings = get_settings()
    provider = (persona.tts_provider or "elevenlabs").strip().lower()
    tts_voice_id = (persona.tts_voice_id or "").strip()
    tts_voice = (persona.tts_voice or persona.voice or "").strip()

    if is_bharatgpt(provider) or is_bharatgpt(tts_voice) or is_bharatgpt(tts_voice_id):
        conv.line("INFO", "TTS", "BharatGPT requested — routing to Google TTS")

        # Known BharatGPT ElevenLabs voice IDs → gender.
        _BHARATGPT_VOICE_GENDER: dict[str, str] = {
            "zELrJnEGQSGWhiTNcUKq": "female",
            "siw1N9V8LmYeEWKyWBxv": "male",
        }

        voice_key = tts_voice_id or tts_voice or ""
        gender_key = _BHARATGPT_VOICE_GENDER.get(voice_key, "female")

        # Always use Indian-accent Gemini voices for BharatGPT.
        gemini_voice = _map_locale_gender_to_gemini("hi-in", gender_key)
        conv.line("INFO", "TTS", f"BharatGPT → Google TTS voice: {gemini_voice} (gender={gender_key})")

        return ResolvedTtsConfig(
            provider="google",
            voice_id=gemini_voice,
            model=settings.runtime.tts.google.model,
            language="hi",  # ensures greeting voice picker selects Indian ElevenLabs voices
        )

    if provider == "google":
        raw_voice = tts_voice_id or tts_voice or settings.runtime.tts.google.default_voice
        resolved = resolve_gemini_tts_voice(raw_voice, conv=conv)
        return ResolvedTtsConfig(
            provider="google",
            voice_id=resolved.voice,
            model=settings.runtime.tts.google.model,
        )

    if provider in {"open_ai", "openai"}:
        if tts_voice_id:
            return ResolvedTtsConfig(provider="open_ai", voice_id=tts_voice_id)
        if tts_voice and not is_bharatgpt(tts_voice):
            resolved = await fetch_voice_details(tts_voice, conv)
            if resolved and resolved.provider == "open_ai":
                return ResolvedTtsConfig(provider="open_ai", voice_id=resolved.voice_id)
        conv.line("WARN", "TTS", "OpenAI voice not resolved, falling back to ElevenLabs")
        provider = "elevenlabs"

    # ── Fish Audio ─────────────────────────────────────────────────────────
    # ttsProvider = "fish_audio", ttsVoiceId = Fish Audio model _id
    # Set via POST /voice-cloning/:id/assign-persona on the backend.
    if provider == "fish_audio":
        if not tts_voice_id:
            conv.line(
                "WARN", "TTS",
                "fish_audio provider set but ttsVoiceId is empty — "
                "falling back to ElevenLabs. Assign a cloned voice first.",
            )
            provider = "elevenlabs"
        else:
            # Pass the persona's primary language so the Fish Audio tokenizer
            # picks the correct phoneme set (crucial for Indian names).
            fish_language = lang.primary_language or "en"
            conv.line(
                "INFO", "TTS",
                f"Fish Audio TTS: voice_id={tts_voice_id} language={fish_language}",
            )
            return ResolvedTtsConfig(
                provider="fish_audio",
                voice_id=tts_voice_id,
                language=fish_language,
            )

    # ElevenLabs branch (default)
    voice_id = settings.runtime.tts.elevenlabs.voices.indian_female
    if tts_voice_id and not is_bharatgpt(tts_voice_id):
        voice_id = tts_voice_id
    elif tts_voice and not is_bharatgpt(tts_voice):
        resolved = await fetch_voice_details(tts_voice, conv)
        if resolved and resolved.provider == "elevenlabs":
            voice_id = resolved.voice_id
        elif resolved and resolved.provider == "open_ai":
            return ResolvedTtsConfig(provider="open_ai", voice_id=resolved.voice_id)

    return ResolvedTtsConfig(
        provider="elevenlabs",
        voice_id=voice_id,
        model=pick_elevenlabs_model(lang.language_preference),
        language=None if len(lang.language_preference) > 1 else (lang.primary_language or "en"),
    )


def build_elevenlabs_tts(cfg: ResolvedTtsConfig) -> elevenlabs.TTS:
    s = get_settings().runtime.tts.elevenlabs
    # ElevenLabs plugin normalizes `language` via `LanguageCode(...)` when the
    # kwarg is provided, so we must OMIT it entirely (not pass None) for
    # multi-language personas — passing None crashes inside the plugin.
    kwargs: dict[str, Any] = {
        "voice_id": cfg.voice_id,
        "model": cfg.model or s.model,
        "encoding": s.encoding,
        "streaming_latency": s.streaming_latency,
        "voice_settings": elevenlabs.VoiceSettings(
            stability=s.voice_settings.stability,
            similarity_boost=s.voice_settings.similarity_boost,
            style=s.voice_settings.style,
            speed=s.voice_settings.speed,
            use_speaker_boost=s.voice_settings.use_speaker_boost,
        ),
    }
    if cfg.language:
        kwargs["language"] = cfg.language
    return elevenlabs.TTS(**kwargs)


def build_google_tts(cfg: ResolvedTtsConfig) -> google.TTS:
    """
    Build a streaming Google Cloud TTS instance using Chirp3 HD voices.

    The voice_name must be in Chirp3 HD format: "<locale>-Chirp3-HD-<Voice>"
    e.g. "en-IN-Chirp3-HD-Kore". The plugin sees "chirp" in the name, auto-
    selects chirp_3 model, and enables gRPC streaming — fixing the 10s latency.

    Credentials are always loaded from Secret Manager via
    _load_google_credentials(). Raises at build time with a clear message
    if credentials cannot be obtained.
    """
    s = get_settings().runtime.tts.google
    creds = _load_google_credentials()

    # Extract locale from Chirp3 HD voice name for the language param.
    # e.g. "en-IN-Chirp3-HD-Kore" → "en-IN"
    voice_name = cfg.voice_id
    language: str | None = None
    m = re.match(r"^([a-z]{2,3}-[A-Z]{2})-Chirp3-HD-", voice_name, re.IGNORECASE)
    if m:
        language = f"{m.group(1).split('-')[0].lower()}-{m.group(1).split('-')[1].upper()}"

    kwargs: dict[str, Any] = {
        "voice_name": voice_name,
        "use_streaming": s.use_streaming,
        "location": s.location,
        "credentials_info": creds,
        # Speed knob — 1.0 = default; 1.1–1.2 is a natural-sounding speed-up
        # for Chirp3 HD on telephony (Google Cloud TTS accepts 0.25–4.0).
        "speaking_rate": s.speaking_rate,
    }
    # Chirp 3 unary synthesize_speech rejects PCM (plugin default). Streaming
    # uses PCM; non-streaming must be LINEAR16 / MP3 / OGG_OPUS / MULAW / ALAW.
    if not s.use_streaming:
        kwargs["audio_encoding"] = gtts.AudioEncoding.LINEAR16
    if language:
        kwargs["language"] = language
    return google.TTS(**kwargs)


def build_openai_tts(cfg: ResolvedTtsConfig) -> openai.TTS:
    return openai.TTS(voice=cfg.voice_id)


def build_fish_audio_tts(cfg: ResolvedTtsConfig) -> FishAudioTTS:
    """
    Build a Fish Audio TTS instance from a resolved config.

    Requires FISH_AUDIO_API_KEY in the environment.
    voice_id must be a valid Fish Audio model _id (fishVoiceId from ClonedVoice).

    Env overrides (all optional):
      FISH_AUDIO_MODEL      — model tier. Default "s2.1-pro" (production).
                              Use "s2.1-pro-free" in dev to avoid charges.
                              MUST match backend constants.FISH_AUDIO_TTS_MODEL
                              or preview and live call will sound different.
      FISH_AUDIO_NORMALIZE  — "true" to turn Fish Audio's English-phonetic
                              text normalizer back on. OFF by default because
                              it Americanizes Indian names.
      FISH_AUDIO_LATENCY    — "normal" | "balanced" | "low"
    """
    model = os.environ.get("FISH_AUDIO_MODEL", "s2.1-pro")
    normalize = os.environ.get("FISH_AUDIO_NORMALIZE", "").lower() == "true"
    latency = os.environ.get("FISH_AUDIO_LATENCY", "normal")
    return FishAudioTTS(
        voice_id=cfg.voice_id,
        model=model,
        latency=latency,
        speed=1.0,
        temperature=0.7,
        language=cfg.language,
        normalize=normalize,
    )


def build_tts_from_config(
    cfg: ResolvedTtsConfig,
) -> elevenlabs.TTS | google.TTS | openai.TTS | FishAudioTTS:
    if cfg.provider == "google":
        return build_google_tts(cfg)
    if cfg.provider == "open_ai":
        return build_openai_tts(cfg)
    if cfg.provider == "fish_audio":
        return build_fish_audio_tts(cfg)
    return build_elevenlabs_tts(cfg)

