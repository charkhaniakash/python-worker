"""
Typed access to runtime configuration.

`config/defaults.yaml` holds every operational default (voice IDs, model names,
timeouts, retries, VAD, endpointing…). Environment variables override any of
them via the mapping declared on `EnvOverrides` — new env knobs are added there
and consumed in code via `get_settings()`, never read directly from `os.environ`
outside this module.

`get_settings()` is memoized so repeated access is free.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# YAML config schema (mirrors config/defaults.yaml)
# ---------------------------------------------------------------------------


class AgentConfig(BaseModel):
    name: str
    default_display_name: str
    default_greeting_template: str


class PersonaFetchConfig(BaseModel):
    attempts: int
    per_attempt_timeout_ms: int
    backoff_base_ms: int


class TimeoutOnly(BaseModel):
    timeout_ms: int


class BackendConfig(BaseModel):
    base_url: str
    default_timeout_ms: int
    persona_fetch: PersonaFetchConfig
    usage_report: TimeoutOnly
    call_started: TimeoutOnly


class ChatConfig(BaseModel):
    url: str | None
    path: str
    enabled: bool
    attempt_timeout_seconds: int
    default_language: str
    default_timezone: str
    # Studio web sends llmResponse=true. False on telephony forced the slow
    # Knowledge Mesh / GUARDRAIL / REJECTION_ENGINE path (~6s).
    llm_response: bool = True
    # STT already produced text. inputType=voice selected a heavier backend
    # pipeline even though the body is a string.
    input_type: str = "text"
    channel: str = "telephony"


class STTConfig(BaseModel):
    provider: str
    model: str
    language_defaults: dict[str, str]
    sample_rate_hz: dict[str, int]
    num_channels: int
    detect_language: bool
    endpointing_ms: int | None = None
    utterance_end_ms: int | None = None
    interim_results: bool | None = None
    smart_format: bool | None = None
    punctuate: bool | None = None
    filler_words: bool | None = None
    no_delay: bool | None = None
    numerals: bool | None = None
    keyterms: list[str] = Field(default_factory=list)


class LLMConfig(BaseModel):
    default_google_model: str
    default_openai_model: str
    default_temperature: float
    temperature_bounds: tuple[float, float]
    broken_model_remap: dict[str, str]


class ElevenLabsVoiceSettings(BaseModel):
    stability: float
    similarity_boost: float
    style: float
    use_speaker_boost: bool
    # Speaking-rate multiplier. ElevenLabs safe range ~0.7–1.2; 1.0 = default.
    # Values >1.15 start to sound unnatural on Indian-English voices.
    speed: float = 1.0


class ElevenLabsVoices(BaseModel):
    indian_female: str
    indian_male: str
    us_female: str
    us_male: str


class ElevenLabsTTSConfig(BaseModel):
    model: str
    encoding: str
    streaming_latency: int
    voice_settings: ElevenLabsVoiceSettings
    supported_languages: list[str]
    voices: ElevenLabsVoices


class GoogleTTSConfig(BaseModel):
    model: str
    default_voice: str
    use_vertexai: bool
    use_streaming: bool = True
    location: str = "global"
    # Speaking-rate multiplier. Google Cloud TTS range 0.25–4.0; 1.0 = default.
    # 1.1–1.2 sounds noticeably snappier on Chirp3 HD voices without artifacts.
    speaking_rate: float = 1.0


class OpenAITTSConfig(BaseModel):
    model: str | None


class TTSConfig(BaseModel):
    elevenlabs: ElevenLabsTTSConfig
    google: GoogleTTSConfig
    openai: OpenAITTSConfig


class SessionConfig(BaseModel):
    allow_interruptions: bool
    interruption_mode: Literal["adaptive", "vad"]
    discard_audio_if_uninterruptible: bool
    min_interruption_duration_ms: int
    min_interruption_words: int
    min_endpointing_delay_ms: int
    max_endpointing_delay_ms: int
    telephony_output_sample_rate_hz: int
    telephony_output_num_channels: int
    start_timeout_ms: int
    room_io_ready_timeout_ms: int
    sip_answer_timeout_ms: int
    sip_answer_poll_interval_ms: int
    # Idle-caller detection (caller picks up but never speaks).
    idle_prompt_first_ms: int = 18000
    idle_prompt_interval_ms: int = 18000
    idle_hangup_ms: int = 60000
    idle_prompts: list[str] = []
    idle_goodbye: str = "I haven't heard from you, so I'll end the call. Goodbye."
    user_goodbye_farewell: str = "Thank you for calling. Goodbye!"


class VADConfig(BaseModel):
    min_speech_duration: float
    min_silence_duration: float
    activation_threshold: float
    deactivation_threshold: float | None = None
    prefix_padding_duration: float | None = None
    sample_rate: int | None = None


class NoiseCancellationConfig(BaseModel):
    telephony_enabled: bool
    # nc = keep all speech, strip fans/traffic (robust for earphones / quiet mics).
    # bvc_telephony = isolate "primary" speaker — can mute headset mics.
    # off = no Krisp filter.
    mode: Literal["nc", "bvc_telephony", "off"] = "nc"

    @field_validator("mode", mode="before")
    @classmethod
    def parse_mode(cls, v: Any) -> Any:
        if v is False:
            return "off"
        if v is True:
            return "nc"
        return v



class AmbienceConfig(BaseModel):
    enabled: bool
    sound: str
    volume: float


class TeardownConfig(BaseModel):
    exit_delay_ms: int


class LoggingConfig(BaseModel):
    verbose: bool
    truncate_chars: int


class PromptsConfig(BaseModel):
    background_context_file: str
    voice_output_rules: str
    language_rules_template: str
    outbound_call_opening: str


class CostTrackingConfig(BaseModel):
    stt_provider_key: str
    llm_provider_key_by_model_substring: dict[str, str]
    tts_provider_key_by_label_prefix: dict[str, str]


class GeminiLocaleVoice(BaseModel):
    female: str
    male: str
    default: str


class GeminiVoicesConfig(BaseModel):
    all: list[str]
    masculine: list[str]
    locale_map: dict[str, GeminiLocaleVoice]


class RuntimeConfig(BaseModel):
    """Full schema of `config/defaults.yaml`."""

    agent: AgentConfig
    backend: BackendConfig
    chat: ChatConfig
    stt: STTConfig
    llm: LLMConfig
    tts: TTSConfig
    session: SessionConfig
    vad: VADConfig
    noise_cancellation: NoiseCancellationConfig
    ambience: AmbienceConfig
    teardown: TeardownConfig
    logging: LoggingConfig
    prompts: PromptsConfig
    cost_tracking: CostTrackingConfig
    gemini_voices: GeminiVoicesConfig
    indian_language_codes: list[str]

    @field_validator("indian_language_codes", mode="after")
    @classmethod
    def _normalize_indian_codes(cls, v: list[str]) -> list[str]:
        return [c.lower() for c in v]


# ---------------------------------------------------------------------------
# Environment overrides
# ---------------------------------------------------------------------------


class EnvOverrides(BaseSettings):
    """
    Environment-only knobs. All of these are optional — when unset the value
    from `config/defaults.yaml` is used, or (for credentials) the plugins pick
    up the standard provider env vars themselves.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # LiveKit connection (credentials — env only, never in YAML)
    livekit_url: str | None = None
    livekit_api_key: str | None = None
    livekit_api_secret: str | None = None

    # Backend
    backend_url: str | None = None

    # Chat SSE
    chat_api_url: str | None = None
    use_bharatgpt_chat: str | None = None  # "false" disables
    internal_api_key: str | None = None

    # Agent identity
    agent_name: str | None = None

    # Feature flags
    disable_telephony_nc: str | None = None  # "true" disables
    conv_log_verbose: str | None = None  # "true" enables

    # Provider credentials — read by plugins, exposed here for validation.
    deepgram_api_key: str | None = None
    eleven_api_key: str | None = None
    google_api_key: str | None = None
    google_genai_api_key: str | None = None
    openai_api_key: str | None = None
    fish_audio_api_key: str | None = None   # Fish Audio TTS (cloned voices)

    # Google service-account credentials sourced from Secret Manager.
    # Format: "projects/<project>/secrets/<name>/versions/<version>"
    # e.g.  "projects/948143540104/secrets/key_file_json/versions/latest"
    # When set, the JSON is fetched at startup and passed as credentials_info
    # to google.TTS — no file on disk, no GOOGLE_APPLICATION_CREDENTIALS needed.
    google_credentials_secret: str | None = None

    # Set to "true" to load Google TTS credentials directly from the local file
    # (src/worker/credientials/google.json) instead of Secret Manager.
    # When false/unset (default), GOOGLE_CREDENTIALS_SECRET must be set.
    # The local file is only used to bootstrap the Secret Manager client when
    # ADC is unavailable (dev machines) — never as TTS credentials directly.
    google_credentials_use_local: str | None = None  # "true" = use local file directly

    def use_local_google_credentials(self) -> bool:
        return self._as_bool(self.google_credentials_use_local) is True

    @staticmethod
    def _as_bool(value: str | None) -> bool | None:
        if value is None:
            return None
        return value.strip().lower() == "true"

    def bharatgpt_chat_enabled(self) -> bool | None:
        b = self._as_bool(self.use_bharatgpt_chat)
        # matches TS: only "false" disables — everything else keeps default.
        if self.use_bharatgpt_chat is not None and self.use_bharatgpt_chat.strip().lower() == "false":
            return False
        return b

    def telephony_nc_disabled(self) -> bool:
        return self._as_bool(self.disable_telephony_nc) is True

    def verbose_conv_log(self) -> bool:
        return self._as_bool(self.conv_log_verbose) is True


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "defaults.yaml"


def _resolve_config_path() -> Path:
    override = os.environ.get("WORKER_CONFIG_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_CONFIG_PATH


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Runtime config not found at {path}. Set $WORKER_CONFIG_PATH or restore config/defaults.yaml."
        )
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Runtime config {path} must be a mapping, got {type(data).__name__}")
    return data


class Settings(BaseModel):
    """Combined view: parsed YAML + env overrides."""

    runtime: RuntimeConfig
    env: EnvOverrides
    config_path: Path = Field(exclude=True)
    repo_root: Path = Field(exclude=True)

    # -- Convenience accessors (single source of truth for callers) --

    @property
    def backend_url(self) -> str:
        url = (self.env.backend_url or self.runtime.backend.base_url).strip()
        return url.rstrip("/")

    @property
    def agent_name(self) -> str:
        return (self.env.agent_name or self.runtime.agent.name).strip()

    @property
    def chat_api_url(self) -> str:
        override = (self.env.chat_api_url or "").strip()
        if override:
            return override.rstrip("/")
        if self.runtime.chat.url:
            return self.runtime.chat.url.rstrip("/")
        return f"{self.backend_url}{self.runtime.chat.path}"

    @property
    def verbose_conv_log(self) -> bool:
        return self.env.verbose_conv_log() or self.runtime.logging.verbose

    @property
    def telephony_nc_enabled(self) -> bool:
        if self.env.telephony_nc_disabled():
            return False
        return self.runtime.noise_cancellation.telephony_enabled

    def background_context(self) -> str:
        path = (self.repo_root / self.runtime.prompts.background_context_file).resolve()
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    path = _resolve_config_path()
    data = _load_yaml(path)
    runtime = RuntimeConfig.model_validate(data)
    env = EnvOverrides()  # reads .env via SettingsConfigDict
    repo_root = path.parent.parent  # config/defaults.yaml -> repo root
    return Settings(runtime=runtime, env=env, config_path=path, repo_root=repo_root)


def reload_settings() -> Settings:
    """Force a reload — useful in tests."""
    get_settings.cache_clear()
    return get_settings()
