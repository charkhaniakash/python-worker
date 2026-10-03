"""
Fish Audio TTS plugin for LiveKit Agents.

Wraps the Fish Audio REST TTS API (POST /v1/tts) in a LiveKit-compatible
`TTS` class so the voice pipeline can use cloned voices from Fish Audio
for real-time telephony calls.

Fish Audio API docs: https://docs.fish.audio/api-reference/
Model used: s2.1-pro (or s2.1-pro-free for dev)

Design:
  - Implements the non-streaming `TTS.synthesize()` path via `ChunkedStream`.
  - Fish Audio returns a chunked HTTP response of raw audio bytes (MP3/PCM).
  - We collect the full response into a single buffer, initialize the
    AudioEmitter with the correct MIME type, then push the bytes in one shot.
  - This is "pseudo-streaming" — we wait for the full audio before emitting,
    which adds latency proportional to text length. For production workloads
    consider upgrading to the WebSocket streaming API when available.

Usage:
    tts = FishAudioTTS(
        voice_id="<Fish Audio model _id>",
        api_key=os.environ["FISH_AUDIO_API_KEY"],
    )
    # passed directly to AgentSession(tts=tts, ...)
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from livekit.agents import tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents._exceptions import APIConnectionError, APIStatusError

# ─── Constants ───────────────────────────────────────────────────────────────

FISH_AUDIO_BASE_URL = "https://api.fish.audio"
FISH_AUDIO_TTS_ENDPOINT = "/v1/tts"

# Fish Audio TTS output — PCM at 44100 Hz, mono
# We request mp3 for smaller payloads and better phone compatibility.
_OUTPUT_FORMAT   = "mp3"
_SAMPLE_RATE     = 44100
_NUM_CHANNELS    = 1
_MIME_TYPE       = "audio/mpeg"

# Default model — use "s2.1-pro-free" during development to avoid charges.
# Switch to "s2.1-pro" for production quality.
_DEFAULT_MODEL = "s2.1-pro"


# ─── ChunkedStream ───────────────────────────────────────────────────────────

class _FishAudioChunkedStream(tts.ChunkedStream):
    """
    Fetches audio from Fish Audio for a single text segment.

    Fish Audio streams the response as chunked HTTP, but we collect the full
    buffer first (simpler, avoids partial-frame issues with MP3 framing).
    """

    def __init__(
        self,
        *,
        fish_tts: "FishAudioTTS",
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=fish_tts, input_text=input_text, conn_options=conn_options)
        self._fish_tts = fish_tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=f"fish-{id(self)}",
            sample_rate=_SAMPLE_RATE,
            num_channels=_NUM_CHANNELS,
            mime_type=_MIME_TYPE,
            stream=False,
        )

        payload: dict[str, Any] = {
            "text":               self._input_text,
            "reference_id":       self._fish_tts.voice_id,
            "format":             _OUTPUT_FORMAT,
            "latency":            self._fish_tts.latency,
            "temperature":        self._fish_tts.temperature,
            "top_p":              0.7,

            "normalize":          self._fish_tts.normalize,
            "chunk_length":       300,
            "mp3_bitrate":        128,
            "repetition_penalty": 1.2,
            "prosody": {
                "speed":               self._fish_tts.speed,
                "volume":              0,
                "normalize_loudness":  True,
            },
        }

        if self._fish_tts.language:
            payload["language"] = self._fish_tts.language

        headers = {
            "Authorization": f"Bearer {self._fish_tts.api_key}",
            "Content-Type":  "application/json",
            "model":         self._fish_tts.model,
        }

        try:
            async with httpx.AsyncClient(timeout=self._conn_options.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{FISH_AUDIO_BASE_URL}{FISH_AUDIO_TTS_ENDPOINT}",
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        raise APIStatusError(
                            message=f"Fish Audio TTS returned {response.status_code}: "
                                    f"{body.decode('utf-8', errors='replace')}",
                            status_code=response.status_code,
                            request=None,  # type: ignore[arg-type]
                            body=body,
                        )

                    # Collect chunked response and push bytes to the emitter
                    async for chunk in response.aiter_bytes():
                        if chunk:
                            output_emitter.push(chunk)

        except httpx.TimeoutException as exc:
            raise APIConnectionError(f"Fish Audio TTS timed out: {exc}") from exc
        except httpx.RequestError as exc:
            raise APIConnectionError(f"Fish Audio TTS connection error: {exc}") from exc

        output_emitter.flush()


# ─── TTS plugin ──────────────────────────────────────────────────────────────

class FishAudioTTS(tts.TTS):
    """
    LiveKit TTS plugin backed by Fish Audio.

    Parameters
    ----------
    voice_id    Fish Audio model ``_id`` (the ``fishVoiceId`` stored in your DB).
    api_key     Fish Audio API key. Defaults to ``FISH_AUDIO_API_KEY`` env var.
    model       Fish Audio model tier.  ``"s2.1-pro"`` (production) or
                ``"s2.1-pro-free"`` (development, lower quality).
    latency     ``"normal"`` | ``"balanced"`` | ``"low"``.
    speed       Speaking speed multiplier (0.5 – 2.0).
    temperature Expressiveness (0.0 – 1.0).
    """

    def __init__(
        self,
        *,
        voice_id: str,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        latency: str = "normal",
        speed: float = 1.0,
        temperature: float = 0.7,
        language: str | None = None,
        normalize: bool = False,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=_SAMPLE_RATE,
            num_channels=_NUM_CHANNELS,
        )

        resolved_key = api_key or os.environ.get("FISH_AUDIO_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "Fish Audio API key is required. "
                "Set FISH_AUDIO_API_KEY environment variable or pass api_key= to FishAudioTTS."
            )

        self._voice_id   = voice_id
        self._api_key    = resolved_key
        self._model      = model
        self._latency    = latency
        self._speed      = speed
        self._temperature = temperature
        self._language   = language
        self._normalize  = normalize

    # ─── Public properties ────────────────────────────────────────────────

    @property
    def voice_id(self) -> str:
        return self._voice_id

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def model(self) -> str:
        return self._model

    @property
    def latency(self) -> str:
        return self._latency

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def language(self) -> str | None:
        return self._language

    @property
    def normalize(self) -> bool:
        return self._normalize

    @property
    def provider(self) -> str:
        return "fish_audio"

    # ─── LiveKit TTS interface ────────────────────────────────────────────

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> _FishAudioChunkedStream:
        return _FishAudioChunkedStream(
            fish_tts=self,
            input_text=text,
            conn_options=conn_options,
        )
