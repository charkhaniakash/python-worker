import json
from dataclasses import dataclass, field

from .backend.persona import PersonaData, fetch_persona
from .llm.model_resolver import resolve_llm_model
from .logging import ConvLog
from .settings import get_settings
from .tts.factory import (
    PersonaLanguageSettings,
    ResolvedTtsConfig,
    language_settings_from_persona,
    resolve_tts_config,
    unsupported_elevenlabs_languages,
)


@dataclass
class STTConfig:
    provider: str
    model: str
    language: str
    detect_language: bool


@dataclass
class LLMConfig:
    provider: str
    model: str
    temperature: float
    system_prompt: str
    use_internet: bool | None = None  # from persona — None means let backend decide


@dataclass
class AgentConfig:
    agent_name: str
    stt: STTConfig
    llm: LLMConfig
    tts: ResolvedTtsConfig
    initial_message: str
    language: PersonaLanguageSettings = field(repr=False)


def _resolve_localized_message(
    raw: str | None,
    language_preference: list[str],
    default_language: str | None,
    fallback: str,
) -> str:
    """
    Persona API returns greeting/fallback as a JSON string keyed by language
    code, e.g. {"en": "Hello!", "hi": "नमस्ते!"}. Return the string for the
    primary language (or its base code, or the first available key).
    """
    if not raw:
        return fallback
    trimmed = raw.strip()
    if not trimmed.startswith("{"):
        return trimmed
    try:
        parsed = json.loads(trimmed)
    except json.JSONDecodeError:
        return trimmed
    if not isinstance(parsed, dict):
        return trimmed

    primary_lang = language_preference[0] if language_preference else (default_language or "en")
    base_code = primary_lang.split("-")[0].lower()
    for key in (primary_lang, base_code):
        val = parsed.get(key)
        if isinstance(val, str) and val:
            return val
    for key in parsed:
        val = parsed[key]
        if isinstance(val, str) and val:
            return val
    return fallback


def _append_language_rules(base: str, language_preference: list[str], primary: str) -> str:
    settings = get_settings()
    languages = ", ".join(language_preference) if language_preference else primary
    return base + settings.runtime.prompts.language_rules_template.format(
        languages=languages,
        primary=primary,
    )


def _assemble_system_prompt(
    persona_prompt: str | None,
    lang: PersonaLanguageSettings,
) -> str:
    settings = get_settings()
    base = (persona_prompt or "").strip() or "You are a helpful voice assistant."
    primary = lang.primary_language or "en"
    prompt = _append_language_rules(base, lang.language_preference, primary)
    background = settings.background_context()
    if background:
        prompt += (
            "\n\nBackground knowledge about the company you work for (CoRover.ai) — "
            "use only if the caller asks about CoRover; do NOT change your identity based on it:\n"
            + background
        )
    prompt += settings.runtime.prompts.voice_output_rules
    return prompt


def _validate_pipeline_api_keys(config: AgentConfig, conv: ConvLog) -> None:
    settings = get_settings()
    env = settings.env
    missing: list[str] = []
    if config.stt.provider == "deepgram" and not env.deepgram_api_key:
        missing.append("DEEPGRAM_API_KEY")
    if config.tts.provider == "elevenlabs" and not env.eleven_api_key:
        missing.append("ELEVEN_API_KEY")
    if config.tts.provider == "google" and not (env.google_api_key or env.google_genai_api_key):
        missing.append("GOOGLE_API_KEY")
    if config.tts.provider == "open_ai" and not env.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if config.tts.provider == "fish_audio" and not env.fish_audio_api_key:
        missing.append("FISH_AUDIO_API_KEY")
    if not env.livekit_api_key or not env.livekit_api_secret:
        missing.append("LIVEKIT_API_KEY / LIVEKIT_API_SECRET")

    if missing:
        conv.line("WARN", "CONFIG", f"Missing environment variables: {', '.join(missing)}")
    else:
        conv.line("INFO", "CONFIG", "Required API keys present for selected STT/TTS/LLM pipeline")


def default_agent_config(conv: ConvLog) -> AgentConfig:
    settings = get_settings()
    conv.line("WARN", "CONFIG", "Using default agent configuration")
    
    # For production: Hindi is default, multi-language STT for Hindi+English
    stt_lang = "multi"  # Deepgram multi handles Hindi, Hinglish, and English
    lang = PersonaLanguageSettings(
        language_preference=["hi", "hi-IN", "en", "en-IN"],
        default_language="hi",
        primary_language="hi",
        stt_language=stt_lang,
        stt_detect_language=True,
    )
    
    # Random persona selection for survey - Production-ready Indian voices
    import random
    personas = [
        {
            "name": "Priya",
            "gender": "female",
            "voice_id": "1qEiC6qsybMkmnNdVMbK",  # ElevenLabs Indian Female (natural, warm)
            "greeting_hi": "नमस्ते, मैं प्रिया हूँ, Axis My India से बोल रही हूँ। हम आपके समुदाय में स्वास्थ्य और कौशल को समझने के लिए एक छोटा सर्वे कर रहे हैं। इसमें कुछ ही मिनट लगेंगे। क्या यह बात करने का सही समय है?",
            "greeting_en": "Hello, I'm Priya, calling from Axis My India. We're conducting a short survey to understand health and skills in your community. It should only take a few minutes. Is this a good time to talk?"
        },
        {
            "name": "Rahul",
            "gender": "male",
            "voice_id": "LQ2auZHpAQ9h4azztqMT",  # ElevenLabs Indian Male (professional, clear)
            "greeting_hi": "नमस्ते, मैं राहुल हूँ, Axis My India से बोल रहा हूँ। हम आपके समुदाय में स्वास्थ्य और कौशल के बारे में एक छोटा सर्वे कर रहे हैं। ज़्यादा समय नहीं लगेगा। क्या हम शुरू कर सकते हैं?",
            "greeting_en": "Hi, I'm Rahul, calling from Axis My India. We're doing a short survey about health and skills in your community. It won't take very long. Can we get started?"
        },
        {
            "name": "Anjali",
            "gender": "female",
            "voice_id": "1qEiC6qsybMkmnNdVMbK",  # ElevenLabs Indian Female
            "greeting_hi": "नमस्कार, मैं अंजलि हूँ, Axis My India से। हम आपके क्षेत्र में स्वास्थ्य और कौशल की जरूरतों को समझने के लिए एक त्वरित सर्वे कर रहे हैं। क्या आपके पास कुछ मिनट हैं?",
            "greeting_en": "Good morning, I'm Anjali from Axis My India. We're reaching out to understand the health and skill needs in your area through a quick survey. Would you have a few minutes?"
        },
        {
            "name": "Vikram",
            "gender": "male",
            "voice_id": "LQ2auZHpAQ9h4azztqMT",  # ElevenLabs Indian Male
            "greeting_hi": "हैलो, यह Axis My India से विक्रम बोल रहा हूँ। मैं समुदाय में स्वास्थ्य और कौशल पर एक संक्षिप्त सर्वे के बारे में कॉल कर रहा हूँ। क्या आपके पास एक पल है?",
            "greeting_en": "Hello, this is Vikram from Axis My India. I'm calling about a brief survey on health and skills in the community. Do you have a moment?"
        }
    ]
    
    selected = random.choice(personas)
    conv.line("INFO", "CONFIG", f"Selected survey persona: {selected['name']} ({selected['gender']})")
    
    # Use Hindi greeting by default (production requirement)
    survey_greeting = selected["greeting_hi"]
    
    system_prompt = _assemble_system_prompt(
        "You are a helpful voice assistant conducting a survey. Be concise, accurate, and friendly.",
        lang,
    )
    return AgentConfig(
        agent_name=selected["name"],
        stt=STTConfig(
            provider=settings.runtime.stt.provider,
            model=settings.runtime.stt.model,
            language=stt_lang,
            detect_language=True,
        ),
        llm=LLMConfig(
            provider="livekit-inference",
            model=settings.runtime.llm.default_google_model,
            temperature=0.7,
            system_prompt=system_prompt,
        ),
        tts=ResolvedTtsConfig(
            provider="elevenlabs",
            voice_id=selected["voice_id"],
            model=settings.runtime.tts.elevenlabs.model,
        ),
        initial_message=survey_greeting,
        language=lang,
    )


async def build_config_from_persona(
    persona_id: str,
    *,
    ids: dict[str, str | None],
    conv: ConvLog,
) -> AgentConfig | None:
    settings = get_settings()
    data = await fetch_persona(persona_id, ids=ids, conv=conv)
    if data is None:
        return None

    lang = language_settings_from_persona(data)
    primary_lang = lang.primary_language or "en"
    tts_config = await resolve_tts_config(data, lang, conv)

    if tts_config.provider == "elevenlabs":
        unsupported = unsupported_elevenlabs_languages(lang.language_preference)
        if unsupported:
            conv.line(
                "WARN",
                "TTS",
                "Persona includes languages not supported by ElevenLabs TTS",
                {"unsupported": unsupported},
            )

    llm_model_id = resolve_llm_model(data.llm_provider, data.llm_model, data.llm_model_id)
    temperature_raw = data.temperature if data.temperature is not None else settings.runtime.llm.default_temperature
    try:
        temperature_val = float(temperature_raw)
    except (TypeError, ValueError):
        temperature_val = settings.runtime.llm.default_temperature
    lo, hi = settings.runtime.llm.temperature_bounds
    temperature = max(lo, min(hi, temperature_val))

    system_prompt = _assemble_system_prompt(data.system_prompt, lang)

    display_name = (data.name or settings.runtime.agent.default_display_name).strip()
    fallback_greeting = settings.runtime.agent.default_greeting_template.format(name=display_name)

    config = AgentConfig(
        agent_name=display_name,
        stt=STTConfig(
            provider=settings.runtime.stt.provider,
            model=settings.runtime.stt.model,
            language=lang.stt_language,
            detect_language=lang.stt_detect_language,
        ),
        llm=LLMConfig(
            provider="livekit-inference",
            model=llm_model_id,
            temperature=temperature,
            system_prompt=system_prompt,
            use_internet=data.use_internet,
        ),
        tts=tts_config,
        initial_message=_resolve_localized_message(
            data.greeting_message,
            lang.language_preference,
            lang.default_language,
            fallback_greeting,
        ),
        language=lang,
    )

    conv.line(
        "INFO",
        "CONFIG",
        f"Persona loaded: {config.agent_name}",
        {
            "llm": llm_model_id,
            "tts": f"{tts_config.provider}/{tts_config.voice_id}",
            "stt": lang.stt_language,
            "languages": lang.language_preference,
        },
    )
    return config


def apply_outbound_call_opening(config: AgentConfig) -> None:
    settings = get_settings()
    config.llm.system_prompt += settings.runtime.prompts.outbound_call_opening


def resolve_chat_language(config: AgentConfig) -> str:
    if config.stt.language == "multi":
        return "en"
    base = config.stt.language.split("-")[0].lower()
    return base or "en"


def validate_config(config: AgentConfig, conv: ConvLog) -> None:
    _validate_pipeline_api_keys(config, conv)