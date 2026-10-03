"""
Persona model string → LiveKit Inference model id (`provider/model`).

BharatGPT is served via Google Gemini through LiveKit Inference. Broken /
unsupported ids the backend still advertises are remapped via YAML config.
"""

from __future__ import annotations

from ..settings import get_settings


def is_bharatgpt(value: str | None) -> bool:
    if not value:
        return False
    normalized = value.lower().replace(" ", "").replace("_", "").replace("-", "")
    return "bharatgpt" in normalized


def _normalize_inference_id(model_id: str, default_provider: str) -> str:
    trimmed = model_id.strip()
    if not trimmed:
        return ""
    return trimmed if "/" in trimmed else f"{default_provider}/{trimmed}"


def resolve_llm_model(
    llm_provider: str | None,
    llm_model: str | None,
    llm_model_id: str | None,
) -> str:
    s = get_settings().runtime.llm
    provider = (llm_provider or "").strip().lower()
    model = (llm_model or "").strip()
    model_id = (llm_model_id or "").strip()

    # BharatGPT → Gemini (via Inference)
    if is_bharatgpt(provider) or is_bharatgpt(model) or is_bharatgpt(model_id):
        print("[LLM] BharatGPT requested — routing to Google Gemini via LiveKit Inference")
        if model_id and not is_bharatgpt(model_id):
            return _normalize_inference_id(model_id, "google")
        return s.default_google_model

    if model_id:
        if "/" in model_id:
            normalized = model_id.replace("gemini/", "google/", 1) if model_id.startswith("gemini/") else model_id
            remap = s.broken_model_remap.get(normalized)
            if remap:
                print(f"[LLM] Remapping unsupported model '{normalized}' → {remap}")
                return remap
            return normalized
        return _normalize_inference_id(model_id, provider or "google")

    model_lower = model.lower()
    if provider == "openai":
        return (
            _normalize_inference_id(model, "openai")
            if "gpt" in model_lower
            else s.default_openai_model
        )
    if provider == "google":
        return (
            _normalize_inference_id(model, "google")
            if "gemini" in model_lower
            else s.default_google_model
        )
    if provider == "anthropic":
        if "claude" in model_lower:
            return _normalize_inference_id(model, "anthropic")
        print(f"[LLM] Unknown Anthropic model '{model}', falling back to Google Gemini")
        return s.default_google_model

    if "gemini" in model_lower:
        return _normalize_inference_id(model, "google")
    if "gpt" in model_lower:
        return _normalize_inference_id(model, "openai")
    if "claude" in model_lower:
        return _normalize_inference_id(model, "anthropic")
    return s.default_google_model


def llm_provider_key(model_id: str) -> str:
    """Map a resolved model id to a cost-tracking provider key."""
    keys = get_settings().runtime.cost_tracking.llm_provider_key_by_model_substring
    model_lower = model_id.lower()
    for substring, key in keys.items():
        if substring == "default":
            continue
        if substring in model_lower:
            return key
    return keys.get("default", "gemini_flash_lite")
