"""Provider selection + configuration.

Reads provider/model/key from an explicit LLMConfig or from environment
variables, and builds the matching adapter. Publishers can ship the app and let
each end-user pick any provider in the UI, or preset one via env vars.
"""

from __future__ import annotations

import os
from typing import Dict, List

from .base import LLMConfig, LLMProvider
from .providers import PROVIDER_CLASSES

# Sensible current defaults per provider; overridable in the UI / env.
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5",
    "gemini": "gemini-1.5-flash",
    "none": "",
}

# Common model choices surfaced in the UI dropdown.
MODEL_CHOICES = {
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini", "o4-mini"],
    "anthropic": ["claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5-20251001"],
    "gemini": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
    "none": [""],
}

# Env var names checked (in order) for each provider's key.
ENV_KEYS = {
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "none": [],
}


def env_key_for(provider: str) -> str:
    for var in ENV_KEYS.get(provider, []):
        if os.environ.get(var):
            return os.environ[var]
    return ""


def default_config() -> LLMConfig:
    """Pick a provider from whatever env keys are present (else 'none')."""
    for provider in ("anthropic", "openai", "gemini"):
        key = env_key_for(provider)
        if key:
            return LLMConfig(provider=provider,
                             model=DEFAULT_MODELS[provider], api_key=key)
    return LLMConfig(provider="none", model="")


def get_provider(config: LLMConfig) -> LLMProvider:
    cls = PROVIDER_CLASSES.get(config.provider, PROVIDER_CLASSES["none"])
    if config.provider != "none" and not config.model:
        config.model = DEFAULT_MODELS.get(config.provider, "")
    if config.provider != "none" and not config.api_key:
        config.api_key = env_key_for(config.provider)
    return cls(config)


def list_providers() -> List[str]:
    return list(PROVIDER_CLASSES.keys())


def sdk_installed(provider: str) -> bool:
    """Whether the vendor SDK is importable (independent of having a key)."""
    import importlib
    modmap = {
        "openai": "openai",
        "anthropic": "anthropic",
        "gemini": "google.generativeai",
        "none": None,
    }
    mod = modmap.get(provider)
    if not mod:
        return True
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False
