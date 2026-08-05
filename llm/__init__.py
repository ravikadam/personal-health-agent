"""Provider-agnostic LLM layer.

Public surface:
    from llm import LLMConfig, get_provider, default_config, list_providers
"""

from .base import LLMConfig, LLMProvider, LLMResult
from .factory import (DEFAULT_MODELS, MODEL_CHOICES, default_config,
                      env_key_for, get_provider, list_providers, sdk_installed)

__all__ = [
    "LLMConfig", "LLMProvider", "LLMResult",
    "get_provider", "default_config", "list_providers", "sdk_installed",
    "env_key_for", "DEFAULT_MODELS", "MODEL_CHOICES",
]
