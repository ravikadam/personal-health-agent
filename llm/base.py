"""Provider-agnostic LLM interface.

The rest of the app depends only on `LLMProvider` and `LLMConfig` — never on a
specific vendor SDK. Concrete adapters (OpenAI, Anthropic, Gemini) live in
`providers.py` and are selected at runtime via `factory.get_provider`.

Design goals:
  * Swap providers without touching call sites.
  * Degrade gracefully: if no key/SDK is present, a NullProvider keeps the app
    fully functional in deterministic, rule-based mode.
  * One small surface: `complete()` for chat, `extract_json()` for structured
    extraction.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LLMConfig:
    provider: str = "none"            # none | openai | anthropic | gemini
    model: str = ""                   # provider-specific; factory fills default
    api_key: str = ""                 # never logged / persisted to disk
    temperature: float = 0.2
    max_tokens: int = 1024
    extra: Dict = field(default_factory=dict)


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    used_llm: bool = True             # False when a rule-based fallback answered


class LLMProvider(ABC):
    """Common contract every vendor adapter implements."""

    name: str = "base"

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def available(self) -> bool:
        """True if this provider can actually make a call (SDK + key present)."""

    @abstractmethod
    def complete(self, system: str, user: str,
                 history: Optional[List[Dict]] = None) -> LLMResult:
        """Single-turn (optionally with history) chat completion → text."""

    def extract_json(self, system: str, user: str) -> Optional[list | dict]:
        """Call `complete` and best-effort parse a JSON payload from the reply.

        Returns None on failure so callers can fall back to rules.
        """
        try:
            res = self.complete(system, user)
            return _first_json(res.text)
        except Exception:
            return None


def _first_json(text: str) -> Optional[list | dict]:
    """Extract the first JSON array/object from a model reply."""
    if not text:
        return None
    # Strip code fences if present
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    # Find the outermost [...] or {...}
    for opener, closer in (("[", "]"), ("{", "}")):
        start = candidate.find(opener)
        end = candidate.rfind(closer)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(candidate[start:end + 1])
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(candidate.strip())
    except json.JSONDecodeError:
        return None
