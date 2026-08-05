"""Concrete LLM adapters: OpenAI, Anthropic (Claude), Gemini, and Null.

Each adapter lazily imports its SDK so the app runs even when only one (or
none) is installed. All expose the same `LLMProvider` contract.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .base import LLMConfig, LLMProvider, LLMResult


def _call_tolerant(fn, kwargs: dict, _depth: int = 0):
    """Call an SDK method, and if it rejects a parameter as unsupported/
    deprecated for the model, drop that parameter and retry.

    Newer models (e.g. Claude Sonnet 5, OpenAI o-series) reject `temperature`
    and sometimes `max_tokens`; older ones require them. This keeps one code
    path working across all of them instead of hardcoding per-model rules.
    """
    try:
        return fn(**kwargs)
    except Exception as exc:  # narrow by message, then re-raise if unrelated
        msg = str(exc).lower()
        if _depth > 2:
            raise
        for param in ("temperature", "max_tokens", "top_p"):
            if param in kwargs and param in msg and (
                    "deprecat" in msg or "unsupported" in msg
                    or "not supported" in msg or "invalid" in msg):
                kwargs = {k: v for k, v in kwargs.items() if k != param}
                return _call_tolerant(fn, kwargs, _depth + 1)
        raise


class NullProvider(LLMProvider):
    """No-LLM fallback. `available()` is False so callers use rule-based paths."""

    name = "none"

    def available(self) -> bool:
        return False

    def complete(self, system: str, user: str,
                 history: Optional[List[Dict]] = None) -> LLMResult:
        return LLMResult(text="", provider="none", model="", used_llm=False)

    def extract_json(self, system: str, user: str):
        return None


class OpenAIProvider(LLMProvider):
    name = "openai"

    def available(self) -> bool:
        if not self.config.api_key:
            return False
        try:
            import openai  # noqa: F401
            return True
        except Exception:
            return False

    def complete(self, system: str, user: str,
                 history: Optional[List[Dict]] = None) -> LLMResult:
        from openai import OpenAI

        client = OpenAI(api_key=self.config.api_key)
        messages = [{"role": "system", "content": system}]
        messages += history or []
        messages.append({"role": "user", "content": user})
        kwargs = dict(model=self.config.model, messages=messages,
                      temperature=self.config.temperature,
                      max_tokens=self.config.max_tokens)
        resp = _call_tolerant(client.chat.completions.create, kwargs)
        return LLMResult(text=resp.choices[0].message.content or "",
                         provider=self.name, model=self.config.model)


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def available(self) -> bool:
        if not self.config.api_key:
            return False
        try:
            import anthropic  # noqa: F401
            return True
        except Exception:
            return False

    def complete(self, system: str, user: str,
                 history: Optional[List[Dict]] = None) -> LLMResult:
        import anthropic

        client = anthropic.Anthropic(api_key=self.config.api_key)
        messages = list(history or [])
        messages.append({"role": "user", "content": user})
        kwargs = dict(model=self.config.model, system=system,
                      messages=messages, temperature=self.config.temperature,
                      max_tokens=self.config.max_tokens)
        resp = _call_tolerant(client.messages.create, kwargs)
        parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
        return LLMResult(text="".join(parts), provider=self.name,
                         model=self.config.model)


class GeminiProvider(LLMProvider):
    name = "gemini"

    def available(self) -> bool:
        if not self.config.api_key:
            return False
        try:
            import google.generativeai  # noqa: F401
            return True
        except Exception:
            return False

    def complete(self, system: str, user: str,
                 history: Optional[List[Dict]] = None) -> LLMResult:
        import google.generativeai as genai

        genai.configure(api_key=self.config.api_key)
        model = genai.GenerativeModel(
            model_name=self.config.model,
            system_instruction=system,
            generation_config={
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_tokens,
            },
        )
        # Flatten optional history into the prompt for a single-turn call.
        convo = ""
        for m in history or []:
            convo += f"{m['role']}: {m['content']}\n"
        convo += user
        resp = model.generate_content(convo)
        return LLMResult(text=getattr(resp, "text", "") or "",
                         provider=self.name, model=self.config.model)


PROVIDER_CLASSES = {
    "none": NullProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
}
