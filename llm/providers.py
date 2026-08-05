"""Concrete LLM adapters: OpenAI, Anthropic (Claude), Gemini, and Null.

Each adapter lazily imports its SDK so the app runs even when only one (or
none) is installed. All expose the same `LLMProvider` contract.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .base import LLMConfig, LLMProvider, LLMResult


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
        resp = client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
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
        resp = client.messages.create(
            model=self.config.model,
            system=system,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
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
