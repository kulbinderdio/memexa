"""
LLM provider abstraction for memexa-web.

Settings are loaded from the database at call time so that configuration
changes take effect immediately without restarting the server.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any

import httpx

from db import get_settings

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SUMMARISE_PROMPT = """You are a knowledge management assistant. Analyse the following article and return ONLY valid JSON with exactly two keys:
- "summary": a concise 2-4 sentence summary of the key ideas
- "tags": an array of 3-7 lowercase keyword tags (single words or short phrases)

Respond with ONLY the JSON object, no markdown fences, no preamble.

Article:
{text}"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_summarise_response(raw: str) -> tuple[str, list[str]]:
    """Extract summary and tags from an LLM JSON response.

    Falls back gracefully if the model returns malformed JSON.
    """
    match = _JSON_RE.search(raw)
    if match:
        try:
            data = json.loads(match.group())
            summary = str(data.get("summary", "")).strip()
            tags_raw = data.get("tags", [])
            tags: list[str] = [str(t).strip() for t in tags_raw if t]
            if summary:
                return summary, tags
        except json.JSONDecodeError:
            pass

    # Last-resort: return the raw text as the summary
    return raw.strip()[:500], []


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...

    @abstractmethod
    async def chat(self, messages: list[dict[str, str]]) -> str: ...

    @abstractmethod
    async def summarise(self, text: str) -> tuple[str, list[str]]: ...


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str, chat_model: str, embed_model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embed_model = embed_model

    async def embed(self, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.base_url}/api/embed",
                json={"model": self.embed_model, "input": text},
            )
            resp.raise_for_status()
            return resp.json()["embeddings"][0]

    async def chat(self, messages: list[dict[str, str]]) -> str:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.chat_model,
                    "messages": messages,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]

    async def summarise(self, text: str) -> tuple[str, list[str]]:
        truncated = text[:4000]
        prompt = _SUMMARISE_PROMPT.format(text=truncated)
        raw = await self.chat([{"role": "user", "content": prompt}])
        return _parse_summarise_response(raw)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    _BASE = "https://api.openai.com/v1"

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def embed(self, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self._BASE}/embeddings",
                headers=self._auth_headers(),
                json={"model": "text-embedding-3-small", "input": text},
            )
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]

    async def chat(self, messages: list[dict[str, str]]) -> str:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self._BASE}/chat/completions",
                headers=self._auth_headers(),
                json={"model": self.model, "messages": messages},
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    async def summarise(self, text: str) -> tuple[str, list[str]]:
        truncated = text[:4000]
        prompt = _SUMMARISE_PROMPT.format(text=truncated)
        raw = await self.chat([{"role": "user", "content": prompt}])
        return _parse_summarise_response(raw)


# ---------------------------------------------------------------------------
# Claude (Anthropic)
# ---------------------------------------------------------------------------

class ClaudeProvider(LLMProvider):
    _BASE = "https://api.anthropic.com/v1"
    _ANTHROPIC_VERSION = "2023-06-01"

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self._ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    async def embed(self, text: str) -> list[float]:  # type: ignore[override]
        raise NotImplementedError(
            "Claude does not support embeddings — "
            "switch to Ollama or OpenAI for semantic features"
        )

    async def chat(self, messages: list[dict[str, str]]) -> str:
        # Separate system messages from user/assistant turns
        system_parts: list[str] = []
        turns: list[dict[str, str]] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_parts.append(msg["content"])
            else:
                turns.append(msg)

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": turns,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self._BASE}/messages",
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            content = resp.json()["content"]
            # content is a list of blocks; find the first text block
            for block in content:
                if block.get("type") == "text":
                    return block["text"]
            return ""

    async def summarise(self, text: str) -> tuple[str, list[str]]:
        truncated = text[:4000]
        prompt = _SUMMARISE_PROMPT.format(text=truncated)
        raw = await self.chat([{"role": "user", "content": prompt}])
        return _parse_summarise_response(raw)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

async def get_provider() -> LLMProvider:
    """Load current provider from DB settings and return the right instance."""
    settings = await get_settings()
    provider = settings.get("llm_provider", "ollama")

    if provider == "openai":
        return OpenAIProvider(
            api_key=settings.get("openai_api_key", ""),
            model=settings.get("openai_model", "gpt-4o-mini"),
        )
    elif provider == "claude":
        return ClaudeProvider(
            api_key=settings.get("claude_api_key", ""),
            model=settings.get("claude_model", "claude-sonnet-4-6"),
        )
    else:  # default: ollama
        return OllamaProvider(
            base_url=os.environ.get("OLLAMA_BASE_URL") or settings.get("ollama_base_url", "http://localhost:11434"),
            chat_model=settings.get("ollama_chat_model", "gemma3:4b"),
            embed_model=settings.get("ollama_embed_model", "mxbai-embed-large"),
        )
