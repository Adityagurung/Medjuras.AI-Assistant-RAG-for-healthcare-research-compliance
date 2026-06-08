"""MyGenAssist (OpenAI-compatible v2) client helpers."""
from __future__ import annotations

import json
import os
from typing import Any, List, Optional, Type, TypeVar

import certifi
import httpx
from openai import OpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

DEFAULT_BASE_URL = "https://chat.int.bayer.com/api/v2"
DEFAULT_CHAT_MODEL = "gpt-oss-120b"
DEFAULT_AUX_MODEL = "gpt-5.1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def use_mygenassist() -> bool:
    return bool(os.getenv("MYGENASSIST_API_KEY", "").strip())


def get_mygenassist_base_url() -> str:
    return os.getenv("MYGENASSIST_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def get_chat_model() -> str:
    return os.getenv("MYGENASSIST_CHAT_MODEL", DEFAULT_CHAT_MODEL)


def get_aux_model() -> str:
    return os.getenv("MYGENASSIST_AUX_MODEL", DEFAULT_AUX_MODEL)


def get_embedding_model() -> str:
    return os.getenv("MYGENASSIST_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def get_max_tokens(env_key: str, default: int) -> int:
    raw = os.getenv(env_key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _http_client() -> httpx.Client:
    timeout = float(os.getenv("MYGENASSIST_TIMEOUT", os.getenv("OPENAI_TIMEOUT", "120")))
    return httpx.Client(verify=certifi.where(), timeout=timeout)


def get_llm_client(api_key: Optional[str] = None) -> OpenAI:
    """Return MyGenAssist client when configured, otherwise OpenAI."""
    if use_mygenassist():
        key = api_key or os.getenv("MYGENASSIST_API_KEY")
        return OpenAI(
            api_key=key,
            base_url=get_mygenassist_base_url(),
            http_client=_http_client(),
        )

    key = api_key or os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=key, http_client=_http_client())


def structured_chat_parse(
    client: OpenAI,
    *,
    model: str,
    messages: List[dict[str, Any]],
    response_format: Type[T],
    temperature: float = 0,
) -> T:
    """Try beta structured parse first; fall back to JSON-in-prompt parsing."""
    try:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is not None:
            return parsed
    except Exception:
        pass

    schema_hint = json.dumps(response_format.model_json_schema(), indent=2)
    fallback_messages = list(messages)
    fallback_messages.append(
        {
            "role": "system",
            "content": (
                "Reply with JSON only matching this schema:\n"
                f"{schema_hint}"
            ),
        }
    )
    completion = client.chat.completions.create(
        model=model,
        messages=fallback_messages,
        temperature=temperature,
        max_tokens=get_max_tokens("MYGENASSIST_JUDGE_MAX_TOKENS", 512),
    )
    raw = (completion.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:].strip()
    data = json.loads(raw)
    return response_format.model_validate(data)
