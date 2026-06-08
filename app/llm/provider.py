"""Pluggable LLM backend for bootcamp notebooks (OpenAI or stub)."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def get_provider() -> str:
    """openai | stub"""
    return (os.getenv("LLM_PROVIDER") or "openai").strip().lower()


def resolve_model(model: Optional[str] = None) -> str:
    from llm.mygenassist_client import get_chat_model, use_mygenassist

    provider = get_provider()
    if provider == "stub":
        return model or "stub"
    if use_mygenassist():
        return model or get_chat_model()
    return model or os.getenv("OPENAI_DEFAULT_MODEL") or DEFAULT_OPENAI_MODEL


def chat_completion(
    *,
    system: str,
    user: str,
    model: Optional[str] = None,
    temperature: float = 0.1,
) -> str:
    provider = get_provider()
    if provider == "stub":
        return json.dumps(
            {
                "answer": "stub response",
                "note": "stub provider — set LLM_PROVIDER=openai",
            }
        )
    return _openai_chat(system, user, model=model, temperature=temperature)


def _openai_chat(system: str, user: str, model: Optional[str], temperature: float) -> str:
    from llm.mygenassist_client import get_aux_model, get_llm_client, use_mygenassist

    if use_mygenassist() and model is None:
        model = get_aux_model()
    model = model or os.getenv("OPENAI_DEFAULT_MODEL", DEFAULT_OPENAI_MODEL)
    client = get_llm_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def extract_json_object(text: str) -> Any:
    text = (text or "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found")
    return json.loads(match.group(0))
