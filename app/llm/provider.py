"""Pluggable LLM backend for bootcamp notebooks (OpenAI, Hugging Face, or stub)."""
from __future__ import annotations

import json
import os
import re
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _init_ssl_certs():
    import certifi
    ca = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", ca)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", ca)


_init_ssl_certs()

ProviderName = str
DEFAULT_HF_MODEL = "zai-org/GLM-4.7-Flash"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def get_provider() -> ProviderName:
    """openai | huggingface | stub"""
    return (os.getenv("LLM_PROVIDER") or "openai").strip().lower()


def _looks_like_openai_model(model: str) -> bool:
    m = model.lower()
    return m.startswith("gpt-") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4")


def resolve_model(model: Optional[str] = None) -> Optional[str]:
    """Pick the correct model id for the active provider."""
    provider = get_provider()
    if provider == "huggingface":
        chosen = model or os.getenv("HF_MODEL") or DEFAULT_HF_MODEL
        if _looks_like_openai_model(chosen):
            chosen = os.getenv("HF_MODEL") or DEFAULT_HF_MODEL
        return chosen
    if provider == "openai":
        return model or os.getenv("OPENAI_DEFAULT_MODEL") or DEFAULT_OPENAI_MODEL
    return model


def chat_completion(
    *,
    system: str,
    user: str,
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> str:
    provider = get_provider()
    model = resolve_model(model)
    if provider == "stub":
        return _stub_response(system, user)
    if provider == "huggingface":
        return _hf_chat(system, user, model=model, temperature=temperature)
    return _openai_chat(system, user, model=model, temperature=temperature)


def _stub_response(system: str, user: str) -> str:
    return json.dumps(
        {
            "question": "What does this medical chunk describe?",
            "doc_id": "stub",
            "note": "stub provider — set LLM_PROVIDER=openai or huggingface",
        }
    )


def _openai_chat(system: str, user: str, model: Optional[str], temperature: float) -> str:
    from llm.openai_client import get_openai_client

    model = model or os.getenv("OPENAI_DEFAULT_MODEL", DEFAULT_OPENAI_MODEL)
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def _hf_chat(system: str, user: str, model: Optional[str], temperature: float) -> str:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is required when LLM_PROVIDER=huggingface")

    model = model or os.getenv("HF_MODEL") or DEFAULT_HF_MODEL

    try:
        from huggingface_hub import InferenceClient
    except ImportError as e:
        raise ImportError("pip install huggingface_hub") from e

    import certifi
    import httpx
    client = InferenceClient(
        token=token,
        timeout=float(os.getenv("HF_TIMEOUT", "120")),
    )
    out = client.chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=model,
        max_tokens=int(os.getenv("HF_MAX_TOKENS", "512")),
        temperature=temperature,
    )
    if hasattr(out, "choices") and out.choices:
        return (out.choices[0].message.content or "").strip()
    if isinstance(out, dict):
        return (
            out.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
    return str(out)


def extract_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("{"):
        return json.loads(text)
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"No JSON object in model response: {text[:200]}")
    return json.loads(match.group(0))
