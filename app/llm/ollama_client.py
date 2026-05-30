"""Local Ollama chat client for Streamlit (llama3.2 by default)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


@dataclass
class OllamaResponse:
    """Structured response from an Ollama chat call."""

    text: str
    model: str


def ollama_chat(
    prompt: str,
    *,
    system: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.1,
) -> OllamaResponse:
    """Call Ollama /api/chat and return assistant text."""
    model = model or OLLAMA_MODEL
    messages: List[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    text = (data.get("message") or {}).get("content", "") or ""
    return OllamaResponse(text=text.strip(), model=model)
