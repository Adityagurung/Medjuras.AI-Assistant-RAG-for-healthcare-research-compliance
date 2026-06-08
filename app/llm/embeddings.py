"""Embedding helpers for indexing and retrieval."""
from __future__ import annotations

import os
from typing import List, Sequence

from dotenv import load_dotenv
from openai import OpenAI

from llm.mygenassist_client import get_embedding_model, get_llm_client, use_mygenassist

load_dotenv()

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMS = int(os.getenv("OPENAI_EMBEDDING_DIMS", "1536"))


def get_embedding_client() -> OpenAI:
    if use_mygenassist():
        return get_llm_client()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing embedding credentials")
    return OpenAI(api_key=api_key)


def resolve_embedding_model(model: str | None = None) -> str:
    if model:
        return model
    if use_mygenassist():
        return get_embedding_model()
    return EMBEDDING_MODEL


def build_embed_text(title: str, text: str) -> str:
    title = (title or "").strip()
    body = (text or "").strip()
    if title and body:
        return f"{title}\n\n{body}"
    return title or body


def embed_texts(texts: Sequence[str], *, model: str | None = None) -> List[List[float]]:
    if not texts:
        return []
    client = get_embedding_client()
    model = resolve_embedding_model(model)
    response = client.embeddings.create(model=model, input=list(texts))
    return [item.embedding for item in response.data]


def embed_query(query: str, *, model: str | None = None) -> List[float]:
    vectors = embed_texts([query], model=model)
    return vectors[0]


def embed_document(title: str, text: str, *, model: str | None = None) -> List[float]:
    return embed_query(build_embed_text(title, text), model=model)
