"""OpenAI embedding helpers for indexing and retrieval."""
from __future__ import annotations

import os
from typing import List, Sequence

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMS = int(os.getenv("OPENAI_EMBEDDING_DIMS", "1536"))


def get_embedding_client() -> OpenAI:
    """Return an OpenAI client configured for embedding calls."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for embeddings.")
    return OpenAI(api_key=api_key)


def build_embed_text(title: str, text: str) -> str:
    """Combine title and body the same way at index and query time."""
    title = (title or "").strip()
    body = (text or "").strip()
    if title and body:
        return f"{title}\n\n{body}"
    return title or body


def embed_texts(texts: Sequence[str], *, model: str | None = None) -> List[List[float]]:
    """Embed a batch of strings with OpenAI."""
    if not texts:
        return []
    client = get_embedding_client()
    model = model or EMBEDDING_MODEL
    response = client.embeddings.create(model=model, input=list(texts))
    return [item.embedding for item in response.data]


def embed_query(query: str, *, model: str | None = None) -> List[float]:
    """Embed a single search query."""
    vectors = embed_texts([query], model=model)
    return vectors[0]


def embed_document(title: str, text: str, *, model: str | None = None) -> List[float]:
    """Embed a document using title + text."""
    return embed_query(build_embed_text(title, text), model=model)
