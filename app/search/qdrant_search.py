"""Semantic search against Qdrant using OpenAI embeddings."""
from __future__ import annotations

import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from llm.embeddings import embed_query
from search.search_utils import Hit

load_dotenv()

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "medjuris")
_QDRANT: QdrantClient | None = None


def get_qdrant_client(local: bool = False) -> QdrantClient:
    """Return a cached Qdrant client (localhost when local=True)."""
    global _QDRANT
    if _QDRANT is None:
        if local:
            url = os.getenv("QDRANT_LOCAL_URL", "http://localhost:6333")
        else:
            url = os.getenv("QDRANT_URL", "http://qdrant:6333")
        _QDRANT = QdrantClient(url, prefer_grpc=False, timeout=30, check_compatibility=False)
    return _QDRANT


def _query_points(client: QdrantClient, query_vector: list, limit: int):
    """Run dense vector search via query_points API."""
    try:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        ).points
    except UnexpectedResponse as exc:
        if getattr(exc, "status_code", None) == 404:
            raise RuntimeError(
                f"Qdrant collection {COLLECTION_NAME!r} not found. "
                "Run indexing (notebook 1 pipeline + qdrant ingest) first."
            ) from exc
        raise


def search_qdrant(query: str, top_k: int = 5, local: bool = False) -> list[Hit]:
    """Semantic search; returns Hit objects with similarity score."""
    client = get_qdrant_client(local=local)
    vector = embed_query(query)
    points = _query_points(client, vector, top_k)
    results: list[Hit] = []
    for hit in points:
        payload = hit.payload or {}
        results.append(
            Hit(
                id=payload.get("id", "unknown"),
                title=payload.get("title", ""),
                text=payload.get("text", ""),
                source_type=payload.get("source_type", ""),
                rrf_score=hit.score,
            )
        )
    return results


def qdrant_ids(query: str, limit: int = 50, local: bool = False) -> list[str]:
    """Return document ids for hybrid RRF fusion."""
    client = get_qdrant_client(local=local)
    vector = embed_query(query)
    points = _query_points(client, vector, limit)
    ids = []
    for p in points:
        payload = p.payload or {}
        doc_id = payload.get("id")
        if doc_id:
            ids.append(doc_id)
    return ids


dense_ids = qdrant_ids
