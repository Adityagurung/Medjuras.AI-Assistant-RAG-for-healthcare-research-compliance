"""Hybrid retrieval: Qdrant dense + Elasticsearch BM25 fused with RRF."""
from __future__ import annotations

import os

from dotenv import load_dotenv

from search.es_search import es_ids
from search.qdrant_search import qdrant_ids
from search.hydrate import hydrate_hits
from search.search_utils import Hit

load_dotenv()

RRF_K = int(os.getenv("RRF_K", "60"))
WEIGHT_DENSE = float(os.getenv("RRF_WEIGHT_DENSE", "2.0"))
WEIGHT_SPARSE = float(os.getenv("RRF_WEIGHT_SPARSE", "1.0"))


def reciprocal_rank_fusion(
    ranked_lists: list[tuple[list[str], float]],
    *,
    k: int = RRF_K,
    top_k: int = 5,
) -> list[str]:
    """Fuse multiple ranked id lists with weighted RRF."""
    scores: dict[str, float] = {}
    for ids, weight in ranked_lists:
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight * (1.0 / (k + rank))
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]


def hybrid_search(query: str, top_k: int = 5, local: bool = True) -> list[Hit]:
    """Run dual retrieval and return hydrated Hit objects with rrf_score."""
    dense = qdrant_ids(query, limit=50, local=local)
    sparse = es_ids(query, limit=50, local=local)
    fused_ids = reciprocal_rank_fusion(
        [(dense, WEIGHT_DENSE), (sparse, WEIGHT_SPARSE)],
        top_k=top_k,
    )
    docs = hydrate_hits(fused_ids)
    rank_scores = {
        doc_id: WEIGHT_DENSE * (1.0 / (RRF_K + i))
        for i, doc_id in enumerate(dense[:top_k], start=1)
    }
    for i, doc_id in enumerate(sparse[:top_k], start=1):
        rank_scores[doc_id] = rank_scores.get(doc_id, 0.0) + WEIGHT_SPARSE * (
            1.0 / (RRF_K + i)
        )
    for d in docs:
        d.rrf_score = rank_scores.get(d.id, 0.0)
    docs.sort(key=lambda h: -h.rrf_score)
    return docs


eu_hybrid_search = hybrid_search
