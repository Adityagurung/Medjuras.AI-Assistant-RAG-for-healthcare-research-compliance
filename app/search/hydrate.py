"""Load chunk payloads by id from processed chunks.jsonl."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Dict, List

from ingestion.paths import CHUNKS_JSONL
from search.search_utils import Hit

_CACHE: Dict[str, dict] | None = None


def _load_docs() -> Dict[str, dict]:
    """Load all chunks into an id -> row dict (cached)."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    rows: Dict[str, dict] = {}
    if not CHUNKS_JSONL.exists():
        _CACHE = rows
        return rows
    with CHUNKS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cid = row.get("id") or row.get("chunk_id")
            if cid:
                rows[str(cid)] = row
    _CACHE = rows
    return rows


def hydrate_hits(ids: List[str]) -> List[Hit]:
    """Return Hit objects for chunk ids from chunks.jsonl."""
    docs = _load_docs()
    hits: List[Hit] = []
    for doc_id in ids:
        row = docs.get(doc_id)
        if not row:
            continue
        hits.append(
            Hit(
                id=row["id"],
                title=row.get("title", ""),
                text=row.get("text", ""),
                source_type=row.get("source_type", "chunk"),
                rrf_score=0.0,
            )
        )
    return hits


def clear_hydrate_cache() -> None:
    """Clear cached chunks (after re-ingest)."""
    global _CACHE
    _CACHE = None
