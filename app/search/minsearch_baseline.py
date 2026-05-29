"""TF-IDF sparse retrieval over local chunk corpus (processed or seed JSONL)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_CHUNKS = PROJECT_ROOT / "data" / "processed" / "chunks" / "chunks.jsonl"
SEED_CHUNKS = PROJECT_ROOT / "data" / "seed" / "chunks.jsonl"

_DOCS: Dict[str, dict] = {}
_INDEX = None


def _chunks_path() -> Path:
    if PROCESSED_CHUNKS.is_file():
        return PROCESSED_CHUNKS
    if SEED_CHUNKS.is_file():
        return SEED_CHUNKS
    raise FileNotFoundError(
        f"No chunks corpus found. Expected {PROCESSED_CHUNKS} or {SEED_CHUNKS}."
    )


def _load_chunks() -> List[dict]:
    path = _chunks_path()
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load() -> None:
    global _DOCS, _INDEX
    if _INDEX is not None:
        return

    rows = _load_chunks()
    _DOCS = {row["id"]: row for row in rows if row.get("id")}

    scripts_dir = PROJECT_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from minsearch import Index

    docs = [
        {
            "id": row["id"],
            "title": row.get("title", ""),
            "text": row.get("text", ""),
            "source_type": row.get("source_type", "eu_corpus"),
        }
        for row in rows
        if row.get("id")
    ]
    _INDEX = Index(
        text_fields=["title", "text"],
        keyword_fields=["source_type"],
    ).fit(docs)


def minsearch_ids(q: str, limit: int = 50) -> List[str]:
    _load()
    hits = _INDEX.search(q, num_results=limit)
    return [doc["id"] for doc in hits]


def hydrate_hits(ids: List[str]) -> List[SimpleNamespace]:
    _load()
    out: List[SimpleNamespace] = []
    for doc_id in ids:
        row = _DOCS.get(doc_id)
        if not row:
            continue
        out.append(
            SimpleNamespace(
                id=row["id"],
                title=row.get("title", ""),
                text=row.get("text", ""),
                source_type=row.get("source_type", "eu_corpus"),
                rrf_score=0.0,
            )
        )
    return out
