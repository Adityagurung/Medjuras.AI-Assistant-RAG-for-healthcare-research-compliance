"""Index chunked corpus into Qdrant with OpenAI embeddings."""
from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm.auto import tqdm

from ingestion.file_io import load_jsonl
from ingestion.paths import CHUNKS_JSONL, ensure_data_dirs
from llm.embeddings import EMBEDDING_DIMS, build_embed_text, embed_texts

load_dotenv()

EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))


def str2bool(v) -> bool:
    """Parse common truthy CLI string values."""
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "t", "yes", "y")


def recreate_collection(client: QdrantClient, collection: str) -> None:
    """Drop and create a dense-only Qdrant collection."""
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=EMBEDDING_DIMS, distance=Distance.COSINE),
    )


def prepare_rows(rows: List[Dict]) -> List[Tuple[str, Dict]]:
    """Normalize chunk rows for Qdrant payload."""
    out: List[Tuple[str, Dict]] = []
    for d in rows:
        point_id = str(d.get("id") or d.get("chunk_id") or uuid.uuid4())
        payload = {
            "id": point_id,
            "doc_id": d.get("doc_id", ""),
            "source_type": d.get("source_type", "chunk"),
            "title": d.get("title", ""),
            "text": d.get("text", ""),
            "jurisdiction": d.get("jurisdiction", "GLOBAL"),
            "source": d.get("source", "medrag"),
        }
        out.append((point_id, payload))
    return out


def ingest_chunks(
    chunks_path: Path | None = None,
    *,
    qdrant_url: str | None = None,
    collection: str | None = None,
    wipe: bool = False,
    batch_size: int = EMBED_BATCH_SIZE,
) -> int:
    """Upsert embedded chunks into Qdrant; return point count."""
    ensure_data_dirs()
    chunks_path = chunks_path or CHUNKS_JSONL
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks file: {chunks_path}. Run notebook 1 first.")

    qdrant_url = qdrant_url or os.getenv(
        "QDRANT_LOCAL_URL", os.getenv("QDRANT_URL", "http://localhost:6333")
    )
    collection = collection or os.getenv("QDRANT_COLLECTION", "medjuris")
    print(f"Qdrant URL: {qdrant_url} | collection: {collection}", flush=True)
    client = QdrantClient(url=qdrant_url, prefer_grpc=False, timeout=60, check_compatibility=False)

    if wipe or not client.collection_exists(collection):
        recreate_collection(client, collection)

    prepared = prepare_rows(load_jsonl(chunks_path, desc="Loading chunks.jsonl"))
    total = 0
    for i in tqdm(range(0, len(prepared), batch_size), desc="Qdrant ingest batches"):
        batch = prepared[i : i + batch_size]
        texts = [build_embed_text(p["title"], p["text"]) for _pid, p in batch]
        vectors = embed_texts(texts)
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, pid)),
                vector=vec,
                payload=payload,
            )
            for (pid, payload), vec in zip(batch, vectors)
        ]
        client.upsert(collection_name=collection, points=points)
        total += len(points)
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index chunks.jsonl into Qdrant")
    parser.add_argument("--wipe", type=str2bool, default=True)
    parser.add_argument("--chunks", type=str, default=str(CHUNKS_JSONL))
    args = parser.parse_args()
    n = ingest_chunks(Path(args.chunks), wipe=args.wipe)
    print(f"Upserted {n} points into {os.getenv('QDRANT_COLLECTION', 'medjuris')}")
