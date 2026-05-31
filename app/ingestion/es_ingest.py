"""Index chunked corpus into Elasticsearch (BM25 fields only; dense vectors in Qdrant)."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, BulkIndexError
from tqdm.auto import tqdm

from ingestion.file_io import load_jsonl
from ingestion.paths import CHUNKS_JSONL, ensure_data_dirs

load_dotenv()

BULK_CHUNK = int(os.getenv("ES_BULK_CHUNK", "200"))


def str2bool(v) -> bool:
    """Parse common truthy CLI string values."""
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "t", "yes", "y")


def wait_for_es(es: Elasticsearch, timeout: int = 120) -> None:
    """Block until Elasticsearch responds to ping."""
    print("Waiting for Elasticsearch...", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        try:
            if es.ping():
                print("Elasticsearch is up.", flush=True)
                try:
                    es.cluster.put_settings(
                        body={
                            "persistent": {
                                "cluster.routing.allocation.disk.threshold_enabled": False
                            }
                        }
                    )
                except Exception:
                    pass
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(
        "Elasticsearch did not become ready. Start it with: docker compose up -d elasticsearch"
    )


def index_settings() -> dict:
    """Elasticsearch mapping for BM25 keyword search (no dense vectors)."""
    return {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "source_type": {"type": "keyword"},
                "title": {"type": "text", "analyzer": "english"},
                "text": {"type": "text", "analyzer": "english"},
                "jurisdiction": {"type": "keyword"},
                "source": {"type": "keyword"},
                "wiki_id": {"type": "keyword"},
            }
        },
    }


def prepare_docs(rows: List[Dict]) -> List[Tuple[str, Dict]]:
    """Build ES document payloads from chunk rows."""
    docs: List[Tuple[str, Dict]] = []
    for d in rows:
        doc_id = str(d.get("id") or d.get("chunk_id") or "")
        if not doc_id:
            continue
        doc = {
            "id": doc_id,
            "doc_id": d.get("doc_id", ""),
            "source_type": d.get("source_type", "chunk"),
            "title": d.get("title", ""),
            "text": d.get("text", ""),
            "jurisdiction": d.get("jurisdiction", "GLOBAL"),
            "source": d.get("source", "medrag"),
            "wiki_id": d.get("wiki_id"),
        }
        docs.append((doc_id, doc))
    return docs


def ensure_index(es: Elasticsearch, index: str, wipe: bool) -> None:
    """Create or recreate the target index."""
    if wipe:
        es.indices.delete(index=index, ignore_unavailable=True)
        for _ in range(30):
            if not es.indices.exists(index=index):
                break
            time.sleep(1)
    if not es.indices.exists(index=index):
        es.indices.create(index=index, body=index_settings())


def wait_for_index(es: Elasticsearch, index: str, timeout: int = 90) -> None:
    """Wait until the index primary shard is active."""
    print(f"Waiting for index '{index}'...", flush=True)
    time.sleep(3)
    health = es.cluster.health(index=index, wait_for_status="yellow", timeout=f"{timeout}s")
    print(f"Index '{index}' status: {health.get('status')}", flush=True)


def ingest_chunks(
    chunks_path: Path | None = None,
    *,
    es_url: str | None = None,
    index: str | None = None,
    wipe: bool = False,
) -> int:
    """Bulk-index chunks for BM25 search; return count indexed."""
    ensure_data_dirs()
    chunks_path = chunks_path or CHUNKS_JSONL
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks file: {chunks_path}. Run notebook 1 first.")

    es_url = es_url or os.getenv("ES_LOCAL_URL", os.getenv("ES_URL", "http://localhost:9200"))
    index = index or os.getenv("ES_INDEX", "medical_docs")
    print(f"ES URL: {es_url} | index: {index}", flush=True)
    es = Elasticsearch(es_url, request_timeout=120, retry_on_timeout=True, max_retries=5)
    wait_for_es(es)
    ensure_index(es, index, wipe=wipe)
    wait_for_index(es, index)

    prepared = prepare_docs(load_jsonl(chunks_path, desc="Loading chunks.jsonl"))
    indexed = 0
    for i in tqdm(range(0, len(prepared), BULK_CHUNK), desc="ES bulk ingest"):
        batch = prepared[i : i + BULK_CHUNK]
        actions = [{"_index": index, "_id": doc_id, "_source": doc} for doc_id, doc in batch]
        try:
            ok, _ = bulk(es, actions, raise_on_error=False)
            indexed += ok
        except BulkIndexError as exc:
            indexed += len(actions) - len(exc.errors)
    es.indices.refresh(index=index)
    return indexed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index chunks.jsonl into Elasticsearch")
    parser.add_argument("--wipe", type=str2bool, default=False)
    parser.add_argument("--chunks", type=str, default=str(CHUNKS_JSONL))
    args = parser.parse_args()
    n = ingest_chunks(Path(args.chunks), wipe=args.wipe)
    print(f"Indexed {n} documents into {os.getenv('ES_INDEX', 'medical_docs')}")
