# scripts/load_to_elasticsearch.py
import argparse
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
import hashlib
import json
import os
import time
from typing import Dict, Iterable, List

import numpy as np
from elasticsearch import Elasticsearch, exceptions, helpers
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from ingestion.paths import CHUNKS_JSONL, MEDRAG_RAW_DIR


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "t", "yes", "y")


def wait_for_es(es, timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        try:
            if es.ping():
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("Elasticsearch did not become ready in time")


def load_json_array(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_doc_id(d, source_type, prefix=True, hash_if_missing=False):
    """
    Stable, collision-proof _id so reruns don't duplicate.
    """
    base = d.get("id") or d.get("title", "")
    if not base:
        base = hashlib.sha1(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()[
            :20
        ]
    if prefix:
        base = f"{source_type}:{base}"
    if hash_if_missing:
        # optional: hash everything to avoid weird characters in IDs
        base = hashlib.sha1(base.encode("utf-8")).hexdigest()[:20]
    return base


def embed_title_plus_text(title: str, text: str) -> List[float]:
    # Creates document embeddings with the same model you’ll use for queries
    combo = (title or "").strip()
    if text:
        combo = (combo + "\n\n" + text).strip() if combo else text
    vec = model.encode([combo], normalize_embeddings=True)[0]
    return vec.tolist() if isinstance(vec, np.ndarray) else vec


def iter_docs() -> Iterable[Dict]:
    for path, source_type in SOURCES:
        rows = load_json_array(path) if str(path).endswith(".json") else [json.loads(ln) for ln in open(path, "r", encoding="utf-8") if ln.strip()]
        for d in rows:
            _id_raw = d.get("id", "")
            doc_id = make_doc_id(
                d, source_type, prefix=PREFIX_IDS, hash_if_missing=False
            )
            doc = {
                "id": _id_raw if not PREFIX_IDS else f"{source_type}:{_id_raw}",
                "source_type": source_type,  # derived from file
                "title": d.get("title", ""),
                "text": d.get("text", ""),
                # "wiki_id": d.get("wiki_id", ""),
                "source": d.get("source", ""),
                "url": d.get("url", None),  # only if present
            }
            doc["text_vector"] = embed_title_plus_text(doc["title"], doc["text"])
            yield doc_id, doc


def ensure_index(wipe: bool):
    exists = es.indices.exists(index=INDEX)
    if wipe:
        print(f"🧹 Wiping index '{INDEX}' (if exists) and recreating…")
        # delete alias with the same name (if any)
        try:
            if es.indices.exists_alias(name=INDEX):
                print(f"🔗 Found alias '{INDEX}', deleting…")
                es.indices.delete_alias(index="*", name=INDEX)
        except Exception as e:
            print(f"(alias delete skipped: {e})")

        # delete index
        es.indices.delete(index=INDEX, ignore_unavailable=True)

        # wait until it's actually gone
        for _ in range(30):
            if not es.indices.exists(index=INDEX) and not es.indices.exists_alias(
                name=INDEX
            ):
                break
            time.sleep(1)

        # now (re)create
        es.indices.create(
            index=INDEX, body=index_settings, timeout="60s", master_timeout="60s"
        )
        return

    if not exists:
        print(f"📦 Creating index '{INDEX}' (did not exist)…")
        es.indices.create(
            index=INDEX, body=index_settings, timeout="60s", master_timeout="60s"
        )
    else:
        print(f"📦 Using existing index '{INDEX}' (will upsert/overwrite by _id).")


def es_knn(query, k=10, num_candidates=1000):
    qvec = model.encode([query], normalize_embeddings=True)[0].tolist()
    body = {
        "knn": {
            "field": "text_vector",
            "query_vector": qvec,
            "k": k,
            "num_candidates": num_candidates,
        },
        "_source": ["id", "source_type", "title", "text", "wiki_id", "source", "url"],
    }
    return es.search(index=INDEX, body=body)["hits"]["hits"]


def es_bm25(query, k=10, source_type=None):
    body = {
        "query": {
            "bool": {
                "must": {
                    "multi_match": {"query": query, "fields": ["title^2", "text"]}
                },
                "filter": (
                    [{"term": {"source_type": source_type}}] if source_type else []
                ),
            }
        },
        "_source": ["id", "source_type", "title", "text", "wiki_id", "source", "url"],
        "size": k,
    }
    return es.search(index=INDEX, body=body)["hits"]["hits"]


def old_main():
    total, skipped = 0, 0
    for doc in tqdm(iter_docs(), desc="Indexing to ES"):
        es.index(index=INDEX, id=doc["id"], document=doc, request_timeout=100)
        total += 1

        print(
            f"→ Indexed {doc.get('id','<no-id>')} [{doc.get('source_type','?')}] {doc.get('title','')[:80]}"
        )

    es.indices.refresh(index=INDEX)
    print(f"✅ Ingested/updated {total} docs into ES index '{INDEX}'")
    print("Count:", es.count(index=INDEX))


def main(verbose_skips=False):
    total, skipped = 0, 0
    VERBOSE_SKIPS = verbose_skips  # If you want to see the skipps
    for doc_id, doc in tqdm(
        iter_docs(),
        desc="Indexing to ES",
        unit="doc",
        dynamic_ncols=True,
        total=sum((len(load_json_array(p)) if str(p).endswith(".json") else sum(1 for ln in open(p, "r", encoding="utf-8") if ln.strip())) for p, _ in SOURCES),
    ):
        # skip embedding if already present
        if es.exists(index=INDEX, id=doc_id):
            skipped += 1
            if VERBOSE_SKIPS:
                print(
                    f"↻ Skipped (exists) {doc_id} [{doc.get('source_type','?')}] {doc.get('title','')[:80]}"
                )
            continue

        # now embed and create
        doc["text_vector"] = embed_title_plus_text(doc["title"], doc["text"])
        try:
            es.create(index=INDEX, id=doc_id, document=doc, request_timeout=100)
            print(
                f"→ Created {doc_id} [{doc.get('source_type','?')}] {doc.get('title','')[:80]}"
            )
            total += 1
        except exceptions.ConflictError:
            skipped += 1
            if VERBOSE_SKIPS:
                print(
                    f"↻ Skipped (exists) {doc_id} [{doc.get('source_type','?')}] {doc.get('title','')[:80]}"
                )

        if (total + skipped) % 500 == 0:
            print(f"… processed {total + skipped} (indexed={total}, skipped={skipped})")

    es.indices.refresh(index=INDEX)
    print(f"✅ Done. Indexed={total}, Skipped(existing)={skipped}")
    print("Count:", es.count(index=INDEX))


if __name__ == "__main__":
    ES_URL = os.getenv("ES_URL", "http://localhost:9200")
    INDEX = os.getenv("ES_INDEX", "medical_docs")
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims; cosine-friendly
    model = SentenceTransformer(MODEL_NAME)

    index_settings = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "source_type": {"type": "keyword"},  # wikipedia | textbook | pubmed
                "title": {"type": "text", "analyzer": "english"},
                "text": {"type": "text", "analyzer": "english"},
                # "wiki_id": {"type": "keyword"},
                "source": {"type": "keyword"},
                "url": {"type": "keyword"},
                "text_vector": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        },
    }

    # Seed JSON files under data/raw/medrag/ (see ingestion.paths.MEDRAG_RAW_DIR)
    # unify heterogeneous data under a common shape;can filter by source_type later.
    if CHUNKS_JSONL.exists():
        SOURCES = [(str(CHUNKS_JSONL), "chunk")]
    else:
        SOURCES = [
            (str(MEDRAG_RAW_DIR / "medical_textbook_seed.json"), "textbook"),
            (str(MEDRAG_RAW_DIR / "medical_pubmed_seed.json"), "pubmed"),
        ]

    PREFIX_IDS = False  # this id didnt work "textbook:Anatomy_Gray_2"

    es = Elasticsearch(
        ES_URL,
        request_timeout=100,
        retry_on_timeout=True,
        max_retries=5,
        http_compress=True,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wipe",
        type=str2bool,
        default=False,
        help="If true, delete & recreate the index before ingest; else upsert/extend.",
    )
    args = parser.parse_args()

    print(f"🔧 Connecting to ES at {ES_URL} | index='{INDEX}' | wipe={args.wipe}")
    wait_for_es(es)
    ensure_index(wipe=args.wipe)
    try:
        main(verbose_skips=False)
    except KeyboardInterrupt:
        es.indices.refresh(index=INDEX)
        print("\n⏹️ Cancelled by user.")
        print("Count:", es.count(index=INDEX))

    print([d["_source"]["title"] for d in es_bm25("gross anatomy")])
    print([d["_source"]["title"] for d in es_knn("gross anatomy")])
