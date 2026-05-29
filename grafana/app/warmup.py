import os
import time

from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

ES_URL = os.getenv("ES_URL", "http://elasticsearch:9200")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "medical_rag_sparse")
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
ES_INDEX = os.getenv("ES_INDEX", "medical_docs")


def main():
    # Load model (uses cache if already present)
    print("📦 Loading embedding model…")
    model = SentenceTransformer(MODEL_NAME)
    _ = model.encode(["warmup"], normalize_embeddings=True)

    # Probe ES/Qdrant and do tiny no-op queries to warm caches
    print("🌐 Probing backends…")
    es = Elasticsearch(ES_URL, request_timeout=10)
    es.cluster.health(wait_for_status="yellow", timeout="10s")
    try:
        es.search(index=ES_INDEX, body={"query": {"match_all": {}}, "size": 1})
    except Exception:
        pass

    qdr = QdrantClient(QDRANT_URL, timeout=10, prefer_grpc=False)
    qdr.get_collections()
    try:
        vec = model.encode(["dummy"], normalize_embeddings=True)[0].tolist()
        qdr.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vec,
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
    except Exception:
        pass

    print("✅ Warmup OK")


if __name__ == "__main__":
    main()
