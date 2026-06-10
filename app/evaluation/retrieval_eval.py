import json
import os

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from elasticsearch import Elasticsearch
from evaluation.eval_utils import evaluate
from qdrant_client import QdrantClient
from search.es_search import search_elasticsearch
from search.hybrid_search import hybrid_search
from search.qdrant_search import search_qdrant

# Configuration - no client instantiation at module level
ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "medical_docs")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "medical_rag_sparse")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_es_client(local=False):
    """Get ES client with correct URL based on environment"""
    if local:
        url = os.getenv("ES_LOCAL_URL", "http://localhost:9200")
    else:
        url = os.getenv("ES_URL", "http://elasticsearch:9200")
    return Elasticsearch(url, request_timeout=30)


def get_qdrant_client(local=False):
    """Get Qdrant client with correct URL based on environment"""
    if local:
        url = os.getenv("QDRANT_LOCAL_URL", "http://localhost:6333")
    else:
        url = os.getenv("QDRANT_URL", "http://qdrant:6333")
    return QdrantClient(url, timeout=30, prefer_grpc=False)


if __name__ == "__main__":
    from evaluation.config_paths import ground_truth_path as gt_path_fn
    gt_path = gt_path_fn()
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_raw = json.load(f)
    gt = [{"query": row["question"], "doc_id": row["doc_id"]} for row in gt_raw]

    top_k = 2
    # --- Run evaluation on the ground truth ---
    hybrid_metrics = evaluate(gt, hybrid_search, top_k=top_k)

    hybrid_hit = hybrid_metrics[f"Hit@{top_k}"]
    hybrid_mrr = hybrid_metrics[f"MRR@{top_k}"]
    hybrid_ndcg = hybrid_metrics[f"nDCG@{top_k}"]

    print("\n📊 Hybrid Evaluation Results (aggregate, per-query metrics)")
    print(
        f"Hit@{top_k}: {hybrid_hit:.3f} | MRR@{top_k}: {hybrid_mrr:.3f} | nDCG@{top_k}: {hybrid_ndcg:.3f}"
    )

    es_metrics = evaluate(gt, search_elasticsearch, top_k=top_k)

    es_hit = es_metrics[f"Hit@{top_k}"]
    es_mrr = es_metrics[f"MRR@{top_k}"]
    es_ndcg = es_metrics[f"nDCG@{top_k}"]

    print("\n📊 Qdrant Evaluation Results (aggregate, per-query metrics)")
    print(
        f"Hit@{top_k}: {es_hit:.3f} | MRR@{top_k}: {es_mrr:.3f} | nDCG@{top_k}: {es_ndcg:.3f}"
    )

    qdrant_metrics = evaluate(gt, search_qdrant, top_k=top_k)

    qdrant_hit = qdrant_metrics[f"Hit@{top_k}"]
    qdrant_mrr = qdrant_metrics[f"MRR@{top_k}"]
    qdrant_ndcg = qdrant_metrics[f"nDCG@{top_k}"]

    print("\n📊 Elastic Search Evaluation Results (aggregate, per-query metrics)")
    print(
        f"Hit@{top_k}: {qdrant_hit:.3f} | MRR@{top_k}: {qdrant_mrr:.3f} | nDCG@{top_k}: {qdrant_ndcg:.3f}"
    )

    sample_queries = [gt[0]["query"], gt[6]["query"], gt[-1]["query"]]
    print("\n🧾 Hydrated Top-3 Results Per Retriever (title + text)")
    print("=" * 50)
    for query in sample_queries:
        print(f"\n📋 Query: {query}")
        try:

            hybrid_results = hybrid_search(query, top_k=top_k)
            es_results = search_elasticsearch(query, top_k=top_k)
            qdrant_results = search_qdrant(query, top_k=top_k)

            for r in [hybrid_results, es_results, qdrant_results]:
                if not r:
                    print("  (no results)")
                else:
                    for i, d in enumerate(r, 1):
                        print(f"\n  {i}. Score: {d.rrf_score:.4f}")
                        print(f"     Source: {d.source_type}")
                        print(f"     Title:  {d.title}")
                        snippet = d.text if len(d.text) < 600 else d.text[:600] + "..."
                        print(f"     Text:   {snippet}")
                        print(
                            "---------------------------------------------------------------"
                        )

        except Exception as e:
            print(f"❌ Error: {e}")
        print("-" * 30)
