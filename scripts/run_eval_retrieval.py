"""Run retrieval evaluation (notebooks 3–5) using ground_truth.json or a placeholder."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from evaluation.config_paths import ground_truth_path
from evaluation.eval_utils import evaluate
from evaluation.results_utils import results_path
from ingestion.paths import CHUNKS_JSONL, GROUND_TRUTH_JSON
from search.es_search import search_elasticsearch
from search.hybrid_search import hybrid_search
from search.qdrant_search import search_qdrant


def ensure_ground_truth(min_rows: int = 30) -> None:
    """Use existing GT or build a small placeholder from chunks."""
    p = ground_truth_path()
    if p.exists() and len(json.loads(p.read_text(encoding="utf-8"))) >= min_rows:
        return
    rows = [
        json.loads(line)
        for line in CHUNKS_JSONL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:min_rows]
    gt = [
        {"question": f"What does this passage about {r.get('title', 'medicine')} explain?", "doc_id": r["id"]}
        for r in rows
    ]
    GROUND_TRUTH_JSON.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(gt, indent=2), encoding="utf-8")
    print(f"Wrote placeholder ground truth ({len(gt)} rows) to {p}")


def main():
    ensure_ground_truth()
    gt = json.loads(ground_truth_path().read_text(encoding="utf-8"))
    ev = [{"query": r["question"], "doc_id": r["doc_id"]} for r in gt]
    # ES and Qdrant only if not already done
    todo = [
        (
            "ES BM25",
            lambda q, **kw: search_elasticsearch(q, top_k=kw.get("top_k", 10), local=kw.get("local", True)),
            "es_bm25_metrics.json",
        ),
        (
            "Qdrant",
            lambda q, **kw: search_qdrant(q, top_k=kw.get("top_k", 10), local=kw.get("local", True)),
            "qdrant_dense_metrics.json",
        ),
        (
            "Hybrid RRF",
            lambda q, **kw: hybrid_search(q, top_k=kw.get("top_k", 10), local=kw.get("local", True)),
            "hybrid_rrf_metrics.json",
        ),
    ]
    import os as _os
    if _os.getenv("EVAL_ONLY_HYBRID"):
        todo = [todo[-1]]
    for label, fn, fname in todo:
        metrics = evaluate(ev, fn, top_k=10, local=True)
        out = results_path(fname)
        out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"\n{label} -> {out}\n{metrics}")


if __name__ == "__main__":
    main()
