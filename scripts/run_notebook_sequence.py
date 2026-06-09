"""Run all notebook pipelines in order (same logic as notebooks/)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "notebooks"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


def step(name: str, fn):
    """Run one pipeline step with clear logging."""
    print(f"\n{'=' * 60}\nSTEP: {name}\n{'=' * 60}", flush=True)
    fn()
    print(f"Done: {name}", flush=True)


def main():
    from evaluation.config_paths import ground_truth_path
    from evaluation.eval_utils import evaluate
    from evaluation.ground_truth import generate_ground_truth
    from evaluation.llm_evaluator import run_full_llm_evaluation, save_llm_evaluation_results
    from evaluation.results_utils import copy_to_results, results_path
    from ingestion.es_ingest import ingest_chunks as es_ingest
    from ingestion.paths import CHUNKS_JSONL, GROUND_TRUTH_JSON, ensure_data_dirs
    from ingestion.pipeline import run_ingest_pipeline
    from ingestion.qdrant_ingest import ingest_chunks as qdrant_ingest
    from search.es_search import search_elasticsearch
    from search.hybrid_search import hybrid_search
    from search.qdrant_search import search_qdrant

    ensure_data_dirs()

    # 1 — ingest (skip download if chunks already exist)
    if not CHUNKS_JSONL.exists():
        step("1_ingest", lambda: copy_to_results(run_ingest_pipeline(), "chunks.jsonl"))
    else:
        print("Chunks exist; skipping HF download. Delete chunks.jsonl to re-download.")

    # 1b — index (requires Docker ES + Qdrant)
    step("1b_es_index", lambda: es_ingest(wipe=True))
    step("1b_qdrant_index", lambda: qdrant_ingest(wipe=True))

    # 2 — ground truth
    def gt():
        p = generate_ground_truth(CHUNKS_JSONL, sample_size=200, out_path=GROUND_TRUTH_JSON)
        copy_to_results(p, "ground_truth.json")

    step("2_ground_truth", gt)

    gt = json.loads(ground_truth_path().read_text(encoding="utf-8"))
    gt_eval = [{"query": r["question"], "doc_id": r["doc_id"]} for r in gt]

    # 3–5 — retrieval metrics
    for label, fn, fname in [
        ("3_es_bm25", lambda q: search_elasticsearch(q, top_k=10, local=True), "es_bm25_metrics.json"),
        ("4_qdrant", lambda q: search_qdrant(q, top_k=10, local=True), "qdrant_dense_metrics.json"),
        ("5_hybrid", lambda q: hybrid_search(q, top_k=10, local=True), "hybrid_rrf_metrics.json"),
    ]:
        def run_eval(search_fn=fn, file_name=fname):
            metrics = evaluate(gt_eval, search_fn, top_k=10)
            out = results_path(file_name)
            out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            print(metrics)

        step(label, run_eval)

    # 6 — comprehensive LLM evaluation (offline RAG judge + agentic)
    def llm_eval():
        summary = run_full_llm_evaluation(
            rag_max_samples=20,
            agentic_max_samples=10,
            local=True,
        )
        saved = save_llm_evaluation_results(summary)
        print(summary["summary"])
        print("Saved:", saved)

    step("6_llm_evaluation", llm_eval)


if __name__ == "__main__":
    main()
