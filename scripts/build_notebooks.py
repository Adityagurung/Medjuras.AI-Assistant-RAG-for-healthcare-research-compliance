"""Generate the six MedJuras evaluation notebooks."""
import json
from pathlib import Path


def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str):
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": text.splitlines(keepends=True),
    }


BOOT = "import _bootstrap  # noqa: F401\n"

NOTEBOOKS = {
    "1_ingest_medrag.ipynb": [
        md(
            "# Notebook 1: Ingest MedRAG from local JSON\n\n"
            "Downloads **textbooks (700)**, **pubmed (1000)**, and **wikipedia (300, medical-filtered)** "
            "rows, merges to `records.json`, chunks to `data/processed/chunks/chunks.jsonl`, then indexes "
            "**Elasticsearch** and **Qdrant** with OpenAI `text-embedding-3-small`.\n\n"
            "**Prerequisites:** `.env` with `OPENAI_API_KEY`. For indexing: "
            "`docker compose up -d elasticsearch qdrant`.\n"
        ),
        code(
            BOOT
            + "from ingestion.pipeline import run_ingest_pipeline\n"
            + "from evaluation.results_utils import copy_to_results\n\n"
            + "path = run_ingest_pipeline()\n"
            + "print('Chunks:', path)\n"
            + "copy_to_results(path, 'chunks.jsonl')\n"
        ),
        code(
            BOOT
            + "from ingestion.es_ingest import ingest_chunks as es_ingest\n"
            + "from ingestion.qdrant_ingest import ingest_chunks as qdrant_ingest\n\n"
            + "n_es = es_ingest(wipe=True)\n"
            + "n_qd = qdrant_ingest(wipe=True)\n"
            + "print(f'ES={n_es}, Qdrant={n_qd}')\n"
        ),
    ],
    "2_ground_truth_data.ipynb": [
        md(
            "# Notebook 2: Ground truth (200 Q&A)\n\n"
            "Generates **200** exam-style questions answerable **only** from ingested chunks using "
            "**gpt-4o-mini**. Writes `data/evaluation/ground_truth.json` and copies to `results/`.\n"
        ),
        code(
            BOOT
            + "from ingestion.paths import CHUNKS_JSONL, GROUND_TRUTH_JSON\n"
            + "from evaluation.ground_truth import generate_ground_truth\n"
            + "from evaluation.results_utils import copy_to_results\n\n"
            + "p = generate_ground_truth(CHUNKS_JSONL, sample_size=200, out_path=GROUND_TRUTH_JSON)\n"
            + "copy_to_results(p, 'ground_truth.json')\n"
            + "print(p)\n"
        ),
    ],
    "3_keyword_search_evaluation_minsearch.ipynb": [
        md(
            "# Notebook 3: Keyword / BM25 evaluation (Elasticsearch)\n\n"
            "Measures Hit@K, MRR, MAP, nDCG using **Elasticsearch BM25**.\n"
        ),
        code(
            BOOT
            + "import json\n"
            + "from evaluation.config_paths import ground_truth_path\n"
            + "from evaluation.eval_utils import evaluate\n"
            + "from evaluation.results_utils import results_path\n"
            + "from search.es_search import search_elasticsearch\n\n"
            + "gt = json.loads(ground_truth_path().read_text(encoding='utf-8'))\n"
            + "gt_eval = [{'query': r['question'], 'doc_id': r['doc_id']} for r in gt]\n"
            + "metrics = evaluate(gt_eval, lambda q, **kw: search_elasticsearch(q, top_k=10, local=True), top_k=10)\n"
            + "out = results_path('es_bm25_metrics.json')\n"
            + "out.write_text(json.dumps(metrics, indent=2), encoding='utf-8')\n"
            + "print(metrics)\n"
        ),
    ],
    "4_semantic_search_evaluation_qdrant.ipynb": [
        md(
            "# Notebook 4: Semantic evaluation (Qdrant)\n\n"
            "Evaluates dense retrieval with OpenAI embeddings.\n"
        ),
        code(
            BOOT
            + "import json\n"
            + "from evaluation.config_paths import ground_truth_path\n"
            + "from evaluation.eval_utils import evaluate\n"
            + "from evaluation.results_utils import results_path\n"
            + "from search.qdrant_search import search_qdrant\n\n"
            + "gt = json.loads(ground_truth_path().read_text(encoding='utf-8'))\n"
            + "gt_eval = [{'query': r['question'], 'doc_id': r['doc_id']} for r in gt]\n"
            + "metrics = evaluate(gt_eval, lambda q, **kw: search_qdrant(q, top_k=10, local=True), top_k=10)\n"
            + "out = results_path('qdrant_dense_metrics.json')\n"
            + "out.write_text(json.dumps(metrics, indent=2), encoding='utf-8')\n"
            + "print(metrics)\n"
        ),
    ],
    "5_hybrid_search_evaluation_qdrant.ipynb": [
        md(
            "# Notebook 5: Hybrid RRF evaluation\n\n"
            "Fuses Qdrant dense (2.0) and Elasticsearch BM25 (1.0) with RRF k=60.\n"
        ),
        code(
            BOOT
            + "import json\n"
            + "from evaluation.config_paths import ground_truth_path\n"
            + "from evaluation.eval_utils import evaluate\n"
            + "from evaluation.results_utils import results_path\n"
            + "from search.hybrid_search import hybrid_search\n\n"
            + "gt = json.loads(ground_truth_path().read_text(encoding='utf-8'))\n"
            + "gt_eval = [{'query': r['question'], 'doc_id': r['doc_id']} for r in gt]\n"
            + "metrics = evaluate(gt_eval, lambda q, **kw: hybrid_search(q, top_k=10, local=True), top_k=10)\n"
            + "out = results_path('hybrid_rrf_metrics.json')\n"
            + "out.write_text(json.dumps(metrics, indent=2), encoding='utf-8')\n"
            + "print(metrics)\n"
        ),
    ],
    "offline-rag-evaluation.ipynb": [
        md(
            "# Offline RAG + LLM-as-judge\n\n"
            "Hybrid RAG with **gpt-4o-mini** and LLM judge scoring. "
            "Adjust `max_samples` for cost/time.\n"
        ),
        code(
            BOOT
            + "import json\n"
            + "from evaluation.llm_judge import evaluate_rag_batch\n"
            + "from evaluation.results_utils import results_path\n\n"
            + "summary = evaluate_rag_batch(max_samples=20, local=True)\n"
            + "out = results_path('offline_rag_judge.json')\n"
            + "out.write_text(json.dumps(summary, indent=2), encoding='utf-8')\n"
            + "print(summary)\n"
        ),
    ],
}


def main():
    root = Path(__file__).resolve().parents[1] / "notebooks"
    root.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        (root / name).write_text(json.dumps(nb(cells), indent=1), encoding="utf-8")
    print(f"Wrote {len(NOTEBOOKS)} notebooks to {root}")


if __name__ == "__main__":
    main()
