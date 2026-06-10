"""Generate the MedJuras evaluation notebooks."""
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

specs = {
    "1_ingest_medrag.ipynb": [
        md(
            "# Notebook 1: Ingest MedRAG from local JSON\n\n"
            "Downloads textbooks, pubmed, and medical-filtered wikipedia rows, "
            "chunks them, then indexes Elasticsearch and Qdrant.\n"
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
            + "print(f'ES count', n_es, 'Qdrant count', n_qd)\n"
        ),
    ],
    "2_ground_truth_data.ipynb": [
        md(
            "# Notebook 2: Ground truth (200 Q&A)\n\n"
            "Generates 200 exam-style questions from ingested chunks.\n"
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
        md("# Notebook 3: Keyword / BM25 evaluation (Elasticsearch)\n\nMeasures Hit@K, MRR, and nDCG using Elasticsearch BM25.\n"),
        code(
            BOOT
            + "import json\n"
            + "from evaluation.config_paths import load_ground_truth\n"
            + "from evaluation.eval_utils import evaluate\n"
            + "from evaluation.results_utils import results_path, results_images_dir\n"
            + "from evaluation.retrieval_results_charts import save_single_method_plot\n"
            + "from search.es_search import search_elasticsearch\n\n"
            + "TOP_K = 10\n"
            + "gt = load_ground_truth()\n"
            + "gt_rows = [{'query': r['question'], 'doc_id': r['doc_id']} for r in gt]\n"
            + "metrics = evaluate(gt_rows, lambda q, **kw: search_elasticsearch(q, top_k=TOP_K, local=True), top_k=TOP_K)\n"
            + "out = results_path('es_bm25_metrics.json')\n"
            + "out.write_text(json.dumps(metrics, indent=2), encoding='utf-8')\n"
            + "png = save_single_method_plot('Elasticsearch', metrics, TOP_K, results_images_dir() / 'es_bm25_metrics.png')\n"
            + "print(metrics)\n"
            + "print('Saved chart:', png)\n"
        ),
    ],
    "4_semantic_search_evaluation_qdrant.ipynb": [
        md("# Notebook 4: Semantic evaluation (Qdrant)\n\nMeasures dense retrieval quality with OpenAI embeddings.\n"),
        code(
            BOOT
            + "import json\n"
            + "from evaluation.config_paths import load_ground_truth\n"
            + "from evaluation.eval_utils import evaluate\n"
            + "from evaluation.results_utils import results_path, results_images_dir\n"
            + "from evaluation.retrieval_results_charts import save_single_method_plot\n"
            + "from search.qdrant_search import search_qdrant\n\n"
            + "TOP_K = 10\n"
            + "gt = load_ground_truth()\n"
            + "gt_rows = [{'query': r['question'], 'doc_id': r['doc_id']} for r in gt]\n"
            + "metrics = evaluate(gt_rows, lambda q, **kw: search_qdrant(q, top_k=TOP_K, local=True), top_k=TOP_K)\n"
            + "out = results_path('qdrant_dense_metrics.json')\n"
            + "out.write_text(json.dumps(metrics, indent=2), encoding='utf-8')\n"
            + "png = save_single_method_plot('Qdrant', metrics, TOP_K, results_images_dir() / 'qdrant_dense_metrics.png')\n"
            + "print(metrics)\n"
            + "print('Saved chart:', png)\n"
        ),
    ],
    "5_hybrid_search_evaluation_qdrant.ipynb": [
        md("# Notebook 5: Hybrid RRF evaluation\n\nFuses Qdrant dense and Elasticsearch BM25 with reciprocal rank fusion.\n"),
        code(
            BOOT
            + "import json\n"
            + "from evaluation.config_paths import load_ground_truth\n"
            + "from evaluation.eval_utils import evaluate\n"
            + "from evaluation.results_utils import results_path, results_images_dir\n"
            + "from evaluation.retrieval_results_charts import save_single_method_plot, save_comparison_from_result_files\n"
            + "from ingestion.paths import RESULTS_DIR\n"
            + "from search.hybrid_search import hybrid_search\n\n"
            + "TOP_K = 10\n"
            + "gt = load_ground_truth()\n"
            + "gt_rows = [{'query': r['question'], 'doc_id': r['doc_id']} for r in gt]\n"
            + "metrics = evaluate(gt_rows, lambda q, **kw: hybrid_search(q, top_k=TOP_K, local=True), top_k=TOP_K)\n"
            + "out = results_path('hybrid_rrf_metrics.json')\n"
            + "out.write_text(json.dumps(metrics, indent=2), encoding='utf-8')\n"
            + "png = save_single_method_plot('Hybrid', metrics, TOP_K, results_images_dir() / 'hybrid_rrf_metrics.png')\n"
            + "print(metrics)\n"
            + "print('Saved chart:', png)\n\n"
            + "comparison_png = save_comparison_from_result_files(RESULTS_DIR, TOP_K)\n"
            + "print('Saved comparison:', comparison_png)\n"
        ),
    ],
    "offline-rag-evaluation.ipynb": [
        md("# Offline RAG + LLM-as-judge\n\nHybrid RAG with gpt-4o-mini and an LLM judge.\n"),
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
    for name in specs:
        (root / name).write_text(json.dumps(nb(specs[name]), indent=1), encoding="utf-8")
    print("Wrote", len(specs), "notebooks to", root)


if __name__ == "__main__":
    main()
