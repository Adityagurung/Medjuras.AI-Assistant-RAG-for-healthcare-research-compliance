"""Unified LLM evaluation for the MedJuras pipeline.

Orchestrates:
  1. Offline hybrid RAG + 11-criteria LLM judge (llm_judge.evaluate_rag_batch)
  2. Agentic tool-use RAG evaluation (llm_evaluation.evaluate_agentic_batch)

Used by notebook 6, scripts/run_notebook_sequence.py, and CLI entry points.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from evaluation.results_utils import results_path


def run_offline_rag_evaluation(
    *,
    max_samples: int = 20,
    local: bool = True,
) -> Dict[str, Any]:
    """Hybrid retrieve -> generate -> LLM judge on ground-truth questions."""
    from evaluation.llm_judge import evaluate_rag_batch

    return evaluate_rag_batch(max_samples=max_samples, local=local)


def run_agentic_evaluation(
    *,
    max_samples: int = 10,
    local: bool = True,
    approach_name: Optional[str] = None,
    compare_approaches: bool = False,
) -> Dict[str, Any]:
    """Multi-iteration agentic RAG with tool-use scoring."""
    from evaluation.llm_evaluation import evaluate_agentic_batch

    return evaluate_agentic_batch(
        max_samples=max_samples,
        local=local,
        approach_name=approach_name,
        compare_approaches=compare_approaches,
    )


def run_full_llm_evaluation(
    *,
    rag_max_samples: int = 20,
    agentic_max_samples: int = 10,
    local: bool = True,
    compare_approaches: bool = False,
) -> Dict[str, Any]:
    """Run both offline RAG judge and agentic evaluation; return combined summary."""
    offline_rag = run_offline_rag_evaluation(max_samples=rag_max_samples, local=local)
    agentic = run_agentic_evaluation(
        max_samples=agentic_max_samples,
        local=local,
        compare_approaches=compare_approaches,
    )

    return {
        "offline_rag": offline_rag,
        "agentic": agentic,
        "summary": {
            "rag_n": offline_rag.get("n", 0),
            "rag_mean_overall": offline_rag.get("mean_overall", 0.0),
            "agentic_n": agentic.get("n", 0),
            "agentic_mean_overall": agentic.get("mean_overall", 0.0),
            "agentic_approach": agentic.get("approach_name"),
            "agentic_tool_calls": agentic.get("total_tool_calls", 0),
        },
    }


def save_llm_evaluation_results(
    results: Dict[str, Any],
    *,
    combined_name: str = "llm_evaluation.json",
    save_legacy: bool = True,
) -> Dict[str, Path]:
    """Persist evaluation outputs under results/."""
    paths: Dict[str, Path] = {}

    combined = results_path(combined_name)
    combined.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    paths["combined"] = combined

    if save_legacy and "offline_rag" in results:
        legacy_rag = results_path("offline_rag_judge.json")
        legacy_rag.write_text(
            json.dumps(results["offline_rag"], indent=2, default=str),
            encoding="utf-8",
        )
        paths["offline_rag_judge"] = legacy_rag

    if save_legacy and "agentic" in results:
        legacy_agentic = results_path("agentic_evaluation.json")
        legacy_agentic.write_text(
            json.dumps(results["agentic"], indent=2, default=str),
            encoding="utf-8",
        )
        paths["agentic_evaluation"] = legacy_agentic

    return paths


if __name__ == "__main__":
    summary = run_full_llm_evaluation()
    saved = save_llm_evaluation_results(summary)
    print(json.dumps(summary["summary"], indent=2))
    print("Saved:", {k: str(v) for k, v in saved.items()})