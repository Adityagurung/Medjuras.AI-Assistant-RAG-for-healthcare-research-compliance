"""Generate retrieval ground truth from chunk corpus via configured LLM provider."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from evaluation.config_paths import ground_truth_path
from ingestion.paths import ensure_data_dirs
from llm.provider import chat_completion, extract_json_object, get_provider, resolve_model

load_dotenv()


class GroundTruthRow(BaseModel):
    question: str
    doc_id: str = Field(description="Chunk id that answers the question")


class GroundTruthBatch(BaseModel):
    items: List[GroundTruthRow]


def load_chunks_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for line in tqdm(lines, desc="Loading chunks", unit="line"):
        rows.append(json.loads(line))
    return rows


def generate_ground_truth(
    chunks_path: Path,
    *,
    sample_size: int = 50,
    out_path: Optional[Path] = None,
    model: Optional[str] = None,
) -> Path:
    ensure_data_dirs()
    chunks = load_chunks_jsonl(chunks_path)
    if not chunks:
        raise ValueError(f"No chunks in {chunks_path}")

    sample = random.sample(chunks, min(sample_size, len(chunks)))
    provider = get_provider()
    if provider == "stub" or (
        provider == "openai" and not os.getenv("OPENAI_API_KEY")
    ) or (
        provider == "huggingface" and not (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    ):
        items = [
            {
                "question": f"What does {c.get('title', 'this document')} say about EU healthcare compliance?",
                "doc_id": c.get("id", ""),
            }
            for c in sample[: min(5, len(sample))]
        ]
        out = out_path or ground_truth_path()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(items, indent=2), encoding="utf-8")
        return out

    model = resolve_model(model)
    items: List[Dict[str, str]] = []
    system = (
        "EU medico-legal RAG ground-truth generator. "
        "Reply with JSON only with keys question and doc_id."
    )

    for c in tqdm(sample, desc=f"Generating ground truth ({provider})", unit="chunk"):
        user = (
            "Create one concise exam-style question answerable ONLY from this chunk.\n"
            f"chunk_id: {c.get('id')}\n"
            f"title: {c.get('title')}\n"
            f"text: {c.get('text', '')[:2000]}"
        )
        try:
            raw = chat_completion(system=system, user=user, model=model, temperature=0)
        except Exception as api_err:
            print("  API error, using fallback question:", api_err)
            items.append({"question": f"What does {c.get('title', 'this document')} say about EU healthcare compliance?", "doc_id": str(c.get("id", ""))})
            continue
        try:
            data = extract_json_object(raw)
            question = data.get("question", "").strip()
            doc_id = data.get("doc_id") or str(c.get("id", ""))
            if question:
                items.append({"question": question, "doc_id": doc_id})
                continue
        except Exception:
            pass
        # Fallback: use raw response if present, else deterministic non-empty question
        fallback_q = (raw or "").strip()
        if not fallback_q:
            fallback_q = f"What does {c.get('title', 'this document')} say about EU healthcare compliance?"
        items.append({"question": fallback_q[:500], "doc_id": str(c.get("id", ""))})

    out = out_path or ground_truth_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


if __name__ == "__main__":
    from ingestion.paths import CHUNKS_JSONL

    p = generate_ground_truth(CHUNKS_JSONL)
    print(f"Wrote ground truth to {p}")
