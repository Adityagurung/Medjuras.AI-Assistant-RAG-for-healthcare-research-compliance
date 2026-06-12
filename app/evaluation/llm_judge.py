from pydantic import BaseModel, Field
from typing import List
import os
import json
from llm.mygenassist_client import get_aux_model, get_chat_model, get_llm_client, structured_chat_parse, use_mygenassist

CRITERIA = ['factual_accuracy','citation_quality','eu_jurisdiction_alignment','regulatory_compliance','privacy_gdpr','clarity','completeness','harm_avoidance','source_recency','bias_fairness','actionable_guidance']

class CriterionScore(BaseModel):
    name: str
    score: int = Field(ge=1, le=5)
    rationale: str

class JudgeResult(BaseModel):
    criteria: List[CriterionScore]
    overall: float = 0.0

def llm_judge(question, answer, context=None, model=None):
    prompt = 'Question: ' + str(question) + '\nAnswer: ' + str(answer) + '\nContext: ' + str(context or '')
    client = get_llm_client()
    chosen_model = model or (get_aux_model() if use_mygenassist() else os.getenv('OPENAI_DEFAULT_MODEL', 'gpt-4o-mini'))
    try:
        parsed = structured_chat_parse(
            client,
            model=chosen_model,
            messages=[
                {'role': 'system', 'content': 'Score EU medico-legal RAG answers.'},
                {'role': 'user', 'content': prompt},
            ],
            response_format=JudgeResult,
            temperature=0,
        )
    except Exception:
        return JudgeResult(criteria=[], overall=0.0)
    if not parsed.overall:
        total = sum(c.score for c in parsed.criteria)
        parsed.overall = total / max(len(parsed.criteria), 1)
    return parsed


def evaluate_rag_batch(
    *,
    max_samples: int = 20,
    local: bool = True,
) -> dict:
    """
    Run hybrid RAG answers on ground-truth questions and LLM-judge scores.

    Args:
        max_samples: Number of ground-truth rows to evaluate.
        local: Use localhost ES/Qdrant URLs.

    Returns:
        Summary dict with per-question scores and mean overall.
    """
    from evaluation.config_paths import load_ground_truth
    from search.hybrid_search import hybrid_search
    from tqdm.auto import tqdm

    rows = load_ground_truth()[:max_samples]
    client = get_llm_client()
    model = get_chat_model() if use_mygenassist() else os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
    results = []

    for row in tqdm(rows, desc="RAG + LLM judge", unit="query"):
        question = row["question"]
        hits = hybrid_search(question, top_k=5, local=local)
        context = "\n\n".join(f"{h.title}\n{h.text[:500]}" for h in hits)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Answer from context only. Jurisdiction: GLOBAL. Cite chunk titles.",
                },
                {"role": "user", "content": f"Q: {question}\n\nContext:\n{context}"},
            ],
            temperature=0.1,
            max_tokens=400,
        )
        answer = (resp.choices[0].message.content or "").strip()
        judged = llm_judge(question, answer, context=context)
        results.append(
            {
                "question": question,
                "expected_doc_id": row.get("doc_id"),
                "overall": judged.overall,
                "criteria": [c.model_dump() for c in judged.criteria],
            }
        )

    overalls = [r["overall"] for r in results if r["overall"]]
    return {
        "n": len(results),
        "mean_overall": sum(overalls) / len(overalls) if overalls else 0.0,
        "results": results,
    }
