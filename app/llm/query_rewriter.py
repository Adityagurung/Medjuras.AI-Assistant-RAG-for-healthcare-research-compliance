import json
import os

from dotenv import load_dotenv
from evaluation.eval_utils import evaluate
from openai import OpenAI
from search.hybrid_search import hybrid_search

# Load environment variables
load_dotenv()

# Query rewriting prompt template - defined outside the method
QUERY_REWRITE_PROMPT_TEMPLATE = """
{context_section}

Current user query: "{query}"

IMPORTANT: Only rewrite this query if it is a medical/healthcare question that would benefit from search tools.

For non-medical queries (greetings, general conversation, simple questions that don't require medical literature search), return the original query unchanged.

For medical queries, rewrite to be more specific, searchable, and medically precise:

QUERY EXPANSION AND REFORMULATION:
- Add relevant medical context and terminology
- Expand abbreviated or unclear terms (e.g., "MI" → "myocardial infarction")
- Include related concepts that would improve search results

SYNONYM HANDLING:
- Use standard medical terminology alongside common terms
- Include alternative phrasings (e.g., "heart attack" + "myocardial infarction")
- Add plural forms and variations where helpful

INTENT CLARIFICATION:
- Make vague queries more specific (e.g., "diabetes" → "diabetes type 2 management")
- Add context about what type of information is sought (symptoms, treatment, causes, etc.)
- Consider the conversation history to understand the user's actual intent

Keep the rewritten query concise but comprehensive. If the original query is already well-formed, make minimal changes.

Rewritten query:"""


def get_openai_client():
    return OpenAI()


def rewrite_query_with_context(query: str, chat_history: list = None) -> str:
    """
    Rewrite user query to be more specific, searchable, and medically precise.
    Includes query expansion, synonym handling, and intent clarification.
    """
    # Build context from recent conversation
    recent_context = ""
    if chat_history:
        recent_context = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in chat_history[-6:]]
        )

    context_section = (
        f"""Recent conversation context:
{recent_context}"""
        if recent_context
        else "No previous conversation context."
    )

    rewrite_prompt = QUERY_REWRITE_PROMPT_TEMPLATE.format(
        context_section=context_section, query=query
    )

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": rewrite_prompt}],
            temperature=0.1,
            max_tokens=100,
        )
        rewritten = response.choices[0].message.content.strip()
        # Remove any surrounding quotes that might be returned
        if rewritten.startswith('"') and rewritten.endswith('"'):
            rewritten = rewritten[1:-1]
        return rewritten
    except Exception as e:
        print(f"Error in query rewriting: {e}")
        return query  # Fallback to original query


if __name__ == "__main__":
    ground_truth_path = "data/evaluation/ground_truth.json"
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        gt_raw = json.load(f)
    gt = [{"query": row["question"], "doc_id": row["doc_id"]} for row in gt_raw]

    gt = gt[:10]

    top_k = 5

    # Baseline: evaluate original queries
    metrics_orig = evaluate(gt, hybrid_search, top_k=top_k, local=True)

    # Rewritten: wrap hybrid_search to rewrite the query before searching
    metrics_rw = evaluate(
        gt,
        lambda q, top_k, local: hybrid_search(
            rewrite_query_with_context(q, []), top_k=top_k, local=local
        ),
        top_k=top_k,
        local=True,
    )

    print("\n📊 Original (baseline)")
    print(
        f"Hit@{top_k} / Recall@{top_k}: {metrics_orig[f'Hit@{top_k}']:.3f} | "
        f"MRR@{top_k}: {metrics_orig[f'MRR@{top_k}']:.3f} | "
        f"MAP@{top_k}: {metrics_orig[f'MAP@{top_k}']:.3f} | "
        f"nDCG@{top_k}: {metrics_orig[f'nDCG@{top_k}']:.3f}"
    )

    print("\n📊 Rewritten")
    print(
        f"Hit@{top_k} / Recall@{top_k}: {metrics_rw[f'Hit@{top_k}']:.3f} | "
        f"MRR@{top_k}: {metrics_rw[f'MRR@{top_k}']:.3f} | "
        f"MAP@{top_k}: {metrics_rw[f'MAP@{top_k}']:.3f} | "
        f"nDCG@{top_k}: {metrics_rw[f'nDCG@{top_k}']:.3f}"
    )
    print("=" * 50)
