"""Agentic RAG evaluation: multi-iteration tool use with LLM-as-judge."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from llm.mygenassist_client import get_aux_model, get_chat_model, use_mygenassist
from llm.openai_client import DEFAULT_MODEL, get_openai_client
from llm.rag_utils import build_rag_prompt
from sentence_transformers import SentenceTransformer
from tools.registry import FUNCTION_MAP, TOOLS_JSON
from tqdm.auto import tqdm


def _resolve_chat_model() -> str:
    if use_mygenassist():
        return get_chat_model()
    return os.getenv("OPENAI_DEFAULT_MODEL", DEFAULT_MODEL)


def _resolve_judge_model() -> str:
    if use_mygenassist():
        return get_aux_model()
    return os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")


def _default_approach_config() -> Dict[str, Any]:
    return {"model": _resolve_chat_model(), "temperature": 0.3}


def _ground_truth_rows(max_samples: int) -> List[Dict[str, Any]]:
    from evaluation.config_paths import load_ground_truth

    rows = load_ground_truth()[:max_samples]
    return [
        {
            "question": r["question"],
            "doc_id": r.get("doc_id", ""),
        }
        for r in rows
    ]


def _parse_judge_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


class AgenticLLMEvaluator:
    """Evaluates agentic LLM performance by testing tool usage effectiveness."""

    def __init__(self, embedding_model_name: str = "multi-qa-MiniLM-L6-cos-v1"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.client = get_openai_client()

        self.agentic_approaches = {
            "default": _default_approach_config(),
            "conservative_agent": {
                "model": _resolve_chat_model(),
                "temperature": 0.1,
            },
            "creative_agent": {
                "model": _resolve_chat_model(),
                "temperature": 0.7,
            },
        }

        self.default_settings = {
            "user_type": "Healthcare Provider",
            "response_detail": "Detailed",
            "show_sources": True,
        }

        self.tool_judge_prompt_template = """
You are an expert evaluator for an agentic RAG system that uses tools.
Evaluate how well the agent used tools and provided an accurate medical answer.

Question: {question}
Tools Used: {tools_used}
Final Answer: {final_answer}

Rate the following aspects (0-10 scale):

1. TOOL_APPROPRIATENESS: Did the agent choose appropriate tools for the medical question?
2. ANSWER_QUALITY: How accurate and helpful is the medical information provided?
3. INFORMATION_SYNTHESIS: How well did the agent combine information from multiple sources?

Provide your evaluation in JSON format:

{{
  "tool_appropriateness": <0-10>,
  "answer_quality": <0-10>,
  "information_synthesis": <0-10>,
  "overall_score": <0-10>,
  "explanation": "Brief explanation of the scores"
}}
""".strip()

    def run_agentic_conversation(
        self,
        question: str,
        approach_config: Dict,
        *,
        local: bool = True,
        max_iterations: int = 3,
    ) -> Dict:
        """Run an agentic conversation using the same tools as production RAG."""
        tools_used: List[Dict] = []
        search_results: List[Dict] = []
        search_queries: List[str] = []
        previous_actions: List[str] = []

        for iteration in range(max_iterations):
            prompt, _ranked_results = build_rag_prompt(
                question=question,
                settings=self.default_settings,
                tools=TOOLS_JSON,
                search_results=search_results,
                search_queries=search_queries,
                previous_actions=previous_actions,
                max_iter=max_iterations,
                curr_iter=iteration,
            )

            try:
                response = self.client.chat.completions.create(
                    model=approach_config["model"],
                    messages=[{"role": "user", "content": prompt}],
                    tools=TOOLS_JSON,
                    tool_choice="auto",
                    temperature=approach_config.get("temperature", 0.3),
                )

                message = response.choices[0].message

                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)

                        if tool_name in FUNCTION_MAP:
                            if tool_name == "eu_hybrid_search":
                                tool_args.setdefault("local", local)
                            tool_result = FUNCTION_MAP[tool_name](**tool_args)
                            tools_used.append(
                                {
                                    "iteration": iteration,
                                    "tool": tool_name,
                                    "args": tool_args,
                                    "result_count": (
                                        len(tool_result)
                                        if isinstance(tool_result, list)
                                        else 1
                                    ),
                                }
                            )

                            if tool_name == "eu_hybrid_search":
                                search_results.extend(tool_result)
                                search_queries.append(tool_args.get("q", ""))

                            previous_actions.append(f"TOOL:{tool_name}({tool_args})")

                elif message.content:
                    return {
                        "final_answer": message.content,
                        "tools_used": tools_used,
                        "iterations": iteration + 1,
                        "search_results": search_results,
                    }

            except Exception as e:
                print(f"Error in iteration {iteration}: {e}")
                break

        return {
            "final_answer": "Unable to complete response within iteration limit",
            "tools_used": tools_used,
            "iterations": max_iterations,
            "search_results": search_results,
        }

    def evaluate_tool_usage(self, question: str, conversation_result: Dict) -> Dict:
        """Score tool appropriateness, answer quality, and synthesis."""
        tools_used_summary = json.dumps(conversation_result["tools_used"], indent=2)
        prompt = self.tool_judge_prompt_template.format(
            question=question,
            tools_used=tools_used_summary,
            final_answer=conversation_result["final_answer"],
        )

        try:
            response = self.client.chat.completions.create(
                model=_resolve_judge_model(),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return _parse_judge_json(response.choices[0].message.content or "")
        except Exception as e:
            print(f"Error in tool usage evaluation: {e}")
            return {
                "tool_appropriateness": 0,
                "answer_quality": 0,
                "information_synthesis": 0,
                "overall_score": 0,
                "explanation": f"Error: {e}",
            }

    def evaluate_agentic_approach(
        self,
        test_data: List[Dict],
        approach_name: str,
        approach_config: Dict,
        *,
        local: bool = True,
    ) -> Dict:
        """Evaluate a single agentic LLM approach on ground-truth questions."""
        results = []
        tool_scores = []

        for item in tqdm(test_data, desc=f"Agentic {approach_name}", unit="query"):
            conversation_result = self.run_agentic_conversation(
                item["question"],
                approach_config,
                local=local,
            )
            tool_evaluation = self.evaluate_tool_usage(item["question"], conversation_result)
            tool_scores.append(tool_evaluation)

            results.append(
                {
                    "question": item["question"],
                    "expected_doc_id": item.get("doc_id", ""),
                    "final_answer": conversation_result["final_answer"],
                    "tools_used": conversation_result["tools_used"],
                    "iterations": conversation_result["iterations"],
                    "tool_appropriateness": tool_evaluation.get("tool_appropriateness", 0),
                    "answer_quality": tool_evaluation.get("answer_quality", 0),
                    "information_synthesis": tool_evaluation.get("information_synthesis", 0),
                    "overall_tool_score": tool_evaluation.get("overall_score", 0),
                    "evaluation_explanation": tool_evaluation.get("explanation", ""),
                }
            )

        metrics = {
            "approach_name": approach_name,
            "model": approach_config.get("model"),
            "total_samples": len(test_data),
            "avg_tool_appropriateness": float(
                np.mean([s.get("tool_appropriateness", 0) for s in tool_scores])
            ),
            "avg_answer_quality": float(
                np.mean([s.get("answer_quality", 0) for s in tool_scores])
            ),
            "avg_information_synthesis": float(
                np.mean([s.get("information_synthesis", 0) for s in tool_scores])
            ),
            "avg_overall_tool_score": float(
                np.mean([s.get("overall_score", 0) for s in tool_scores])
            ),
            "avg_iterations": float(np.mean([r["iterations"] for r in results])),
            "total_tool_calls": sum(len(r["tools_used"]) for r in results),
            "avg_tools_per_question": float(
                np.mean([len(r["tools_used"]) for r in results])
            ),
        }

        return {"metrics": metrics, "detailed_results": results}

    def compare_agentic_approaches(
        self,
        test_data: List[Dict],
        *,
        local: bool = True,
    ) -> Dict:
        """Compare configured agentic approaches and pick the best composite score."""
        all_results = {}
        all_metrics = []

        for approach_name, approach_config in self.agentic_approaches.items():
            result = self.evaluate_agentic_approach(
                test_data,
                approach_name,
                approach_config,
                local=local,
            )
            all_results[approach_name] = result
            all_metrics.append(result["metrics"])

        metrics_df = pd.DataFrame(all_metrics)
        metrics_df["composite_score"] = (
            0.4 * (metrics_df["avg_overall_tool_score"] / 10)
            + 0.4 * (metrics_df["avg_answer_quality"] / 10)
            + 0.2 * (1 / metrics_df["avg_iterations"].fillna(3))
        )

        best_approach = metrics_df.loc[
            metrics_df["composite_score"].idxmax(), "approach_name"
        ]

        return {
            "best_approach": best_approach,
            "metrics_comparison": metrics_df.to_dict("records"),
            "detailed_results": all_results,
        }


def evaluate_agentic_batch(
    *,
    max_samples: int = 10,
    local: bool = True,
    approach_name: Optional[str] = None,
    compare_approaches: bool = False,
) -> Dict[str, Any]:
    """
    Run agentic RAG evaluation on ground-truth questions.

    Args:
        max_samples: Number of ground-truth rows to evaluate.
        local: Pass local=True to eu_hybrid_search tool calls.
        approach_name: Named approach from AgenticLLMEvaluator.agentic_approaches.
        compare_approaches: If True, compare all configured approaches (costly).

    Returns:
        Summary dict compatible with llm_evaluator.run_full_llm_evaluation.
    """
    evaluator = AgenticLLMEvaluator()
    test_data = _ground_truth_rows(max_samples)

    if compare_approaches:
        comparison = evaluator.compare_agentic_approaches(test_data, local=local)
        best = comparison["best_approach"]
        best_result = comparison["detailed_results"][best]
        metrics = best_result["metrics"]
        return {
            "mode": "compare",
            "n": metrics["total_samples"],
            "mean_overall": metrics["avg_overall_tool_score"],
            "approach_name": best,
            "best_approach": best,
            "metrics_comparison": comparison["metrics_comparison"],
            "total_tool_calls": metrics["total_tool_calls"],
            "results": best_result["detailed_results"],
        }

    name = approach_name or "default"
    config = evaluator.agentic_approaches.get(name, _default_approach_config())
    if name not in evaluator.agentic_approaches:
        name = "default"
        config = _default_approach_config()

    result = evaluator.evaluate_agentic_approach(
        test_data,
        name,
        config,
        local=local,
    )
    metrics = result["metrics"]
    return {
        "mode": "single",
        "n": metrics["total_samples"],
        "mean_overall": metrics["avg_overall_tool_score"],
        "approach_name": name,
        "model": metrics.get("model"),
        "total_tool_calls": metrics["total_tool_calls"],
        "metrics": metrics,
        "results": result["detailed_results"],
    }


if __name__ == "__main__":
    summary = evaluate_agentic_batch(max_samples=5, local=True)
    print(json.dumps(summary, indent=2, default=str))