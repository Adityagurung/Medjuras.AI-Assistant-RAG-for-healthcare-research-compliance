import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import openai
import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from llm.rag_utils import build_rag_context, build_rag_prompt
from tools.registry import FUNCTION_MAP, TOOLS_JSON


class AgenticLLMEvaluator:
    """
    Evaluates agentic LLM performance by testing tool usage effectiveness.
    Tests how well different LLM approaches use the available RAG tools.
    """

    def __init__(self, embedding_model_name: str = "multi-qa-MiniLM-L6-cos-v1"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.client = OpenAI()

        # Define multiple LLM approaches to compare for agentic behavior
        self.agentic_approaches = {
            "gpt-4o-mini": {"model": "gpt-4o-mini", "temperature": 0.3},
            "gpt-4o": {"model": "gpt-4o", "temperature": 0.3},
            "gpt-3.5-turbo": {"model": "gpt-3.5-turbo", "temperature": 0.3},
            "conservative_agent": {"model": "gpt-4o-mini", "temperature": 0.1},
            "creative_agent": {"model": "gpt-4o-mini", "temperature": 0.7},
        }

        # Settings for the RAG system
        self.default_settings = {
            "user_type": "Healthcare Provider",
            "response_detail": "Detailed",
            "show_sources": True,
        }

        # Agentic Tool Usage Judge prompt template
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
        self, question: str, approach_config: Dict, max_iterations: int = 3
    ) -> Dict:
        """Run an agentic conversation using tools like your RAG system"""
        tools_used = []
        search_results = []
        search_queries = []
        previous_actions = []

        for iteration in range(max_iterations):
            # Build the RAG prompt using your existing system
            prompt, ranked_results = build_rag_prompt(
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
                # Call LLM with tools
                response = self.client.chat.completions.create(
                    model=approach_config["model"],
                    messages=[{"role": "user", "content": prompt}],
                    tools=TOOLS_JSON,
                    tool_choice="auto",
                    temperature=approach_config.get("temperature", 0.3),
                )

                message = response.choices[0].message

                # Check if LLM wants to use a tool
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)

                        # Execute the tool
                        if tool_name in FUNCTION_MAP:
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

                            # Update search context
                            if tool_name == "hybrid_search":
                                search_results.extend(tool_result)
                                search_queries.append(tool_args.get("query", ""))

                            previous_actions.append(f"TOOL:{tool_name}({tool_args})")

                # If we have content (final answer), break
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

        # If we exhausted iterations, return what we have
        return {
            "final_answer": "Unable to complete response within iteration limit",
            "tools_used": tools_used,
            "iterations": max_iterations,
            "search_results": search_results,
        }

    def compute_cosine_similarity(self, answer1: str, answer2: str) -> float:
        """Compute cosine similarity between two answers"""
        try:
            v1 = self.embedding_model.encode(answer1)
            v2 = self.embedding_model.encode(answer2)
            return float(np.dot(v1, v2))
        except Exception as e:
            print(f"Error computing similarity: {e}")
            return 0.0

    def evaluate_tool_usage(self, question: str, conversation_result: Dict) -> Dict:
        """Evaluate how well the agent used tools"""
        tools_used_summary = json.dumps(conversation_result["tools_used"], indent=2)

        prompt = self.tool_judge_prompt_template.format(
            question=question,
            tools_used=tools_used_summary,
            final_answer=conversation_result["final_answer"],
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Use best model for evaluation
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )

            evaluation_text = response.choices[0].message.content
            evaluation = json.loads(evaluation_text)
            return evaluation
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
        sample_size: int = 10,
    ) -> Dict:
        """Evaluate a single agentic LLM approach"""
        print(f"\nEvaluating agentic approach: {approach_name}")

        # Sample test data if larger than sample_size (smaller for agentic due to cost)
        if len(test_data) > sample_size:
            test_sample = (
                pd.DataFrame(test_data)
                .sample(n=sample_size, random_state=42)
                .to_dict("records")
            )
        else:
            test_sample = test_data

        results = []
        tool_scores = []
        cosine_scores = []

        for item in tqdm(test_sample, desc=f"Testing {approach_name}"):
            # Run agentic conversation
            conversation_result = self.run_agentic_conversation(
                item["question"], approach_config
            )

            # Evaluate tool usage
            tool_evaluation = self.evaluate_tool_usage(
                item["question"], conversation_result
            )
            tool_scores.append(tool_evaluation)

            # Compute cosine similarity if ground truth available
            cosine_score = None
            if "ground_truth_answer" in item:
                cosine_score = self.compute_cosine_similarity(
                    conversation_result["final_answer"], item["ground_truth_answer"]
                )
                cosine_scores.append(cosine_score)

            results.append(
                {
                    "question": item["question"],
                    "final_answer": conversation_result["final_answer"],
                    "tools_used": conversation_result["tools_used"],
                    "iterations": conversation_result["iterations"],
                    "ground_truth_answer": item.get("ground_truth_answer", ""),
                    "cosine_similarity": cosine_score,
                    "tool_appropriateness": tool_evaluation.get(
                        "tool_appropriateness", 0
                    ),
                    "answer_quality": tool_evaluation.get("answer_quality", 0),
                    "information_synthesis": tool_evaluation.get(
                        "information_synthesis", 0
                    ),
                    "overall_tool_score": tool_evaluation.get("overall_score", 0),
                    "evaluation_explanation": tool_evaluation.get("explanation", ""),
                }
            )

        # Calculate aggregate metrics
        metrics = {
            "approach_name": approach_name,
            "total_samples": len(test_sample),
            "avg_tool_appropriateness": np.mean(
                [s.get("tool_appropriateness", 0) for s in tool_scores]
            ),
            "avg_answer_quality": np.mean(
                [s.get("answer_quality", 0) for s in tool_scores]
            ),
            "avg_information_synthesis": np.mean(
                [s.get("information_synthesis", 0) for s in tool_scores]
            ),
            "avg_overall_tool_score": np.mean(
                [s.get("overall_score", 0) for s in tool_scores]
            ),
            "cosine_mean": np.mean(cosine_scores) if cosine_scores else None,
            "cosine_std": np.std(cosine_scores) if cosine_scores else None,
            "avg_iterations": np.mean([r["iterations"] for r in results]),
            "total_tool_calls": sum(len(r["tools_used"]) for r in results),
            "avg_tools_per_question": np.mean([len(r["tools_used"]) for r in results]),
        }

        return {"metrics": metrics, "detailed_results": results}

    def compare_agentic_approaches(
        self, test_data: List[Dict], sample_size: int = 10
    ) -> Dict:
        """Compare all agentic LLM approaches and select the best one"""
        print("Starting agentic LLM approach comparison...")

        all_results = {}
        all_metrics = []

        for approach_name, approach_config in self.agentic_approaches.items():
            result = self.evaluate_agentic_approach(
                test_data, approach_name, approach_config, sample_size
            )
            all_results[approach_name] = result
            all_metrics.append(result["metrics"])

        # Convert to DataFrame for easy comparison
        metrics_df = pd.DataFrame(all_metrics)

        # Select best approach based on composite score
        # Weight: 40% tool usage + 40% answer quality + 20% efficiency
        metrics_df["composite_score"] = (
            0.4 * (metrics_df["avg_overall_tool_score"] / 10)  # Normalize to 0-1
            + 0.4 * (metrics_df["avg_answer_quality"] / 10)  # Normalize to 0-1
            + 0.2
            * (
                1 / metrics_df["avg_iterations"].fillna(3)
            )  # Efficiency: fewer iterations better
        )

        best_approach = metrics_df.loc[
            metrics_df["composite_score"].idxmax(), "approach_name"
        ]

        print(f"\n🏆 Best performing agentic approach: {best_approach}")
        print(f"Composite score: {metrics_df['composite_score'].max():.3f}")

        return {
            "best_approach": best_approach,
            "metrics_comparison": metrics_df,
            "detailed_results": all_results,
        }

    def save_results(self, results: Dict, output_dir: str = "evaluation_results"):
        """Save evaluation results to files"""
        os.makedirs(output_dir, exist_ok=True)

        # Save metrics comparison
        results["metrics_comparison"].to_csv(
            f"{output_dir}/llm_metrics_comparison.csv", index=False
        )

        # Save detailed results for each approach
        for approach_name, approach_results in results["detailed_results"].items():
            df = pd.DataFrame(approach_results["detailed_results"])
            df.to_csv(f"{output_dir}/detailed_results_{approach_name}.csv", index=False)

        # Save summary
        summary = {
            "best_approach": results["best_approach"],
            "evaluation_summary": results["metrics_comparison"].to_dict("records"),
        }

        with open(f"{output_dir}/evaluation_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Results saved to {output_dir}/")


def create_medical_test_data():
    """Create sample medical test data for agentic evaluation"""
    return [
        {
            "question": "What are the first-line treatments for community-acquired pneumonia in adults?",
            "ground_truth_answer": "First-line treatments for community-acquired pneumonia include amoxicillin, azithromycin, or fluoroquinolones, depending on patient factors and severity.",
        },
        {
            "question": "What are the diagnostic criteria for Type 2 diabetes mellitus?",
            "ground_truth_answer": "Type 2 diabetes is diagnosed with fasting glucose ≥126 mg/dL, HbA1c ≥6.5%, or random glucose ≥200 mg/dL with symptoms.",
        },
        {
            "question": "What is the pathophysiology of myocardial infarction?",
            "ground_truth_answer": "Myocardial infarction occurs due to coronary artery occlusion, typically from atherosclerotic plaque rupture with thrombosis, leading to myocardial necrosis.",
        },
        {
            "question": "What are the contraindications for MRI scanning?",
            "ground_truth_answer": "MRI contraindications include certain metallic implants, pacemakers, cochlear implants, and claustrophobia in some patients.",
        },
        {
            "question": "What is the mechanism of action of ACE inhibitors?",
            "ground_truth_answer": "ACE inhibitors block angiotensin-converting enzyme, preventing conversion of angiotensin I to angiotensin II, reducing vasoconstriction and aldosterone secretion.",
        },
    ]


if __name__ == "__main__":
    # Example usage for agentic evaluation
    evaluator = AgenticLLMEvaluator()

    # Create medical test data
    test_data = create_medical_test_data()

    # Compare agentic approaches
    results = evaluator.compare_agentic_approaches(test_data, sample_size=5)

    # Save results
    evaluator.save_results(results, "agentic_evaluation_results")

    # Print summary
    print("\n🤖 AGENTIC EVALUATION SUMMARY:")
    print(f"Best approach: {results['best_approach']}")
    print(f"Total approaches tested: {len(evaluator.agentic_approaches)}")
    print("\nMetrics comparison:")
    key_metrics = [
        "approach_name",
        "avg_overall_tool_score",
        "avg_answer_quality",
        "avg_tools_per_question",
        "composite_score",
    ]
    print(results["metrics_comparison"][key_metrics].round(3))
