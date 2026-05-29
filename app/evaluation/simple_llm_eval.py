import json
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


class SimpleLLMEvaluator:
    """
    Simple LLM evaluator that tests multiple approaches without requiring
    external services (ES/Qdrant). Tests pure LLM performance.
    """

    def __init__(self, embedding_model_name: str = "multi-qa-MiniLM-L6-cos-v1"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.client = OpenAI()

        # Multiple LLM approaches to compare
        self.approaches = {
            "gpt-4o-mini": {"model": "gpt-4o-mini", "temperature": 0.3},
            "gpt-4o": {"model": "gpt-4o", "temperature": 0.3},
            "gpt-3.5-turbo": {"model": "gpt-3.5-turbo", "temperature": 0.3},
            "conservative": {"model": "gpt-4o-mini", "temperature": 0.1},
            "creative": {"model": "gpt-4o-mini", "temperature": 0.7},
        }

        # LLM-as-a-Judge prompt
        self.judge_prompt_template = """
You are an expert evaluator for medical AI responses.
Rate the medical answer on these aspects (0-10 scale):

Question: {question}
Ground Truth: {ground_truth}
LLM Answer: {llm_answer}

Evaluate:
1. ACCURACY: How medically accurate is the answer?
2. COMPLETENESS: Does it cover the key points?
3. CLARITY: Is it clear and well-explained?

Provide your evaluation in JSON format:
{{
  "accuracy": <0-10>,
  "completeness": <0-10>,
  "clarity": <0-10>,
  "overall_score": <0-10>,
  "explanation": "Brief explanation"
}}
""".strip()

    def generate_medical_answer(self, question: str, approach_config: Dict) -> str:
        """Generate medical answer using specified LLM approach"""
        prompt = f"""
You are a medical AI assistant. Answer the following medical question accurately and concisely.

Question: {question}

Provide a clear, evidence-based medical answer:
""".strip()

        try:
            response = self.client.chat.completions.create(
                model=approach_config["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=approach_config.get("temperature", 0.3),
                max_tokens=400,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating answer: {e}")
            return ""

    def compute_cosine_similarity(self, answer1: str, answer2: str) -> float:
        """Compute cosine similarity between two answers"""
        try:
            v1 = self.embedding_model.encode(answer1)
            v2 = self.embedding_model.encode(answer2)
            return float(np.dot(v1, v2))
        except Exception as e:
            print(f"Error computing similarity: {e}")
            return 0.0

    def evaluate_with_judge(
        self, question: str, ground_truth: str, llm_answer: str
    ) -> Dict:
        """Evaluate answer using LLM-as-a-Judge"""
        prompt = self.judge_prompt_template.format(
            question=question, ground_truth=ground_truth, llm_answer=llm_answer
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )

            evaluation_text = response.choices[0].message.content
            evaluation = json.loads(evaluation_text)
            return evaluation
        except Exception as e:
            print(f"Error in judge evaluation: {e}")
            return {
                "accuracy": 0,
                "completeness": 0,
                "clarity": 0,
                "overall_score": 0,
                "explanation": f"Error: {e}",
            }

    def evaluate_approach(
        self, test_data: List[Dict], approach_name: str, approach_config: Dict
    ) -> Dict:
        """Evaluate a single LLM approach"""
        print(f"\nEvaluating approach: {approach_name}")

        results = []
        cosine_scores = []
        judge_scores = []

        for item in tqdm(test_data, desc=f"Testing {approach_name}"):
            # Generate answer
            llm_answer = self.generate_medical_answer(item["question"], approach_config)

            # Compute cosine similarity
            cosine_score = self.compute_cosine_similarity(
                llm_answer, item["ground_truth_answer"]
            )
            cosine_scores.append(cosine_score)

            # LLM-as-a-Judge evaluation
            judge_eval = self.evaluate_with_judge(
                item["question"], item["ground_truth_answer"], llm_answer
            )
            judge_scores.append(judge_eval)

            results.append(
                {
                    "question": item["question"],
                    "llm_answer": llm_answer,
                    "ground_truth_answer": item["ground_truth_answer"],
                    "cosine_similarity": cosine_score,
                    "accuracy": judge_eval.get("accuracy", 0),
                    "completeness": judge_eval.get("completeness", 0),
                    "clarity": judge_eval.get("clarity", 0),
                    "overall_judge_score": judge_eval.get("overall_score", 0),
                    "judge_explanation": judge_eval.get("explanation", ""),
                }
            )

        # Calculate metrics
        metrics = {
            "approach_name": approach_name,
            "total_samples": len(test_data),
            "cosine_mean": np.mean(cosine_scores),
            "cosine_std": np.std(cosine_scores),
            "avg_accuracy": np.mean([s.get("accuracy", 0) for s in judge_scores]),
            "avg_completeness": np.mean(
                [s.get("completeness", 0) for s in judge_scores]
            ),
            "avg_clarity": np.mean([s.get("clarity", 0) for s in judge_scores]),
            "avg_overall_judge": np.mean(
                [s.get("overall_score", 0) for s in judge_scores]
            ),
        }

        return {"metrics": metrics, "detailed_results": results}

    def compare_approaches(self, test_data: List[Dict]) -> Dict:
        """Compare all LLM approaches and select the best one"""
        print("Starting LLM approach comparison...")

        all_results = {}
        all_metrics = []

        for approach_name, approach_config in self.approaches.items():
            result = self.evaluate_approach(test_data, approach_name, approach_config)
            all_results[approach_name] = result
            all_metrics.append(result["metrics"])

        # Convert to DataFrame for easy comparison
        metrics_df = pd.DataFrame(all_metrics)

        # Select best approach based on composite score
        # Weight: 50% cosine similarity + 50% judge overall score
        metrics_df["composite_score"] = 0.5 * metrics_df["cosine_mean"] + 0.5 * (
            metrics_df["avg_overall_judge"] / 10
        )  # Normalize to 0-1

        best_approach = metrics_df.loc[
            metrics_df["composite_score"].idxmax(), "approach_name"
        ]

        print(f"\n🏆 Best performing approach: {best_approach}")
        print(f"Composite score: {metrics_df['composite_score'].max():.3f}")

        return {
            "best_approach": best_approach,
            "metrics_comparison": metrics_df,
            "detailed_results": all_results,
        }

    def save_results(
        self, results: Dict, output_dir: str = "simple_evaluation_results"
    ):
        """Save evaluation results to files"""
        os.makedirs(output_dir, exist_ok=True)

        # Save metrics comparison
        results["metrics_comparison"].to_csv(
            f"{output_dir}/simple_metrics_comparison.csv", index=False
        )

        # Save detailed results for each approach
        for approach_name, approach_results in results["detailed_results"].items():
            df = pd.DataFrame(approach_results["detailed_results"])
            df.to_csv(
                f"{output_dir}/simple_detailed_results_{approach_name}.csv", index=False
            )

        # Save summary
        summary = {
            "best_approach": results["best_approach"],
            "evaluation_summary": results["metrics_comparison"].to_dict("records"),
        }

        with open(f"{output_dir}/simple_evaluation_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Results saved to {output_dir}/")


def create_medical_test_data():
    """Create sample medical test data"""
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
    # Simple evaluation that works without external services
    evaluator = SimpleLLMEvaluator()

    # Create medical test data
    test_data = create_medical_test_data()

    # Compare approaches
    results = evaluator.compare_approaches(test_data)

    # Save results
    evaluator.save_results(results)

    # Print summary
    print("\n📊 SIMPLE EVALUATION SUMMARY:")
    print(f"Best approach: {results['best_approach']}")
    print(f"Total approaches tested: {len(evaluator.approaches)}")
    print("\nMetrics comparison:")
    key_metrics = [
        "approach_name",
        "cosine_mean",
        "avg_overall_judge",
        "composite_score",
    ]
    print(results["metrics_comparison"][key_metrics].round(3))
