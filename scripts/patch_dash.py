from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
CH = ROOT / "app/evaluation/llm_eval_charts.py"

BODY_PART1 = """
    fig, axes = plt.subplots(2, 1)
"""
