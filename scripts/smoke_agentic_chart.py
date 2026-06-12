import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from evaluation.llm_eval_charts import best_approach_summary, comparison_from_agentic, save_approach_comparison_dashboard
data = json.loads((ROOT / "results/agentic_evaluation.json").read_text())
cmp = comparison_from_agentic(data)
chart = save_approach_comparison_dashboard(cmp, ROOT / "results/images/agentic_approach_comparison.png")
print(chart)
print(best_approach_summary(cmp))
