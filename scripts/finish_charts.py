from pathlib import Path
import re
P = Path(__file__).resolve().parents[1] / "app/evaluation/llm_eval_charts.py"
t = P.read_text(encoding="utf-8")
