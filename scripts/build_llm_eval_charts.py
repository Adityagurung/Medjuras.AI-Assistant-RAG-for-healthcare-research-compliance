from pathlib import Path
OUT = Path(__file__).resolve().parent.parent / "app" / "evaluation" / "llm_eval_charts.py"
TEXT = """PLACEHOLDER"""
OUT.write_text(TEXT, encoding="utf-8")
print(OUT)
