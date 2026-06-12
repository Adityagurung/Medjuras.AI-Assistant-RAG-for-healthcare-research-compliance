import base64
from pathlib import Path
import base64
from pathlib import Path
SRC = """
"""
Path(__file__).with_name("model_comparison_eval.b64").write_text(base64.b64encode(SRC.encode()).decode())
