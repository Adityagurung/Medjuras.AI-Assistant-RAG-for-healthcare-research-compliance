"""Launch the MedJuras Streamlit UI (run from repo root)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
env = {**dict(**{"PYTHONPATH": str(APP)}), **dict(__import__("os").environ)}
subprocess.run(
    [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP / "main.py"),
        "--server.port=8501",
    ],
    cwd=str(ROOT),
    env={**__import__("os").environ, "PYTHONPATH": str(APP)},
    check=True,
)
