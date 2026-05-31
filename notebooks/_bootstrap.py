"""Add project root and app/ to sys.path for Jupyter kernels."""
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
load_dotenv(ROOT / ".env")
for p in (str(ROOT), str(APP)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ingestion.file_io import configure_notebook_progress  # noqa: E402

configure_notebook_progress()
