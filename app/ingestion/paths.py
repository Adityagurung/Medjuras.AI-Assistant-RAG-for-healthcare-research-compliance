"""Canonical data and results paths for MedJuras.AI."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MEDRAG_RAW_DIR = RAW_DIR / "medrag"
MEDRAG_PROCESSED_DIR = PROCESSED_DIR / "medrag"
MEDRAG_RECORDS_JSON = MEDRAG_PROCESSED_DIR / "records.json"
CHUNKS_DIR = PROCESSED_DIR / "chunks"
CHUNKS_JSONL = CHUNKS_DIR / "chunks.jsonl"
EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"
GROUND_TRUTH_JSON = EVAL_DIR / "ground_truth.json"
RESULTS_DIR = PROJECT_ROOT / "results"
MEDICAL_KEYWORDS_FILE = CONFIG_DIR / "medical_keywords.txt"

RAW_FILES = {
    "textbooks": MEDRAG_RAW_DIR / "textbooks.json",
    "pubmed": MEDRAG_RAW_DIR / "pubmed.json",
    "wikipedia": MEDRAG_RAW_DIR / "wikipedia.json",
}


def ensure_data_dirs() -> None:
    """Create data, processed, evaluation, and results directories."""
    for d in (
        RAW_DIR,
        PROCESSED_DIR,
        MEDRAG_RAW_DIR,
        MEDRAG_PROCESSED_DIR,
        CHUNKS_DIR,
        EVAL_DIR,
        RESULTS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
