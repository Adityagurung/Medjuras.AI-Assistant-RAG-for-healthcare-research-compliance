"""Canonical data paths for MedJurisRAG."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MEDRAG_RAW_DIR = RAW_DIR / "medrag"
MEDRAG_PROCESSED_DIR = PROCESSED_DIR / "medrag"
MEDRAG_RECORDS_JSON = MEDRAG_PROCESSED_DIR / "records.json"
CHUNKS_DIR = PROCESSED_DIR / "chunks"
CHUNKS_JSONL = CHUNKS_DIR / "chunks.jsonl"
EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"
GROUND_TRUTH_JSON = EVAL_DIR / "ground_truth.json"
PDF_RAW_DIR = RAW_DIR / "pdfs"


def ensure_data_dirs() -> None:
    for d in (
        RAW_DIR,
        PROCESSED_DIR,
        MEDRAG_RAW_DIR,
        MEDRAG_PROCESSED_DIR,
        CHUNKS_DIR,
        EVAL_DIR,
        PDF_RAW_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
