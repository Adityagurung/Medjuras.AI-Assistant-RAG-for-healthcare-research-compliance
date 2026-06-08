"""End-to-end ingest: local MedRAG JSON -> records -> chunks JSONL."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from ingestion.chunking import chunk_records, write_chunks_jsonl
from ingestion.file_io import load_json_list
from ingestion.medrag_process import process_medrag
from ingestion.paths import CHUNKS_JSONL, RAW_FILES, ensure_data_dirs


def _ensure_raw_medrag_files() -> None:
    missing = [name for name, path in RAW_FILES.items() if not path.exists()]
    if missing:
        names = ", ".join(missing)
        raise FileNotFoundError(
            "Missing local MedRAG raw files: "
            f"{names}. Place JSON files under the medrag raw directory."
        )


def run_ingest_pipeline(
    *,
    records_path: Optional[Path] = None,
    chunks_path: Optional[Path] = None,
) -> Path:
    """Merge local MedRAG JSON, chunk, and write chunks.jsonl."""
    ensure_data_dirs()
    _ensure_raw_medrag_files()
    records_file = process_medrag(out_path=records_path)
    records = load_json_list(records_file, desc="Loading merged records")
    chunks = chunk_records(records, jurisdiction="GLOBAL")
    out = write_chunks_jsonl(chunks, path=chunks_path or CHUNKS_JSONL)
    return out


if __name__ == "__main__":
    path = run_ingest_pipeline()
    print(f"Pipeline complete: {path}")
