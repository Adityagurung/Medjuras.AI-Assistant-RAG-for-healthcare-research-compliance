"""End-to-end ingest: HF download → records → chunks JSONL."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from ingestion.chunking import chunk_records, write_chunks_jsonl
from ingestion.file_io import load_json_list
from ingestion.hf_download import download_all_medrag
from ingestion.medrag_process import process_medrag
from ingestion.paths import CHUNKS_JSONL, MEDRAG_RECORDS_JSON, ensure_data_dirs


def run_ingest_pipeline(
    *,
    records_path: Optional[Path] = None,
    chunks_path: Optional[Path] = None,
) -> Path:
    """
    Download MedRAG subsets, merge records, chunk, and write chunks.jsonl.

    Returns:
        Path to chunks.jsonl
    """
    ensure_data_dirs()
    download_all_medrag()
    records_file = process_medrag(out_path=records_path)
    records = load_json_list(records_file, desc="Loading merged records")
    chunks = chunk_records(records, jurisdiction="GLOBAL")
    out = write_chunks_jsonl(chunks, path=chunks_path or CHUNKS_JSONL)
    return out


if __name__ == "__main__":
    path = run_ingest_pipeline()
    print(f"Pipeline complete: {path}")
