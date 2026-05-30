"""Normalize raw MedRAG JSON downloads to processed records.json."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm.auto import tqdm

from ingestion.paths import MEDRAG_PROCESSED_DIR, MEDRAG_RAW_DIR, MEDRAG_RECORDS_JSON, RAW_FILES, ensure_data_dirs


def load_raw_json(path: Path) -> List[Dict[str, Any]]:
    """Load a JSON array written by hf_download."""
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def merge_raw_sources(raw_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load and concatenate all three MedRAG raw files."""
    raw_dir = raw_dir or MEDRAG_RAW_DIR
    records: List[Dict[str, Any]] = []
    for source, path in RAW_FILES.items():
        file_path = raw_dir / path.name if (raw_dir / path.name).exists() else path
        rows = load_raw_json(file_path)
        for row in tqdm(rows, desc=f"Loading {source}", unit="doc", leave=False):
            row.setdefault("source_type", source)
            row.setdefault("jurisdiction", "GLOBAL")
            records.append(row)
    return records


def process_medrag(
    raw_dir: Optional[Path] = None,
    out_path: Optional[Path] = None,
) -> Path:
    """Merge raw HF JSON files into data/processed/medrag/records.json."""
    ensure_data_dirs()
    records = merge_raw_sources(raw_dir)
    out = out_path or MEDRAG_RECORDS_JSON
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


if __name__ == "__main__":
    p = process_medrag()
    n = len(json.loads(p.read_text(encoding="utf-8")))
    print(f"Wrote {n} records to {p}")
