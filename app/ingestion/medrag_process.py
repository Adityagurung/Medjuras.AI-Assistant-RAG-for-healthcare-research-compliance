"""Normalize raw MedRAG JSON downloads to processed records.json."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from tqdm.auto import tqdm

from ingestion.paths import MEDRAG_PROCESSED_DIR, MEDRAG_RAW_DIR, MEDRAG_RECORDS_JSON, ensure_data_dirs
from ingestion.pdf_process import load_pdf_records


def _normalize_item(item: Dict[str, Any], idx: int, source: str) -> Dict[str, Any]:
    text = (item.get("text") or item.get("content") or "").strip()
    title = (item.get("title") or item.get("name") or "").strip()
    doc_id = str(item.get("id") or item.get("wiki_id") or idx)
    return {
        "id": doc_id,
        "title": title,
        "text": text,
        "source": source,
        "source_type": item.get("source_type") or source,
        "jurisdiction": item.get("jurisdiction") or "EU",
        "wiki_id": item.get("wiki_id"),
    }


def load_raw_medrag_dir(raw_dir: Path | None = None) -> List[Dict[str, Any]]:
    raw_dir = raw_dir or MEDRAG_RAW_DIR
    records: List[Dict[str, Any]] = []
    if not raw_dir.exists():
        return records
    for path in tqdm(sorted(raw_dir.glob("*.json")), desc="Loading raw MedRAG files", unit="file"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "records" in data:
            data = data["records"]
        if not isinstance(data, list):
            continue
        source = path.stem.replace("medical_", "").replace("_seed", "") or "medrag"
        for i, item in tqdm(enumerate(data), total=len(data), desc=f"Normalizing {path.name}", unit="doc", leave=False):
            if isinstance(item, dict):
                records.append(_normalize_item(item, i, source))
    return records


def process_medrag(raw_dir: Path | None = None, out_path: Path | None = None, *, include_pdfs: bool = True) -> Path:
    ensure_data_dirs()
    records = load_raw_medrag_dir(raw_dir)
    if include_pdfs:
        pdf_records = load_pdf_records()
        if pdf_records:
            print(f"Added {len(pdf_records)} PDF-derived records")
            records.extend(pdf_records)
    out = out_path or MEDRAG_RECORDS_JSON
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


if __name__ == "__main__":
    p = process_medrag()
    print(f"Wrote {len(json.loads(p.read_text()))} records to {p}")
