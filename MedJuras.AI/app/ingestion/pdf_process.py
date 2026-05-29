"""Parse EU PDFs into normalized records via Docling."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
from tqdm.auto import tqdm
from ingestion.docling_pdf import DOCLING_AVAILABLE, PYPDF_AVAILABLE, parse_pdf_docling
from ingestion.paths import PDF_RAW_DIR, ensure_data_dirs
from ingestion.pdf_text import infer_pdf_title, normalize_pdf_text
import re

def _normalize_pdf_record(item: Dict[str, Any], pdf_path: Path, idx: int) -> Dict[str, Any]:
    stem = pdf_path.stem
    text = normalize_pdf_text((item.get("text") or "").strip())
    title = infer_pdf_title(pdf_path, text, item.get("title"))
    doc_id = f"pdf_{stem}" if idx == 0 else f"pdf_{stem}_{idx}"
    return {"id": doc_id, "title": title, "text": text, "source": str(pdf_path.resolve()),
            "source_type": item.get("source_type") or "eu_pdf", "jurisdiction": "EU"}

def load_pdf_records(pdf_dir: Path | None = None) -> List[Dict[str, Any]]:
    ensure_data_dirs()
    pdf_dir = pdf_dir or PDF_RAW_DIR
    records: List[Dict[str, Any]] = []
    if not pdf_dir.exists():
        return records
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        return records
    if not DOCLING_AVAILABLE and not PYPDF_AVAILABLE:
        print("Warning: pip install pypdf or docling for PDF ingestion")
        return records
    for pdf_path in tqdm(pdfs, desc="Parsing EU PDFs", unit="pdf"):
        try:
            parts = parse_pdf_docling(pdf_path)
        except Exception as exc:
            print(f" skip {pdf_path.name}: {exc}")
            continue
        for idx, part in enumerate(parts):
            rec = _normalize_pdf_record(part, pdf_path, idx)
            if len(rec["text"]) > 100:
                records.append(rec)
    return records
