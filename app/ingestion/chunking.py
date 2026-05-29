"""Text chunking for MedJurisRAG (AIRRA + LangChain)."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm.auto import tqdm

from ingestion.paths import CHUNKS_JSONL, ensure_data_dirs
from ingestion.pdf_text import normalize_pdf_text

_PDF_SEPARATORS = ["\n\n", ". ", " ", ""]
_PDF_SOURCE_TYPES = frozenset({"pypdf", "docling_pdf", "eu_pdf"})


def chunk_text(text: str, chunk_size: int = 5000, overlap: int = 500) -> List[str]:
    """Sliding-window chunker from AIRRA rag_local."""
    text = " ".join((text or "").split())
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_records(
    records: Iterable[Dict[str, Any]],
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    source_type: str = "medrag",
    jurisdiction: str = "EU",
) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pdf_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=list(_PDF_SEPARATORS),
    )
    out: List[Dict[str, Any]] = []
    record_list = list(records)
    for rec in tqdm(record_list, desc="Chunking records", unit="doc"):
        doc_id = str(rec.get("id") or rec.get("doc_id") or "")
        title = (rec.get("title") or "").strip()
        body = (rec.get("text") or rec.get("content") or "").strip()
        if not body:
            continue
        st = rec.get("source_type") or source_type
        if st in _PDF_SOURCE_TYPES:
            body = normalize_pdf_text(body)
            parts = pdf_splitter.split_text(body)
        else:
            parts = splitter.split_text(body)
        for idx, part in enumerate(parts):
            chunk_id = hashlib.sha1(f"{doc_id}:{idx}:{part[:80]}".encode()).hexdigest()[:16]
            out.append(
                {
                    "id": f"{doc_id}:{idx}" if doc_id else chunk_id,
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "title": title,
                    "text": part,
                    "chunk_index": idx,
                    "source_type": rec.get("source_type") or source_type,
                    "source": rec.get("source") or "medrag",
                    "jurisdiction": rec.get("jurisdiction") or jurisdiction,
                }
            )
    return out


def write_chunks_jsonl(chunks: List[Dict[str, Any]], path: Optional[Path] = None) -> Path:
    from pathlib import Path as P

    ensure_data_dirs()
    out_path = P(path) if path else CHUNKS_JSONL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in tqdm(chunks, desc="Writing chunks JSONL", unit="chunk"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out_path
