"""Download MedRAG Hugging Face subsets with fixed row limits."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import yaml
from datasets import load_dataset
from tqdm.auto import tqdm

from ingestion.paths import (
    MEDICAL_KEYWORDS_FILE,
    MEDRAG_RAW_DIR,
    RAW_FILES,
    ensure_data_dirs,
)

HF_REPOS = {
    "textbooks": "MedRAG/textbooks",
    "pubmed": "MedRAG/pubmed",
    "wikipedia": "MedRAG/wikipedia",
}


def load_row_limits(config_path: Optional[Path] = None) -> Dict[str, int]:
    """Load per-source row limits from config/config.yaml."""
    from ingestion.paths import CONFIG_DIR

    path = config_path or (CONFIG_DIR / "config.yaml")
    if path.exists():
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        limits = (cfg.get("data") or {}).get("row_limits") or {}
        return {
            "textbooks": int(limits.get("textbooks", 700)),
            "pubmed": int(limits.get("pubmed", 1000)),
            "wikipedia": int(limits.get("wikipedia", 300)),
        }
    return {"textbooks": 700, "pubmed": 1000, "wikipedia": 300}


def load_medical_keywords(path: Optional[Path] = None) -> List[str]:
    """Load medical filter keywords (one per line, # comments ignored)."""
    path = path or MEDICAL_KEYWORDS_FILE
    if not path.exists():
        return ["medical", "disease", "treatment", "symptom", "patient"]
    keywords: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            keywords.append(line.lower())
    return keywords


def normalize_record(row: Dict[str, Any], source_type: str) -> Dict[str, Any]:
    """Map HF row to unified schema (id, title, text, metadata)."""
    text = (row.get("contents") or row.get("content") or "").strip()
    return {
        "id": str(row.get("id", "")),
        "title": (row.get("title") or "").strip(),
        "text": text,
        "source_type": source_type,
        "source": f"MedRAG/{source_type}",
        "jurisdiction": "GLOBAL",
        "wiki_id": row.get("wiki_id"),
    }


def is_medical_row(row: Dict[str, Any], keywords: Iterable[str]) -> bool:
    """Return True if title or contents match any medical keyword."""
    hay = f"{row.get('title', '')} {row.get('contents', row.get('content', ''))}".lower()
    return any(kw in hay for kw in keywords)


def stream_rows(
    repo_id: str,
    *,
    limit: int,
    row_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> List[Dict[str, Any]]:
    """Stream a HF dataset split until `limit` rows pass optional filter."""
    ds = load_dataset(repo_id, split="train", streaming=True)
    out: List[Dict[str, Any]] = []
    for row in tqdm(ds, desc=f"Streaming {repo_id}", unit="row"):
        item = dict(row)
        if row_filter and not row_filter(item):
            continue
        out.append(item)
        if len(out) >= limit:
            break
    return out


def download_source(
    source: str,
    *,
    limit: Optional[int] = None,
    out_path: Optional[Path] = None,
    keywords: Optional[List[str]] = None,
) -> Path:
    """Download one MedRAG source and write JSON array to data/raw/medrag/."""
    ensure_data_dirs()
    limits = load_row_limits()
    limit = limit if limit is not None else limits[source]
    repo_id = HF_REPOS[source]
    out_path = out_path or RAW_FILES[source]

    row_filter: Optional[Callable[[Dict[str, Any]], bool]] = None
    if source == "wikipedia":
        kws = keywords if keywords is not None else load_medical_keywords()
        row_filter = lambda r, kw=kws: is_medical_row(r, kw)

    raw_rows = stream_rows(repo_id, limit=limit, row_filter=row_filter)
    records = [normalize_record(r, source) for r in raw_rows]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def download_all_medrag() -> Dict[str, Path]:
    """Download textbooks, pubmed, and wikipedia subsets."""
    paths = {}
    for source in ("textbooks", "pubmed", "wikipedia"):
        paths[source] = download_source(source)
    return paths


if __name__ == "__main__":
    written = download_all_medrag()
    for name, path in written.items():
        print(f"{name}: {path}")
