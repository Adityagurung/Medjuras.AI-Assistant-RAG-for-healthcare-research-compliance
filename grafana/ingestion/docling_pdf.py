"""PDF parsing: Docling with OCR off, pypdf fallback."""
from __future__ import annotations

import os
from pathlib import Path

from ingestion.pdf_text import infer_pdf_title, normalize_pdf_text

DOCLING_AVAILABLE = False
PYPDF_AVAILABLE = False
_converter = None

def _pdf_parser_mode():
    mode=(os.environ.get("PDF_PARSER") or "pypdf").strip().lower()
    return mode if mode in ("pypdf","docling","auto") else "pypdf"

def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")

def _get_converter():
    global _converter
    if _converter is not None:
        return _converter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions, PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions(
        do_ocr=_env_bool("DOCLING_DO_OCR", False),
        do_table_structure=_env_bool("DOCLING_DO_TABLES", False),
        accelerator_options=AcceleratorOptions(num_threads=2, device=AcceleratorDevice.CPU),
    )
    _converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)},
    )
    return _converter

try:
    import docling  # noqa: F401
    DOCLING_AVAILABLE = True
except Exception:
    pass

try:
    import pypdf  # noqa: F401
    PYPDF_AVAILABLE = True
except Exception:
    pass

def parse_pdf_pypdf(path: Path) -> list[dict]:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    parts = [(page.extract_text() or "").strip() for page in reader.pages]
    text = "\n\n".join(p for p in parts if p).strip()
    if len(text) < 50:
        raise ValueError(f"pypdf extracted too little text from {path.name}")
    return [{
        "title": infer_pdf_title(path, text, None),
        "text": normalize_pdf_text(text),
        "source_type": "pypdf",
        "source": str(path.resolve()),
        "jurisdiction": "EU",
    }]

def parse_pdf_docling(path) -> list[dict]:
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    mode = _pdf_parser_mode()
    if mode == "pypdf" and PYPDF_AVAILABLE:
        return parse_pdf_pypdf(pdf_path)

    if mode in ("docling", "auto") and DOCLING_AVAILABLE:
        try:
            result = _get_converter().convert(str(pdf_path))
            md = normalize_pdf_text((result.document.export_to_markdown() or "").strip())
            if len(md) >= 50:
                return [{
                    "title": infer_pdf_title(pdf_path, md),
                    "text": md,
                    "source_type": "docling_pdf",
                    "source": str(pdf_path.resolve()),
                    "jurisdiction": "EU",
                }]
        except Exception as exc:
            print(f" docling failed for {pdf_path.name}: {exc}")

    if PYPDF_AVAILABLE:
        return parse_pdf_pypdf(pdf_path)

    return [{
        "title": pdf_path.stem,
        "text": f"[Install docling or pypdf] Placeholder for {pdf_path.name}",
        "source_type": "pdf",
        "source": str(pdf_path),
    }]