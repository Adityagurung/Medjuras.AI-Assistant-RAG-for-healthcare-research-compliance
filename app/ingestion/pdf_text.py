"""Clean PDF extract text and infer document titles."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

_NL = "\n"
_NL2 = "\n\n"


def normalize_pdf_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", _NL).replace("\r", _NL)
    text = re.sub(r"\n{3,}", _NL2, text)

    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        lines = [ln.strip() for ln in block.split(_NL) if ln.strip()]
        if not lines:
            continue
        merged: list[str] = []
        for line in lines:
            if not merged:
                merged.append(line)
                continue
            prev = merged[-1]
            if re.match(r"^[\d•\-\*]", line) or prev.endswith((":", ";")):
                merged.append(line)
            elif prev.endswith("-") and line and line[0].islower():
                merged[-1] = prev[:-1] + line
            else:
                merged[-1] = prev + " " + line
        paragraphs.append(" ".join(merged))

    out = _NL2.join(paragraphs)
    return re.sub(r"[ \t]+", " ", out).strip()


def _manifest_titles(pdf_dir: Path) -> dict[str, str]:
    manifest = pdf_dir / "manifest.json"
    if not manifest.is_file():
        return {}
    try:
        rows = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, str] = {}
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and row.get("filename") and row.get("title"):
                out[str(row["filename"])] = str(row["title"]).strip()
    return out


def infer_pdf_title(
    pdf_path: Path,
    text: str,
    metadata_title: Optional[str] = None,
    pdf_dir: Optional[Path] = None,
) -> str:
    pdf_dir = pdf_dir or pdf_path.parent
    manifest_title = _manifest_titles(pdf_dir).get(pdf_path.name)
    if manifest_title:
        return manifest_title

    if metadata_title:
        meta = metadata_title.strip()
        if meta and meta != pdf_path.stem and not re.fullmatch(r"\d+", meta):
            return meta

    for line in normalize_pdf_text(text).split(_NL)[:40]:
        line = line.strip()
        if 20 <= len(line) <= 220 and not re.fullmatch(r"\d+", line):
            if line.upper().startswith("SURVEILLANCE REPORT") and len(line) < 90:
                continue
            return line

    stem = pdf_path.stem
    if re.fullmatch(r"\d+", stem):
        return f"EU regulatory PDF ({pdf_path.name})"
    return stem.replace("_", " ").replace("-", " ")
