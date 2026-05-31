"""File loading helpers with tqdm progress (notebooks and pipelines)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm.auto import tqdm


def configure_notebook_progress() -> None:
    """Use Jupyter widget progress bars when ipywidgets is available."""
    try:
        from IPython import get_ipython

        if get_ipython() is None:
            return
        import ipywidgets  # noqa: F401
        from tqdm.autonotebook import tqdm as notebook_tqdm

        import tqdm.auto as tqdm_auto

        tqdm_auto.tqdm = notebook_tqdm
    except ImportError:
        pass


def read_text_with_progress(path: Path, *, desc: Optional[str] = None) -> str:
    """Read a text file with a byte-level progress bar."""
    desc = desc or f"Reading {path.name}"
    size = path.stat().st_size
    if size == 0:
        return ""
    parts: list[bytes] = []
    block = 256 * 1024
    with path.open("rb") as f, tqdm(total=size, desc=desc, unit="B", unit_scale=True) as bar:
        while True:
            chunk = f.read(block)
            if not chunk:
                break
            parts.append(chunk)
            bar.update(len(chunk))
    return b"".join(parts).decode("utf-8")


def load_json(path: Path, *, desc: Optional[str] = None) -> Any:
    """Load JSON from disk with read progress."""
    text = read_text_with_progress(path, desc=desc or f"Reading {path.name}")
    return json.loads(text) if text else None


def load_json_list(path: Path, *, desc: Optional[str] = None) -> List[Any]:
    """Load a JSON array file; raises if the root value is not a list."""
    data = load_json(path, desc=desc)
    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return data


def load_jsonl(path: Path, *, desc: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load JSONL line-by-line with a line progress bar."""
    desc = desc or f"Loading {path.name}"
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc=desc, unit="line"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
