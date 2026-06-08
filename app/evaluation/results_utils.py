"""Copy notebook outputs into results/ for versioning."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Union

from ingestion.paths import RESULTS_DIR, ensure_data_dirs

PathLike = Union[str, Path]


def results_path(name: str) -> Path:
    """Return path under results/ creating parent dirs."""
    ensure_data_dirs()
    return RESULTS_DIR / name


def results_images_dir() -> Path:
    d = RESULTS_DIR / "images"
    d.mkdir(parents=True, exist_ok=True)
    return d


def copy_to_results(src: PathLike, dest_name: str | None = None) -> Path:
    """Copy a file into results/ and return the destination path."""
    src = Path(src)
    dest = results_path(dest_name or src.name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest
