from pathlib import Path
from typing import Any, Dict, List, Optional


def ground_truth_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data/evaluation/ground_truth.json"


def load_ground_truth(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load ground-truth JSON with a read progress bar."""
    from ingestion.file_io import load_json_list

    p = path or ground_truth_path()
    return load_json_list(p, desc="Loading ground truth")
