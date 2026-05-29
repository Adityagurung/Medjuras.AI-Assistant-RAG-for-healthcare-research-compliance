import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class Hit:
    #'Hit' as in the hybrid search has found some options
    id: str
    title: str
    text: str
    rrf_score: float = 0.0
    source_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def in_docker():
    return os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER") == "1"


import math
from collections import defaultdict


def rerank_rrf(results, w_rrf=1.0, w_cos=1.0, top_k=None):
    # collect ranges for min-max normalization
    rrf_vals = [d["rrf_score"] for d in results if "rrf_score" in d]
    cos_vals = [d["cos_sim_score"] for d in results if "cos_sim_score" in d]

    rrf_min, rrf_max = (min(rrf_vals), max(rrf_vals)) if rrf_vals else (0, 1)
    cos_min, cos_max = (min(cos_vals), max(cos_vals)) if cos_vals else (0, 1)

    ranked = []
    for d in results:
        if "rrf_score" in d:
            rrf_norm = (
                (d["rrf_score"] - rrf_min) / (rrf_max - rrf_min)
                if rrf_max > rrf_min
                else 0
            )
            d["final_score"] = w_rrf * rrf_norm
        elif "cos_sim_score" in d:
            cos_norm = (
                (d["cos_sim_score"] - cos_min) / (cos_max - cos_min)
                if cos_max > cos_min
                else 0
            )
            d["final_score"] = w_cos * cos_norm
        else:
            d["final_score"] = 0  # should not happen
        ranked.append(d)

    ranked.sort(key=lambda x: x["final_score"], reverse=True)
    return ranked[:top_k] if top_k else ranked
