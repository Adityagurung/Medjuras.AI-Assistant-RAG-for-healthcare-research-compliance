from __future__ import annotations

# USD per 1M tokens (input, output)
_MODEL_COST_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-oss-120b": (0.15, 0.60),
    "claude-sonnet-4-5": (3.00, 15.00),
}


def _normalize_model_key(model: str) -> str:
    return (model or "").strip().lower().replace("_", "-")


def _rates_for_model(model: str) -> tuple[float, float]:
    key = _normalize_model_key(model)
    if key in _MODEL_COST_PER_1M:
        return _MODEL_COST_PER_1M[key]
    for known, rates in _MODEL_COST_PER_1M.items():
        if known in key or key in known:
            return rates
    return (0.0, 0.0)


def estimate_llm_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    input_rate, output_rate = _rates_for_model(model)
    return round(
        (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000,
        6,
    )
