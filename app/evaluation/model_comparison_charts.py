"""2x x cosine-similarity dashboard comparing LLM models on the same RAG questions."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from evaluation.results_utils import results_images_dir

PathLike = Union[str, Path]

MODEL_COLORS = {
    "gpt-5.1": "#7ec8e3",
    "ollama": "#8fd19e",
}


def _color_for(model: str) -> str:
    key = model.lower()
    if key in MODEL_COLORS:
        return MODEL_COLORS[key]
    for k, v in MODEL_COLORS.items():
        if k in key or key in k:
            return v
    return "#94a3b8"


def save_model_comparison_dashboard(
    results_by_model: Mapping[str, Sequence[float]],
    *,
    out_path: Optional[PathLike] = None,
    title: str = "Boxplot grouped by model",
) -> Path:
    if not results_by_model:
        raise ValueError("results_by_model is empty")

    models = list(results_by_model.keys())
    rows = []
    for model, scores in results_by_model.items():
        for idx, score in enumerate(scores):
            rows.append({"model": model, "question_idx": idx, "cosine_similarity": float(score)})
    df = pd.DataFrame(rows)

    if out_path is None:
        out_path = results_images_dir() / "model_comparison_dashboard.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#fafafa")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    palette = {m: _color_for(m) for m in models}
    means = df.groupby("model")["cosine_similarity"].mean().reindex(models)

    ax0 = axes[0, 0]
    bars = ax0.bar(
        range(len(models)),
        means.values,
        color=[palette[m] for m in models],
        edgecolor="white",
        linewidth=0.8,
    )
    for bar, val in zip(bars, means.values):
        ax0.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax0.set_title("Average Cosine Similarity by Model", fontsize=11, fontweight="bold")
    ax0.set_ylabel("Cosine Similarity")
    ax0.set_xticks(range(len(models)))
    ax0.set_xticklabels(models, rotation=15, ha="right")
    ax0.set_ylim(0, max(0.65, float(means.max()) + 0.08))
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=palette[m]) for m in models]
    ax0.legend(legend_handles, models, loc="upper right", fontsize=8, framealpha=0.9)

    ax1 = axes[0, 1]
    sns.boxplot(
        data=df,
        x="model",
        y="cosine_similarity",
        hue="model",
        palette=palette,
        ax=ax1,
        legend=False,
        width=0.55,
    )
    ax1.set_title("Boxplot grouped by model", fontsize=11, fontweight="bold")
    ax1.set_xlabel("")
    ax1.set_ylabel("Cosine Similarity")
    ax1.tick_params(axis="x", rotation=15)

    ax2 = axes[1, 0]
    for model in models:
        subset = df.loc[df["model"] == model, "cosine_similarity"]
        ax2.hist(
            subset,
            bins=12,
            alpha=0.55,
            label=model,
            color=palette[model],
            edgecolor="white",
        )
    ax2.set_title("Distribution of All Cosine Similarities", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Cosine Similarity")
    ax2.set_ylabel("Frequency")
    ax2.legend(fontsize=8)

    ax3 = axes[1, 1]
    for model in models:
        subset = df.loc[df["model"] == model].sort_values("question_idx")
        ax3.plot(
            subset["question_idx"],
            subset["cosine_similarity"],
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=model,
            color=palette[model],
            alpha=0.85,
        )
    ax3.set_title("Cosine Similarity Across Questions by Model", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Question index")
    ax3.set_ylabel("Cosine Similarity")
    ax3.legend(fontsize=8, loc="best")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path.resolve()
