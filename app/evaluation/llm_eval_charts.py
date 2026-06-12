"""Charts for agentic approach comparison."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ingestion.paths import RESULTS_DIR

PathLike = Union[str, Path]

COLOR_MAP = {
    "default": "#3b82f6",
    "conservative_agent": "#22c55e",
    "creative_agent": "#f97316",
}

NAME_MAP = {
    "default": "Default (temp 0.3)",
    "conservative_agent": "Conservative (temp 0.1)",
    "creative_agent": "Creative (temp 0.7)",
}

JUDGE_METRICS = [
    ("avg_tool_appropriateness", "Tool appropriateness"),
    ("avg_answer_quality", "Answer quality"),
    ("avg_information_synthesis", "Information synthesis"),
    ("avg_overall_tool_score", "Overall tool score"),
]

METRIC_COLORS = ["#6366f1", "#22c55e", "#f59e0b", "#3b82f6"]

COMPOSITE_FORMULA = (
    "Composite: 40% overall + 40% answer quality + 20% iteration efficiency (0-1 scale)"
)


def _metrics_df(comparison: Mapping[str, Any]) -> pd.DataFrame:
    mc = comparison.get("metrics_comparison")
    if mc is None:
        raise ValueError("comparison missing metrics_comparison")
    if isinstance(mc, pd.DataFrame):
        return mc.copy()
    return pd.DataFrame(list(mc))


def _approach_label(name: str) -> str:
    return NAME_MAP.get(name, str(name))


def _per_question_scores(comparison: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    detailed = comparison.get("detailed_results") or {}
    for approach, payload in detailed.items():
        if isinstance(payload, dict) and "detailed_results" in payload:
            items = payload["detailed_results"]
        elif isinstance(payload, list):
            items = payload
        else:
            continue
        for idx, item in enumerate(items):
            score = float(item.get("overall_tool_score", item.get("overall_score", 0)))
            rows.append({"approach": approach, "question_idx": idx, "score": score})
    return pd.DataFrame(rows)

def comparison_from_agentic(agentic: Mapping[str, Any]) -> dict[str, Any]:
    """Build a chart-ready comparison dict from evaluate_agentic_batch output."""
    out: dict[str, Any] = {
        "metrics_comparison": agentic.get("metrics_comparison"),
        "best_approach": agentic.get("best_approach", agentic.get("approach_name")),
    }
    if "all_approach_results" in agentic:
        out["detailed_results"] = agentic["all_approach_results"]
    return out


def best_approach_summary(comparison: Mapping[str, Any]) -> str:
    df = _metrics_df(comparison)
    best = comparison.get("best_approach", "unknown")
    if "approach_name" not in df.columns:
        return "Best agentic approach: " + str(best)
    if best in df["approach_name"].values:
        row = df.loc[df["approach_name"] == best].iloc[0]
    elif "composite_score" in df.columns:
        row = df.iloc[df["composite_score"].idxmax()]
    else:
        row = df.iloc[0]
    label = _approach_label(str(best))
    parts = [f"Best agentic approach: {label}"]
    if "composite_score" in row.index:
        parts.append(f"composite {row['composite_score']:.3f}")
    if "avg_overall_tool_score" in row.index:
        parts.append(f"overall {row['avg_overall_tool_score']:.2f}/10")
    if "avg_answer_quality" in row.index:
        parts.append(f"answer quality {row['avg_answer_quality']:.2f}/10")
    return " | ".join(parts)

def save_approach_comparison_dashboard(
    comparison: Mapping[str, Any],
    out_path: Optional[PathLike] = None,
    title: str = "Agentic approach comparison",
) -> Path:
    """Save a 2-panel dashboard: judge metrics (0-10) and composite ranking."""

    metrics = _metrics_df(comparison)
    if metrics.empty:
        raise ValueError("metrics_comparison is empty")
    if out_path is None:
        out_path = RESULTS_DIR / "images" / "agentic_approach_comparison.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best = comparison.get("best_approach")
    appr_list = metrics["approach_name"].tolist() if "approach_name" in metrics.columns else list(map(str, metrics.index))
    sns.set_theme(style="whitegrid", context="notebook")
    name_list = list(map(_approach_label, appr_list))
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor="#fafafa")
    fig.suptitle(title, fontsize=15, fontweight="bold")
    ax_metrics = axes[0]
    ax_comp = axes[1]
    n_metrics = len(JUDGE_METRICS)
    n_appr = len(appr_list)
    bar_h = 0.18
    group_gap = 0.35
    y_centers = np.arange(n_appr) * (n_metrics * bar_h + group_gap)
    for m_idx, (col, mlabel) in enumerate(JUDGE_METRICS):
        if col not in metrics.columns:
            continue
        offsets = y_centers + (m_idx - (n_metrics - 1) / 2) * bar_h
        values = metrics[col].tolist()
        ax_metrics.barh(offsets, values, height=bar_h * 0.9, color=METRIC_COLORS[m_idx % len(METRIC_COLORS)], label=mlabel)
        for bar, val in zip(ax_metrics.patches[-len(values):], values):
            ax_metrics.text(val + 0.12, bar.get_y() + bar.get_height() / 2, f"{val:.1f}", fontsize=8)
    ytick_txt = []
    for i, nm in enumerate(name_list):
        prefix = "* " if appr_list[i] == best else "  "
        ytick_txt.append(prefix + nm)
    ax_metrics.set_yticks(y_centers)
    ax_metrics.set_yticklabels(ytick_txt, fontsize=10)
    ax_metrics.set_xlim(0, 10.8)
    ax_metrics.set_xlabel("LLM judge score (0-10)", fontsize=10)
    ax_metrics.set_title("Judge metrics by approach  (* = best composite)", fontsize=12, fontweight="bold", pad=8)
    ax_metrics.legend(loc="lower right", fontsize=9, title="Metric", framealpha=0.9)
    ax_metrics.axvline(5, color="#cbd5e1", linestyle="--", linewidth=0.8, zorder=0)
    if "composite_score" in metrics.columns:
        scores = metrics["composite_score"].tolist()
        y_pos = np.arange(len(appr_list))
        bar_colors = [COLOR_MAP.get(a, "#94a3b8") for a in appr_list]
        edge_widths = [2.5 if a == best else 0.8 for a in appr_list]
        ax_comp.barh(y_pos, scores, color=bar_colors, edgecolor="#1e293b", linewidth=edge_widths, height=0.55)
        for bar, val in zip(ax_comp.patches[-len(scores):], scores):
            ax_comp.text(val + 0.008, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", fontsize=10, fontweight="bold")
        ax_comp.set_yticks(y_pos)
        ax_comp.set_yticklabels(name_list, fontsize=10)
        ax_comp.set_xlim(0, max(scores) * 1.15 + 0.05)
        ax_comp.set_xlabel("Composite score (0-1, higher is better)", fontsize=10)
        ax_comp.set_title("Composite ranking", fontsize=12, fontweight="bold", pad=8)
    summary = best_approach_summary(comparison)
    fig.text(0.5, 0.01, f"{summary}\n{COMPOSITE_FORMULA}", fontsize=9, color="#475569", style="italic")
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path.resolve()
