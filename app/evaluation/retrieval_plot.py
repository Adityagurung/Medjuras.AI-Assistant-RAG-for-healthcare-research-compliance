"""Charts for notebook 06: retrieval eval with retriever explanations."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Dict, Mapping

import matplotlib.pyplot as plt

RETRIEVER_INFO = {
    "Hybrid": {
        "title": "Hybrid",
        "subtitle": "Dense + sparse fusion",
        "accent": "#2563eb",
    },
    "Qdrant": {
        "title": "Qdrant",
        "subtitle": "Dense vectors only",
        "accent": "#059669",
    },
    "Elasticsearch": {
        "title": "Elasticsearch",
        "subtitle": "Sparse / keyword (BM25)",
        "accent": "#d97706",
    },
}

METRIC_FULL_NAME = {
    "Hit": "Hit Rate",
    "MRR": "Mean Reciprocal Rank",
    "nDCG": "Normalized Discounted Cumulative Gain",
}

# Plain-language descriptions (legend / HTML)
METRIC_HELP = {
    "Hit": "Found the right chunk somewhere in the top results",
    "MRR": "How high the first correct chunk was ranked",
    "nDCG": "Rewards putting the right chunk nearer the top",
}


def _metric_keys(top_k: int) -> list[str]:
    return [f"Hit@{top_k}", f"MRR@{top_k}", f"nDCG@{top_k}"]


def _chart_title(top_k: int) -> str:
    return (
        f"Same ground-truth questions — three retrieval approaches (top {top_k})\n"
        "Hit Rate · Mean Reciprocal Rank · "
        "Normalized Discounted Cumulative Gain"
    )


def _metric_legend_label(help_key: str, top_k: int) -> str:
    abbrev = f"{help_key}@{top_k}"
    return f"{abbrev} ({METRIC_FULL_NAME[help_key]}): {METRIC_HELP[help_key]}"


def save_plot(
    metrics_by_method: Mapping[str, Dict[str, float]],
    top_k: int,
    out_dir: Path,
) -> Path:
    """Grouped bar chart; retriever names on x-axis, simple metric legend."""
    methods = [m for m in RETRIEVER_INFO if m in metrics_by_method]
    if not methods:
        raise ValueError("No metrics to plot")

    metric_names = _metric_keys(top_k)
    metric_colors = ["#3b82f6", "#f97316", "#22c55e"]
    x = list(range(len(methods)))
    width = 0.22
    offsets = [(-1.0 + i) * width for i in range(3)]

    fig, ax = plt.subplots(figsize=(12, 7), facecolor="#fafafa")

    for i, metric_name in enumerate(metric_names):
        vals = [metrics_by_method[m][metric_name] for m in methods]
        help_key = metric_name.split("@")[0]
        bars = ax.bar(
            [j + offsets[i] for j in x],
            vals,
            width=width,
            label=_metric_legend_label(help_key, top_k),
            color=metric_colors[i],
            edgecolor="white",
            linewidth=0.8,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#374151",
            )

    ax.set_title(
        _chart_title(top_k),
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel("Score (0–1)", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{RETRIEVER_INFO[m]['title']}\n{RETRIEVER_INFO[m]['subtitle']}" for m in methods],
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        fontsize=9,
        framealpha=0.95,
        title="What each metric means",
    )

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "retrieval_eval_comparison.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def save_plot_html(
    metrics_by_method: Mapping[str, Dict[str, float]],
    top_k: int,
    out_dir: Path,
) -> tuple[Path, str]:
    """CSS bar chart with full retriever + metric explanations."""
    methods = [m for m in RETRIEVER_INFO if m in metrics_by_method]
    metric_names = _metric_keys(top_k)
    metric_css = {
        metric_names[0]: "metric-hit",
        metric_names[1]: "metric-mrr",
        metric_names[3]: "metric-ndcg",
    }

    cards_html: list[str] = []
    for m in methods:
        info = RETRIEVER_INFO[m]
        bars: list[str] = []
        for mn in metric_names:
            val = metrics_by_method[m][mn]
            pct = max(0, min(100, val * 100))
            help_key = mn.split("@")[0]
            full_name = METRIC_FULL_NAME[help_key]
            bars.append(
                f"""
                <div class="metric-row">
                  <div class="metric-label" title="{html.escape(METRIC_HELP[help_key])}">{mn}<br/><span class="metric-full">{html.escape(full_name)}</span></div>
                  <div class="bar-track">
                    <div class="bar-fill {metric_css[mn]}" style="width:{pct:.1f}%"></div>
                  </div>
                  <div class="metric-value">{val:.3f}</div>
                </div>"""
            )
        cards_html.append(
            f"""
            <article class="retriever-card" style="--accent:{info['accent']}">
              <header>
                <h2>{html.escape(info['title'])}</h2>
                <p class="subtitle">{html.escape(info['subtitle'])}</p>
              </header>
              <div class="bars">{''.join(bars)}</div>
            </article>"""
        )

    legend = "".join(
        f'<li><span class="swatch {metric_css[mn]}"></span>'
        f"<strong>{html.escape(_metric_legend_label(mn.split('@')[0], top_k))}</strong></li>"
        for mn in metric_names
    )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{html.escape(_chart_title(top_k).replace(chr(10), " "))}</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --card: #ffffff;
      --text: #1e293b;
      --muted: #64748b;
      --border: #e2e8f0;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 2rem;
      line-height: 1.5;
    }}
    h1 {{ margin: 0 0 0.25rem; font-size: 1.75rem; }}
    .lead {{ color: var(--muted); margin: 0 0 1.5rem; max-width: 52rem; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 1.25rem;
      margin-bottom: 2rem;
    }}
    .retriever-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-top: 4px solid var(--accent);
      border-radius: 12px;
      padding: 1.25rem;
      box-shadow: 0 1px 3px rgba(0,0,0,.06);
    }}
    .retriever-card h2 {{ margin: 0; color: var(--accent); font-size: 1.25rem; }}
    .subtitle {{ margin: 0.25rem 0 1rem; color: var(--muted); font-size: 0.95rem; }}
    .metric-row {{
      display: grid;
      grid-template-columns: 4.5rem 1fr 3rem;
      align-items: center;
      gap: 0.5rem;
      margin-bottom: 0.5rem;
    }}
    .metric-label {{ font-size: 0.8rem; font-weight: 600; line-height: 1.3; }}
    .metric-full {{ font-weight: 400; color: var(--muted); font-size: 0.72rem; }}
    .bar-track {{
      height: 10px;
      background: #e2e8f0;
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar-fill {{ height: 100%; border-radius: 999px; }}
    .metric-hit {{ background: #3b82f6; }}
    .metric-mrr {{ background: #f97316; }}
    .metric-ndcg {{ background: #ef4444; }}
    .metric-value {{ font-size: 0.8rem; font-variant-numeric: tabular-nums; text-align: right; }}
    .legend {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1rem 1.25rem;
      max-width: 52rem;
    }}
    .legend h3 {{ margin: 0 0 0.75rem; font-size: 1rem; }}
    .legend ul {{ margin: 0; padding: 0; list-style: none; }}
    .legend li {{ margin-bottom: 0.5rem; font-size: 0.85rem; display: flex; gap: 0.5rem; }}
    .swatch {{ width: 12px; height: 12px; border-radius: 2px; flex-shrink: 0; margin-top: 0.2rem; }}
  </style>
</head>
<body>
  <h1>{html.escape(_chart_title(top_k).replace(chr(10), " — "))}</h1>
  <p class="lead">Each card is one retrieval approach on the same questions. Bar length = score from 0 to 1.</p>
  <div class="grid">{''.join(cards_html)}</div>
  <section class="legend">
    <h3>What each metric means</h3>
    <ul>{legend}</ul>
  </section>
</body>
</html>"""

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "retrieval_eval_comparison.html"
    out.write_text(doc, encoding="utf-8")
    return out, doc
