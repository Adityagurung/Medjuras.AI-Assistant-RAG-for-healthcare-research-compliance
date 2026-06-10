# Save bar-chart PNGs for notebooks 3-5 retrieval evaluation.
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Union

import matplotlib.pyplot as plt

from evaluation.retrieval_plot import (
    METRIC_FULL_NAME,
    RETRIEVER_INFO,
    _metric_keys,
    save_plot,
)
from ingestion.paths import RESULTS_DIR

PathLike = Union[str, Path]

METHOD_PNG = {
    'Elasticsearch': 'es_bm25_metrics.png',
    'Qdrant': 'qdrant_dense_metrics.png',
    'Hybrid': 'hybrid_rrf_metrics.png',
}


def save_single_method_plot(method, metrics, top_k, out_path=None):
    info=RETRIEVER_INFO.get(method, {'title': method, 'subtitle': '', 'accent': '#2563eb'})
    metric_names = _metric_keys(top_k)
    metric_colors = ['#3b82f6', '#f97316', '#22c55e']
    vals = [float(metrics[m]) for m in metric_names]

    fig, ax=plt.subplots(figsize=(9, 6), facecolor='#fafafa')
    bars=ax.bar(
        range(len(metric_names)),
        vals,
        width=0.6,
        color=metric_colors,
        edgecolor='white',
        linewidth=0.8,
    )
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            format(val, '.3f'),
            ha='center',
            va='bottom',
            fontsize=9,
            color='#374151',
        )

    title=info['title']
    subtitle=info['subtitle']
    heading=(title + ' - ' + subtitle).rstrip(' -')
    ax.set_title(
        heading + chr(10) + 'Retrieval quality (top ' + str(top_k) + ')',
        fontsize=13,
        fontweight='bold',
        color=info.get('accent', '#1e293b'),
        pad=12,
    )
    ax.set_ylabel('Score (0 to 1)', fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(
        [m + chr(10) + METRIC_FULL_NAME[m.split('@')[0]] for m in metric_names],
        fontsize=9,
    )
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    fig.tight_layout()

    if out_path is None:
        fname=METHOD_PNG.get(method, method.lower() + '_metrics.png')
        out_path=RESULTS_DIR / 'images' / fname
    out_path=Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path.resolve()


def save_comparison_from_result_files(results_dir, top_k):
    # Build the grouped comparison chart from the saved metric JSON files.
    results_dir=Path(results_dir)
    mapping={
        'Elasticsearch': 'es_bm25_metrics.json',
        'Qdrant': 'qdrant_dense_metrics.json',
        'Hybrid': 'hybrid_rrf_metrics.json',
    }
    metrics_by_method={}
    for method, fname in mapping.items():
        path=results_dir / fname
        if path.exists():
            metrics_by_method[method]=json.loads(path.read_text(encoding='utf-8'))
    if len(metrics_by_method) < 2:
        raise FileNotFoundError('Need at least two metric JSON files in results/')
    img_dir=results_dir / 'images'
    return save_plot(metrics_by_method, top_k, img_dir).resolve()
