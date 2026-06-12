def save_approach_comparison_dashboard(comparison, out_path=None, title="Agentic approach comparison"):
    metrics = _metrics_df(comparison)
    if metrics.empty:
        raise ValueError("metrics_comparison is empty")
    if out_path is None:
        out_path = RESULTS_DIR / "images" / "agentic_approach_comparison.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ax2 = axes[1, 0]
    if not per_q.empty:
        for approach in approaches:
            subset = per_q.loc[per_q["approach"] == approach, "score"]
            if len(subset):
                ax2.hist(subset, bins=10, alpha=0.55, label=approach, color=palette.get(approach, "#94a3b8"), edgecolor="white")
        ax2.legend(fontsize=8)
    ax2.set_title("Score distribution", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Overall tool score")
    ax2.set_ylabel("Frequency")
    ax3 = axes[1, 1]
    if not per_q.empty:
        for approach in approaches:
            subset = per_q.loc[per_q["approach"] == approach].sort_values("question_idx")
            if len(subset):
                ax3.plot(subset["question_idx"], subset["score"], marker="o", markersize=3, label=approach, color=palette.get(approach, "#94a3b8"))
        ax3.legend(fontsize=8)
    ax3.set_title("Scores across questions", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Question index")
    ax3.set_ylabel("Overall tool score")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path.resolve()
