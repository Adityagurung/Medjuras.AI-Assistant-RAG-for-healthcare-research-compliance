from tqdm.auto import tqdm
import math


# evaluators
def hit_rate(pred_ids, true_ids):
    # Hit@k: fraction of queries where the correct document appears anywhere in the top-k results. (Binary per query: hit or miss.)
    gold = set(true_ids if isinstance(true_ids, (list, tuple, set)) else [true_ids])
    return 1.0 if any(did in gold for did in pred_ids) else 0.0


def mrr(preds, true_id):
    # MRR (Mean Reciprocal Rank): averages 1/rank of the first correct result.
    # If the correct doc is at rank 1 → 1.0; rank 5 → 0.2; not found → 0.0. Unlike Hit@k, MRR rewards putting the right doc higher.
    if preds and hasattr(preds[0], "id"):
        pred_ids = [hit.id for hit in preds]
    else:
        pred_ids = preds

    for i, doc_id in enumerate(pred_ids, 1):
        if doc_id == true_id:
            return 1.0 / i
    return 0.0


def hit_at_k(pred_ids, gold_ids, top_k=10):
    """
    Hit@k — “did we get at least one right in the top-k?”
    Use when you care about *any* relevant doc being retrieved.
    Returns 1.0 if any predicted id in top-k is relevant, else 0.0.
    """
    gold = set(gold_ids if isinstance(gold_ids, (list, tuple, set)) else [gold_ids])
    return 1.0 if any(did in gold for did in pred_ids[:top_k]) else 0.0


def mrr_at_k(pred_ids, gold_ids, top_k=10):
    """
    MRR@k — “how early is the first correct doc?”
    Good for single-answer tasks. Reward decays as 1/rank.
    Returns 1/rank of the first relevant in top-k, else 0.0.
    """
    gold = set(gold_ids if isinstance(gold_ids, (list, tuple, set)) else [gold_ids])
    for i, did in enumerate(pred_ids[:top_k], start=1):
        if did in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(pred_ids, gold_ids, top_k=10):
    """
    nDCG@k — “quality of the whole list, rewarding early hits (log discount).”
    With binary relevance (0/1) this is still useful and less brittle than MRR.
    Returns normalized score in [0,1].
    """
    gold = set(gold_ids if isinstance(gold_ids, (list, tuple, set)) else [gold_ids])
    # DCG
    dcg = 0.0
    for i, did in enumerate(pred_ids[:top_k], start=1):
        if did in gold:
            dcg += 1.0 / math.log2(i + 1)
    # IDCG (ideal DCG)
    ideal_rels = min(len(gold), top_k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_rels + 1))
    return (dcg / idcg) if idcg > 0 else 0.0


def evaluate(gt, retriever, top_k=10, local=False):
    """
    Runs all metrics and averages across queries.
    - gt_rows: [{"query": "...", "doc_id": "X"}] or [{"query": "...", "doc_ids": ["X","Y"]}]
    - retriever: fn(query, k) -> ranked list of doc_ids
    Returns: dict with Hit@k, MRR, nDCG@k
    """
    hits, mrrs, ndcgs = [], [], []
    for row in tqdm(gt, desc="Evaluating retrieval", unit="query"):
        gold_ids = row.get("doc_ids") or [row["doc_id"]]

        preds = retriever(row["query"], top_k=top_k, local=local) or []
        if preds and hasattr(preds[0], "id"):
            pred_ids = [p.id for p in preds][:top_k]
        else:
            pred_ids = preds[:top_k]

        hits.append(hit_at_k(pred_ids, gold_ids, top_k=top_k))
        mrrs.append(mrr_at_k(pred_ids, gold_ids, top_k=top_k))
        ndcgs.append(ndcg_at_k(pred_ids, gold_ids, top_k=top_k))

    n = max(len(gt), 1)
    return {
        f"Hit@{top_k}": sum(hits) / n,
        f"MRR@{top_k}": sum(mrrs) / n,
        f"nDCG@{top_k}": sum(ndcgs) / n,
    }
