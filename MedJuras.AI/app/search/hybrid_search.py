import os
from dotenv import load_dotenv
load_dotenv()
from search.minsearch_baseline import hydrate_hits, minsearch_ids
from search.qdrant_search import dense_ids

def eu_hybrid_search(q, top_k=5, local=False):
    dense = dense_ids(q, limit=50, local=local)
    sparse = minsearch_ids(q, limit=50)
    scores = {}
    for lst, w in ((dense, 2.0), (sparse, 1.0)):
        for rank, doc_id in enumerate(lst, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w * (1.0 / (60 + rank))
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    docs = hydrate_hits([i for i, _ in fused])
    sm = dict(fused)
    for d in docs:
        d.rrf_score = sm.get(d.id, 0.0)
    docs.sort(key=lambda h: -h.rrf_score)
    return docs

hybrid_search = eu_hybrid_search
