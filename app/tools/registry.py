from dotenv import load_dotenv

load_dotenv()

from search.hybrid_search import eu_hybrid_search


def tool_eu_hybrid_search(q, top_k=5, local=False):
    docs = eu_hybrid_search(q, top_k=top_k, local=local)
    return [
        {
            "id": d.id,
            "title": d.title,
            "text": d.text,
            "rrf_score": float(getattr(d, "rrf_score", 0.0)),
            "source_type": getattr(d, "source_type", None),
        }
        for d in docs
    ]


def tool_pubmed_search(*args, **kwargs):
    from tools.pubmed import PubMedTool

    return PubMedTool().pubmed_semantic_search(*args, **kwargs)


def tool_wikipedia_search(*args, **kwargs):
    from tools.wikipedia import WikipediaTool

    return WikipediaTool().wiki_semantic_search(*args, **kwargs)


def tool_clinicaltrials_search(*args, **kwargs):
    import importlib

    ct = importlib.import_module("tools.clinicaltrials").EuTrialClient()
    return ct.search(*args, **kwargs)


FUNCTION_MAP = {
    "eu_hybrid_search": tool_eu_hybrid_search,
    "pubmed_search": tool_pubmed_search,
    "wikipedia_search": tool_wikipedia_search,
    "clinicaltrials_search": tool_clinicaltrials_search,
}

TOOLS_JSON = [
    {
        "type": "function",
        "function": {
            "name": "eu_hybrid_search",
            "description": "Hybrid search over the local medical corpus (Qdrant dense + Elasticsearch BM25).",
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        },
    }
]
