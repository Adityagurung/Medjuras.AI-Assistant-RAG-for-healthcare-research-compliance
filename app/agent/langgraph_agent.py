"""LangGraph agent: corpus hybrid search + live PubMed API."""
from __future__ import annotations

import os
from typing import Annotated, Any, Dict, List, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from llm.openai_client import DEFAULT_MODEL, get_openai_client
from search.hybrid_search import hybrid_search
from tools.pubmed import PubMedTool

load_dotenv()


class AgentState(TypedDict, total=False):
    """State passed between LangGraph nodes."""

    question: str
    local: bool
    corpus_hits: List[Dict[str, Any]]
    pubmed_hits: List[Dict[str, Any]]
    answer: str


def retrieve_corpus(state: AgentState) -> AgentState:
    """Hybrid-search the indexed medical corpus."""
    hits = hybrid_search(state["question"], top_k=5, local=state.get("local", True))
    state["corpus_hits"] = [h.to_dict() for h in hits]
    return state


def retrieve_pubmed(state: AgentState) -> AgentState:
    """Query live PubMed via Entrez (requires ENTREZ_EMAIL)."""
    tool = PubMedTool()
    try:
        state["pubmed_hits"] = tool.pubmed_semantic_search(state["question"], top_k=3)
    except Exception as exc:
        state["pubmed_hits"] = [{"error": str(exc)}]
    return state


def generate_answer(state: AgentState) -> AgentState:
    """Synthesize an answer with OpenAI using retrieved context."""
    client = get_openai_client()
    corpus = state.get("corpus_hits") or []
    pubmed = state.get("pubmed_hits") or []
    context_parts = []
    for i, doc in enumerate(corpus, 1):
        context_parts.append(
            f"[Corpus {i}] {doc.get('title', '')}\n{doc.get('text', '')[:800]}"
        )
    for i, doc in enumerate(pubmed, 1):
        if isinstance(doc, dict) and doc.get("error"):
            continue
        context_parts.append(
            f"[PubMed {i}] {doc.get('title', '')}\n{doc.get('text', doc.get('abstract', ''))[:800]}"
        )
    context = "\n\n".join(context_parts) or "No context retrieved."
    system = (
        "You are MedJuras.AI, a careful medical research assistant. "
        "Answer using ONLY the provided context. Cite sources as [Corpus N] or [PubMed N]. "
        "If context is insufficient, say so. Jurisdiction default: GLOBAL."
    )
    user = f"Question: {state['question']}\n\nContext:\n{context}"
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_DEFAULT_MODEL", DEFAULT_MODEL),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
        max_tokens=600,
    )
    state["answer"] = (resp.choices[0].message.content or "").strip()
    return state


def build_agent_graph() -> Any:
    """Compile the LangGraph workflow."""
    graph = StateGraph(AgentState)
    graph.add_node("retrieve_corpus", retrieve_corpus)
    graph.add_node("retrieve_pubmed", retrieve_pubmed)
    graph.add_node("generate", generate_answer)
    graph.set_entry_point("retrieve_corpus")
    graph.add_edge("retrieve_corpus", "retrieve_pubmed")
    graph.add_edge("retrieve_pubmed", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def run_agent(question: str, *, local: bool = True) -> Dict[str, Any]:
    """Run the agent graph and return final state."""
    app = build_agent_graph()
    final = app.invoke({"question": question, "local": local})
    return final
