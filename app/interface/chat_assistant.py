"""Streamlit chat backend with OpenAI (default) or Ollama."""
from __future__ import annotations

from typing import Any, Dict, List

from llm.ollama_client import ollama_chat
from llm.openai_client import LLMResponse, agentic_llm
from llm.query_rewriter import rewrite_query_with_context
from llm.rag_utils import build_rag_context
from search.hybrid_search import hybrid_search


class ChatAssistant:
    """Medical RAG chat with provider selection in settings."""

    def __init__(
        self, developer_prompt: str = "You are a helpful medical research assistant."
    ):
        self.developer_prompt = developer_prompt
        self.chat_messages: List[Dict[str, Any]] = [
            {"role": "developer", "content": self.developer_prompt}
        ]

    def _ollama_rag_response(self, query: str, settings: Dict[str, Any]) -> LLMResponse:
        """Retrieve with hybrid search and answer via Ollama."""
        re_query = rewrite_query_with_context(query, self.chat_messages)
        hits = hybrid_search(re_query, top_k=5, local=True)
        context_parts = [
            f"[{i}] {h.title}\n{h.text[:600]}" for i, h in enumerate(hits, 1)
        ]
        context = "\n\n".join(context_parts) or "No retrieved context."
        sys_prompt = build_rag_context(settings, self.chat_messages)
        user_prompt = f"Question: {query}\n\nRetrieved context:\n{context}"
        resp = ollama_chat(user_prompt, system=sys_prompt)
        citations = [h.to_dict() for h in hits]
        return LLMResponse(
            text=resp.text,
            model=resp.model,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            used_tools=[("hybrid_search", citations)],
        )

    def query_llm(self, query: str, settings: Dict[str, Any]) -> tuple[LLMResponse, list]:
        """Route to OpenAI agentic RAG or Ollama RAG based on settings."""
        provider = (settings.get("llm_provider") or "openai").lower()
        if provider == "ollama":
            result = self._ollama_rag_response(query, settings)
            citations = []
            if result.used_tools:
                for _name, payload in result.used_tools:
                    if isinstance(payload, list):
                        citations.extend(payload)
            return result, citations

        re_query = rewrite_query_with_context(query, self.chat_messages)
        result, out_citations = agentic_llm(
            query=re_query,
            settings=settings,
            chat_history=self.chat_messages,
            local=True,
        )
        return result, out_citations

    def process_message(self, question: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Handle one user turn and update chat history."""
        if question.strip().lower() == "stop":
            return {"answer": "Chat ended."}

        self.chat_messages.append({"role": "user", "content": question})
        response, out_citations = self.query_llm(query=question, settings=settings)
        self.chat_messages.append(
            {"role": "assistant", "content": response.text, "citations": out_citations}
        )
        return {
            "answer": response.text,
            "model": response.model,
            "tokens": response.total_tokens,
            "citations": out_citations,
            "used_tools": response.used_tools or [],
        }
