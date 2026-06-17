"""Streamlit chat backend via MyGenAssist or local Ollama."""
from __future__ import annotations
import os
import time


from typing import Any, Dict, List

from llm.mygenassist_client import (
    get_streamlit_anthropic_model,
    get_streamlit_openai_model,
    mygenassist_chat,
    use_mygenassist,
)

from llm.ollama_client import ollama_chat
from llm.openai_client import LLMResponse
from monitoring import cost_utils
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
    def _load_chat_history(self, prior_messages=None) -> None:
        self.chat_messages = [{"role": "developer", "content": self.developer_prompt}]
        for msg in prior_messages or []:
            if msg.get("role") in ("user", "assistant") and msg.get("content"):
                self.chat_messages.append({"role": msg["role"], "content": msg["content"]})



    def _require_mygenassist(self) -> None:
        if not use_mygenassist():
            raise ValueError("MYGENASSIST_API_KEY is required. Set it in .env.")


    def _rag_response(
        self,
        query: str,
        settings: Dict[str, Any],
        *,
        provider: str,
        model: str,
    ) -> LLMResponse:
        re_query = rewrite_query_with_context(query, self.chat_messages)
        hits = hybrid_search(re_query, top_k=5, local=True)
        context_parts = [
            f"[{i}] {h.title}\n{h.text[:600]}" for i, h in enumerate(hits, 1)
        ]
        context = "\n\n".join(context_parts) or "No retrieved context."
        sys_prompt = build_rag_context(settings, self.chat_messages)
        user_prompt = f"Question: {query}\n\nRetrieved context:\n{context}"
        if provider == "ollama":
            resp = ollama_chat(user_prompt, system=sys_prompt, model=model)
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
        else:
            self._require_mygenassist()
            resp = mygenassist_chat(user_prompt, system=sys_prompt, model=model)
            prompt_tokens = resp.prompt_tokens
            completion_tokens = resp.completion_tokens
            total_tokens = resp.total_tokens
        citations = [h.to_dict() for h in hits]
        return LLMResponse(
            text=resp.text,
            model=resp.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            used_tools=[("hybrid_search", citations)],
        )

    def query_llm(self, query: str, settings: Dict[str, Any]) -> tuple[LLMResponse, list]:
        provider = (settings.get("llm_provider") or "openai").lower()
        if provider == "ollama":
            result = self._rag_response(query, settings, provider="ollama", model=os.getenv("OLLAMA_MODEL", "llama3.2"))
            citations = []
            if result.used_tools:
                for _name, payload in result.used_tools:
                    if isinstance(payload, list):
                        citations.extend(payload)
            return result, citations
        if provider == "anthropic":
            result = self._rag_response(query, settings, provider="anthropic", model=get_streamlit_anthropic_model())
            citations = []
            if result.used_tools:
                for _name, payload in result.used_tools:
                    if isinstance(payload, list):
                        citations.extend(payload)
            return result, citations
        result = self._rag_response(
            query,
            settings,
            provider="openai",
            model=get_streamlit_openai_model(),
        )
        citations = []
        if result.used_tools:
            for _name, payload in result.used_tools:
                if isinstance(payload, list):
                    citations.extend(payload)
        return result, citations

    def process_message(
        self,
        question: str,
        settings: Dict[str, Any],
        prior_messages: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """Handle one user turn and update chat history."""
        if question.strip().lower() == "stop":
            return {"answer": "Chat ended."}

        self._load_chat_history(prior_messages)
        if not self.chat_messages or self.chat_messages[-1].get("content") != question:
            self.chat_messages.append({"role": "user", "content": question})
        started = time.perf_counter()
        response, out_citations = self.query_llm(query=question, settings=settings)
        elapsed = time.perf_counter() - started
        self.chat_messages.append(
            {"role": "assistant", "content": response.text, "citations": out_citations}
        )
        return {
            "answer": response.text,
            "model": response.model,
            "tokens": response.total_tokens,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "response_time_sec": round(elapsed, 2),
            "estimated_cost_usd": cost_utils.estimate_llm_cost_usd(response.model, response.prompt_tokens, response.completion_tokens),
            "citations": out_citations,
            "used_tools": response.used_tools or [],
        }
