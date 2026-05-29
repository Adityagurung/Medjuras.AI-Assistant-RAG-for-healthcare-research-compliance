# todo CHAT HISTORY/MEMORY
import time
from contextlib import contextmanager
from datetime import datetime

from search.search_utils import Hit, rerank_rrf
from tools.registry import TOOLS_JSON


@contextmanager
def time_block(label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        dur = (time.perf_counter() - start) * 1000  # ms
        print(f"[TIMER] {label}: {dur:.2f} ms")


MEDICAL_SYSTEM_CONTEXT_TEMPLATE = """

ROLE:
You are a medical llm Resident Assistant answering a QUESTION to the best of your ability with your own knowledge and provided CONTEXT.

TASK:
- Your job is to bridge the gap between the users request and the specific highly technical academic speech patterns of your source documentation in the CONTEXT.
- The CONTEXT is built using wikipedia PubMed research papers and textbook documents/shards in your corpus as your source of information.
- At the beginning of the conversation the CONTEXT is empty.
- When you experience a lack of information or require a citation, feel free to utilize the TOOLS available to you to answer the question.
- IMPORTANT: Use the CONVERSATION_HISTORY below to understand context from previous questions and maintain conversational continuity. When users ask about "previous questions" or "what we discussed", refer to the conversation history.
- When your current query seems to lack information or context, consult the CONVERSATION_HISTORY for additional context that may help you provide a more relevant and complete answer.
- If you cannot answer the question using CONTEXT, TOOL, or your own knowledge, tell the user this and prompt for more information or ask if there are any other question that you can help with.


Safety: do not provide personalized medical advice; flag emergencies and tell the user to seek medical assistance.
Style: plain language first; concise; SI units; define acronyms once.
Locale: {locale}; Language: {language}; Today: {today_iso}

SYSTEM CONTEXT:
<USER_CONTEXT>
{user_context}
</USER_CONTEXT>

<CONVERSATION_HISTORY>
{conversation_history}
</CONVERSATION_HISTORY>

- The citations and url links may only come from the knowledge base corpus or TOOLS output.

<RESPONSE_DETAIL_CONTEXT>
{response_detail_context}
</RESPONSE_DETAIL_CONTEXT>

""".strip()


# === System prompt (tool-aware, compact) ===
MEDICAL_AGENTIC_PROMPT_TEMPLATE = """
TASK:
- Answer the user provided QUESTION accurately and succinctly.
- Prefer medical primary/secondary sources and cite them when available.
- If the context doesn't contain the answer, use your own knowledge to answer the question

<QUESTION>
{question}
</QUESTION>

CONTEXT:

<CONTEXT>
{context}
</CONTEXT>

You can perform the following actions:

- ANSWER_CONTEXT: the question using the CONTEXT provided by the search results
- ANSWER: the question using your own knowledge.
- TOOL: Select ONLY ONE of the available tools to use to help answer the question

TOOLS (use only if needed):

<TOOLS>
{tools}
</TOOLS>

REASONING (internal to you; don't expose unless asked):
- FIRST: Check if this is a simple conversational query (greetings, non-medical questions, basic social interaction) - if so, answer directly WITHOUT using any tools.
- SECOND: For medical questions, decide if you can answer confidently from your general medical knowledge - if so, answer directly WITHOUT using any tools.
- THIRD: Only use tools if the medical question requires current research, specific citations, or information you're uncertain about.
- If you decide to use a SEARCH TOOL, call HYBRID_SEARCH first to build/refresh CONTEXT from the local corpus.
- Only if HYBRID_SEARCH is insufficient, call exactly one supplemental tool (PUBMED_SEARCH or WIKIPEDIA_SEARCH).
- If tool output is low-relevance, say so briefly and answer with the best available information.

OUTPUT FORMAT:
- If you used a tool, synthesize a brief answer of unnested 3–6 sentences, from the collected CONTEXT. Respond in natural language only (no JSON).
- If you did not use a tool, answer briefly and clearly, from your own knowledge and never invent fake facts.
- Do NOT include citations, references, URLs, or a 'Citations:' section in your answer.
- Try to use the same keywords and medical terminology of the best data points in the CONTEXT.


STOP CONDITIONS:
- Stop once you’ve answered the question directly and, if applicable, providing ≤3 high-value citations.
- Stop if the user indicates that they are finished with the conversation.
- If you cannot answer the question using CONTEXT or your own knowledge, tell the user this and end the conversation.

RULES:
-Don't perform more than {max_iterations} iterations for a given student question.
The current iteration number: {iteration_number}. If we exceed the allowed number
of iterations, give the best possible answer with the provided information.
-Don't use search queries used at the previous iterations.
-Do not EVER repeat previously performed TOOLS.
-Select ONLY ONE TOOL per iteration.



SEARCH_QUERIES contains the queries that were used to retrieve the CONTEXT documents.

<SEARCH_QUERIES>
{search_queries}
</SEARCH_QUERIES>

PREVIOUS_ACTIONS contains the actions you already performed.

<PREVIOUS_ACTIONS>
{previous_actions}
</PREVIOUS_ACTIONS>

"""


def build_settings_context(settings):
    """Convert UI settings into prompt context strings"""

    user_contexts = {
        "Healthcare Provider": "Provide clinical-level information suitable for medical professionals.",
        "Medical Researcher": "Focus on research findings, methodologies, and scientific evidence.",
        "Patient": "Explain in simple terms that a patient can understand. Always recommend consulting a healthcare provider.",
    }

    response_detail_contexts = {
        "Simple": "Keep responses concise and easy to understand.",
        "Detailed": "Provide comprehensive explanations with relevant details.",
        "Technical": "Include technical details, medical terminology, and specific mechanisms.",
    }

    tool_bias_map = {
        (
            "Patient",
            "Simple",
        ): "Prefer WIKIPEDIA_SEARCH for simple explanations and definitions.",
        (
            "Patient",
            "Detailed",
        ): "Prefer WIKIPEDIA_SEARCH first, then PUBMED_SEARCH if more evidence is needed.",
        (
            "Healthcare Provider",
            "Technical",
        ): "Prefer PUBMED_SEARCH for research-backed evidence.",
        (
            "Medical Researcher",
            "Technical",
        ): "Strongly prefer PUBMED_SEARCH to gather primary literature.",
    }

    tool_bias = tool_bias_map.get(
        (settings["user_type"], settings["response_detail"]),
        "Use HYBRID_SEARCH first; then choose PUBMED_SEARCH for medical/research questions or WIKIPEDIA_SEARCH for definitions/background.",
    )

    return {
        "user_context": user_contexts.get(settings["user_type"]),
        "response_detail_context": response_detail_contexts.get(
            settings["response_detail"]
        ),
        "show_sources": settings["show_sources"],
        "tool_bias": tool_bias,
    }


def build_rag_context(settings, chat_history=None):
    """Build context string with source formatting based on user preferences"""
    # TODO: LANGUAGE SETTING IN APP
    today_iso = datetime.now().isoformat(timespec="seconds")
    # loc = locale.getdefaultlocale()
    # locale_str = f"{loc[0].replace('_', '-')}" if loc and loc[0] else "en-US"
    settings_context = build_settings_context(settings=settings)

    # Format chat history for context (keep last 5 Q&A pairs = 10 messages)
    conversation_history = ""
    if chat_history:
        # only user/assistant
        relevant_messages = [
            msg for msg in chat_history if msg.get("role") in ["user", "assistant"]
        ]

        recent_messages = relevant_messages[-10:]  # Keep last 10 messages (5 Q&A pairs)

        formatted_messages = []
        for msg in recent_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                formatted_messages.append(f"User: {content}")
            elif role == "assistant":
                formatted_messages.append(f"Assistant: {content}")
        conversation_history = "\n".join(formatted_messages)

    return MEDICAL_SYSTEM_CONTEXT_TEMPLATE.format(
        locale="de-DE",  # locale_str :that way the LLM knows whether to use °F vs °C, mg/dL vs mmol/L, spelling (hemoglobin vs haemoglobin), etc.
        language="English",
        today_iso=today_iso,
        user_context=settings_context["user_context"],
        response_detail_context=settings_context["response_detail_context"],
        conversation_history=conversation_history,
    )


def build_rag_prompt(
    question,
    settings,
    tools=TOOLS_JSON,
    search_results=[],
    search_queries=[],
    previous_actions=[],
    max_iter=3,
    curr_iter=0,
    rrf_k=60,
    top_k=6,
):

    if curr_iter == 2:
        print()

    # On final iteration, fuse + slice; otherwise leave as-is
    ranked = (
        rerank_rrf(search_results, w_rrf=1.0, w_cos=1.0, top_k=top_k)
        if curr_iter == max_iter - 1 and search_results
        else search_results
    )

    def fmt(doc):
        title = doc.get("title") or doc.get("id") or "Unknown"
        text = doc.get("text", "")
        score = doc.get("rrf_score")
        if score is not None:
            return f"Source: {title}\nContent: {text}\nRelevance Score: {score:.3f}"
        return f"Source: {title}\nContent: {text}"

    if settings.get("show_sources", True):
        context_text = "\n\n".join(fmt(d) for d in ranked)
    else:
        context_text = "\n\n".join(d.get("text", "") for d in ranked)

    tool_info = "\n".join(
        f"- {t['function']['name']}: {t['function']['description']}"
        for t in tools
        if t.get("type") == "function" and "function" in t
    )

    prompt = MEDICAL_AGENTIC_PROMPT_TEMPLATE.format(
        question=question,
        context=context_text,
        tools=tool_info,
        search_results=ranked,
        search_queries="\n".join(search_queries),
        previous_actions="\n".join(previous_actions),
        max_iterations=max_iter,
        iteration_number=curr_iter + 1,
    )

    if curr_iter == max_iter - 1:
        prompt += "\n\nFINAL_NOTE: This is the final iteration. You must answer now in 3–6 sentences using the provided CONTEXT if possible, otherwise your own knowledge. Do NOT call tools. Respond in natural language only (no JSON)."
    return prompt, ranked


# --------------------------------------------------------------------------------


# ------------------------------
# Quick test harness for prompts
# ------------------------------
if __name__ == "__main__":
    with time_block("TOTAL"):
        # --- sample settings (simulate your UI controls) ---
        settings = {
            "user_type": "Healthcare Provider",  # "Healthcare Provider" | "Medical Researcher" | "Patient"
            "response_detail": "Detailed",  # "Simple" | "Detailed" | "Technical"
            "show_sources": True,
        }

        # --- sample question ---
        question = "What are the first-line treatments for community-acquired pneumonia in adults?"

        # --- sample search results (pretend these came from hybrid search) ---
        sample_results = [
            Hit(
                id="pubmed23n0001_7",
                title="Lysosomal hydrolases of the epidermis. 2. Ester hydrolases.",
                text="Five distinct ester hydrolases (EC 3-1) have been characterized in guinea-pig epidermis...",
                rrf_score=0.912,
                source_type="pubmed",
            ),
            Hit(
                id="Anatomy_Gray_2",
                title="Anatomy_Gray",
                text="How can gross anatomy be studied? The term anatomy is derived from the Greek word temnein...",
                rrf_score=0.731,
                source_type="textbook",
            ),
        ]

        with time_block("build_rag_context"):
            sys_prompt = build_rag_context(settings)

        with time_block("build_rag_prompt"):
            user_prompt = build_rag_prompt(
                question=question,
                settings=settings,
                tools=TOOLS_JSON,
                search_results=sample_results,
                search_queries=["community acquired pneumonia first line"],
                previous_actions=[
                    "TOOL:hybrid_search({'query': 'community acquired pneumonia'})"
                ],
                max_iterations=3,
                iteration_number=0,
            )

        # --- print for eyeballing ---
        print("\n================= SYSTEM PROMPT =================\n")
        print(sys_prompt)

        print("\n================== USER PROMPT ==================\n")
        print(user_prompt)


# If you can answer the QUESTION using CONTEXT, use this template:

# {{
# "action": "ANSWER_CONTEXT",
# "answer": "<your answer>",
# "source": "CONTEXT"
# }}

# If you can, use your own knowledge to answer the question:

# {{
# "action": "ANSWER",
# "answer": "<your answer>",
# "source": "OWN_KNOWLEDGE"
# }}

# If you want to use a tool, use this template:

# {{
# "action": "TOOL",
# "reasoning": "<add your reasoning here>",
# "keywords": ["search query 1", "search query 2", ...]
# }}
