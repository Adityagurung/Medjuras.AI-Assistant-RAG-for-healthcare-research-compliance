import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from llm.query_rewriter import rewrite_query_with_context
from llm.rag_utils import build_rag_context, build_rag_prompt
import certifi
import httpx
from openai import OpenAI
from tools.registry import FUNCTION_MAP, TOOLS_JSON

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def _ssl_verify_context():
    """Build SSL context: certifi + optional corporate CA bundle from env."""
    import ssl
    from pathlib import Path

    ca_paths = []
    for env_name in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CORPORATE_CA_BUNDLE"):
        p = os.getenv(env_name)
        if p and Path(p).is_file():
            ca_paths.append(p)

    ca_paths.append(certifi.where())
    # De-duplicate while preserving order.
    seen = set()
    unique = []
    for p in ca_paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    if len(unique) == 1:
        cafile = unique[0]
    else:
        merged = Path(__file__).resolve().parents[2] / "data" / "merged_ca_bundle.pem"
        merged.parent.mkdir(parents=True, exist_ok=True)
        parts = []
        for p in unique:
            parts.append(Path(p).read_text(encoding="utf-8").strip())
        merged.write_text("\n".join(parts) + "\n", encoding="utf-8")
        cafile = str(merged)
        os.environ["SSL_CERT_FILE"] = cafile
        os.environ["REQUESTS_CA_BUNDLE"] = cafile

    os.environ.setdefault("SSL_CERT_FILE", cafile)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", cafile)
    return ssl.create_default_context(cafile=cafile)



def get_openai_client(api_key=None):
    """OpenAI client with certifi / corporate CA bundle (fixes SSL verify errors)."""
    key = api_key or OPENAI_API_KEY
    return OpenAI(
        api_key=key,
        http_client=httpx.Client(verify=certifi.where(), timeout=float(os.getenv("OPENAI_TIMEOUT", "120"))),
    )


DEFAULT_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
available_models = ["gpt-4o-mini", "gpt-5", "gpt-4o", "o3", "o4"]


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    used_tools: Optional[List[Tuple[str, Any]]] = None


"""

to do, give chat history for context
messages = [
  {"role": "system", "content": "You’re a careful medical assistant. Explain clearly."},
  {"role": "user", "content": "What are early MI symptoms?"}
]

"""


def llm(prompt, sys_prompt=None, model=DEFAULT_MODEL) -> LLMResponse:
    """
    Call OpenAI LLM with the prompt

    "system" – instructions/behavior setup.
    "user" – the user’s prompt.
    "assistant" – previous assistant replies (for context).
    "tool" – the output returned by a tool/function you called (if you use tools).
    """

    try:
        messages = []

        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": prompt})

        client = get_openai_client()
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.1, max_tokens=500
        )

        msg = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        prompt_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tok = getattr(usage, "completion_tokens", 0) if usage else 0
        total_tok = getattr(usage, "total_tokens", prompt_tok + completion_tok)

        return LLMResponse(
            text=msg,
            model=model,
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
            total_tokens=total_tok,
        )

    except Exception as e:
        return LLMResponse(
            text=f"Error calling LLM: {e}",
            model=model,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )


def filter_used_tools(tools, used):
    """Return tools list with already used tool names removed."""
    used_names = {name for name, _ in used}
    return [t for t in tools if t["function"]["name"] not in used_names]


def format_citation(doc):
    stype = doc.get("source_type")
    title = doc.get("title", "Unknown")

    if stype == "pubmed":
        url = doc.get("url", "")
        year = doc.get("year", "")
        return f"{title} ({year}). {url}"
    elif stype == "wikipedia":
        url = doc.get("url", "")
        return f"{title}. {url}"
    else:  # textbook or local corpus
        return title


def agentic_llm(
    query,
    settings,
    tools=TOOLS_JSON,
    function_map=FUNCTION_MAP,
    model=DEFAULT_MODEL,
    temperature=0.1,
    max_tokens=500,
    max_iterations=3,
    local=False,
    tools_per_iter=2,
    chat_history=None,
) -> LLMResponse:
    """
    Call OpenAI LLM with the prompt using an iterative agent loop.

    Process:
    - The model is given a system prompt (behavior setup) and a user prompt (built with context).
    - The loop runs up to `max_iterations` (default: 3).
        • Iteration 0: the model may answer directly or decide to call a tool (ideally HYBRID_SEARCH first).
        • Iteration 1: if more information is needed, the model may call one supplemental tool
          (e.g., PUBMED_SEARCH or WIKIPEDIA_SEARCH).
        • Iteration 2 (final): the model must provide an answer using the accumulated context.
    - Tool calls are executed locally, and their outputs are collected into:
        • `search_results` – evidence gathered from tools
        • `search_queries` – queries issued to tools
        • `previous_actions` – record of which tools were used
    - These collections are passed back into the prompt each iteration so the model can refine its answer.

    Roles:
    - "system" – behavior and task setup
    - "user" – the constructed question + context
    - "assistant" – prior model responses (including tool calls)
    - "tool" – structured outputs from executed tools

    The loop stops early if the model answers without calling a tool, or after the final iteration.
    """
    client = get_openai_client()

    # Build context based on user type and detail level
    sys_prompt = build_rag_context(settings, chat_history)

    # Build RAG prompt
    search_results = []
    search_queries = []
    previous_actions = []
    used_tools = []

    try:
        for i in range(max_iterations):
            print(f"___Iteration {i}___")

            if i == 2:
                print()

            # Use rewritten query if available, otherwise use original
            # current_query = re_query if re_query else query
            try:
                prompt, search_results = build_rag_prompt(
                    question=query,  # re_query,
                    settings=settings,
                    search_results=search_results,
                    search_queries=search_queries,
                    previous_actions=previous_actions,
                    max_iter=max_iterations,
                    curr_iter=i,
                )
            except Exception as e:
                print(f"Error building RAG prompt : {e}")
                raise

            messages = []
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            messages.append({"role": "user", "content": prompt})

            # R1: let the model decide if it wants tools
            filtered_tools = filter_used_tools(tools, used_tools)
            if i == max_iterations - 1:
                filtered_tools = []

            r1 = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=filtered_tools,
                tool_choice="auto",
                temperature=temperature,
                max_tokens=max_tokens,
            )
            m1 = r1.choices[0].message
            tool_calls = getattr(m1, "tool_calls", None) or []

            tool_calls = tool_calls[
                :tools_per_iter
            ]  #!! WHY DOES IT KEEP SELECTING BOTH PZBMED AND WIKIPEDIA TOOL ON THE SECOND ITER?
            # tool_calls = [c for c in tool_calls if c.function.name in {"pubmed_search", "wikipedia_search"}][:1]

            # If last try or no tools were called, we’re done
            if i == max_iterations - 1 or not tool_calls:
                usage = getattr(r1, "usage", None)
                final_response = LLMResponse(
                    text=m1.content or "",
                    model=model,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                    completion_tokens=(
                        getattr(usage, "completion_tokens", 0) if usage else 0
                    ),
                    total_tokens=getattr(usage, "total_tokens", 0) if usage else 0,
                    used_tools=used_tools,
                )

                citations = (
                    []
                    if not used_tools or not search_results
                    else [format_citation(d) for d in search_results]
                )
                return final_response, citations

            # Execute tool calls
            tool_messages = []
            for call in tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments)

                # Add local parameter for functions that support it
                if name in function_map:
                    func = function_map[name]
                    import inspect

                    sig = inspect.signature(func)
                    if "local" in sig.parameters:
                        args["local"] = local

                print(f"DEBUG: About to call tool {name} with args: {args}")
                try:
                    result = function_map[name](**args)
                    print(f"DEBUG: Tool {name} returned result type: {type(result)}")
                except Exception as tool_error:
                    print(f"DEBUG: Tool {name} failed with error: {tool_error}")
                    raise  # Re-raise to trigger outer exception handler

                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": name,
                        "content": json.dumps(result),
                    }
                )
                used_tools.append((name, result))
                search_queries.append(str(args))
                previous_actions.append(f"TOOL:{name}({args})")

                if isinstance(result, list):
                    search_results.extend(result)
                elif isinstance(result, dict) and "hits" in result:
                    search_results.extend(result["hits"])

            # feed tool results back
            messages.append(
                {
                    "role": "assistant",
                    "content": m1.content or "",
                    "tool_calls": m1.tool_calls,
                }
            )
            messages.extend(tool_messages)

            print("tool_messages: ", tool_messages)
            print("used_tools: ", used_tools)
            print("search_queries: ", search_queries)
            print("previous_actions: ", previous_actions)

    except Exception as e:
        return (
            LLMResponse(
                text=f"Error calling LLM: {e}",
                model=model,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                used_tools=None,
            ),
            [],
        )  # for emtoy citations


if __name__ == "__main__":
    print("🤖 OpenAI Client Test")
    print("=" * 50)

    def in_docker():
        return os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER") == "1"

    # streamlit app settings
    settings = {
        "user_type": "Healthcare Provider",  # "Healthcare Provider" | "Medical Researcher" | "Patient"
        "response_detail": "Detailed",  # "Simple" | "Detailed" | "Technical"
        "show_sources": True,
    }

    test_query = [
        # "Hello Howre you?",
        "What are the connections between prolonged corticosteroid use and femoral avascular necrosis?",
        # "What are the symptoms of a heart attack?",
        # "What is avascular necrosis?",
    ]

    for t in test_query:
        print(t)
        print("=" * 50)

        # Run the agent loop (max_iterations defaults to 3 inside agentic_llm)
        result, out_citations = agentic_llm(
            query=t,
            settings=settings,
            model=DEFAULT_MODEL,
            temperature=0.1,
            max_tokens=500,
            local=True,
        )

        print("— Response —")
        print(result.text)
        print("citations: ", out_citations)

        print("\n— Usage —")
        print(f"Model: {result.model}")
        print(
            f"Prompt tokens: {result.prompt_tokens} | "
            f"Completion tokens: {result.completion_tokens} | "
            f"Total: {result.total_tokens}"
        )

        if result.used_tools:
            print("\n— Tools used —")
            for name, _payload in result.used_tools:
                print(f"- {name}")
        else:
            print("\n— Tools used — none")
        print("=" * 50)
