import asyncio
import json
import os
from typing import Any, Dict, List

from fastmcp import Client, FastMCP
from fastmcp.prompts import Prompt
from fastmcp.resources import Resource
from fastmcp.tools import Tool

from app.search.hybrid_search import hybrid_search

"""
Need an end chat tool maybe?

#@make_async_background
https://gofastmcp.com/servers/context
https://gofastmcp.com/servers/progress
"""


mcp = FastMCP("MedicalTools")


async def test_client():
    client = Client("./mediRAG_tools_server.py")

    async with client:
        tools = await client.list_tools()
        print(f"Available tools: {[tool.name for tool in tools]}")


def mcp_tools_to_openai(tools):
    out = []
    for t in tools:
        name = getattr(t, "name", t["name"])
        desc = getattr(t, "description", t.get("description", ""))
        schema = getattr(t, "inputSchema", t.get("inputSchema", {})) or {}

        # Make a shallow copy and prune MCP-specific fields that OpenAI doesn't need
        schema = dict(schema)
        schema.pop("$schema", None)
        schema.pop("$id", None)

        # Ensure it's an object schema (OpenAI expects JSON Schema for an object)
        # TODO: tools util for conversion

        if schema.get("type") != "object":
            schema = {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            }

        # Helpful default to avoid stray args
        schema.setdefault("additionalProperties", False)

        out.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": schema,
                },
            }
        )
    return out


# TODO: @mcp_tool
def knowledge_gap_detector(query: str, medical_results: List[Dict]) -> Dict:
    """
    Analyzes if medical search results adequately answer the user query

    Args:
        query: User's original question
        medical_results: Retrieved chunks from medical knowledge base

    Returns:
        {
            "has_gap": bool,
            "gap_type": "definition|mechanism|context|coverage",
            "confidence": float,
            "missing_concepts": List[str],
            "trigger_wikipedia": bool
        }
    """
    # Cosine similarity analysis
    # Content completeness check
    # Technical complexity assessment
    # Entity coverage analysis
    pass


@mcp.tool(
    name="hybrid_search",
    description="Search the MedRag knowledge base using a Qdrant (semantic search) + ElasticSearch (BM25) hybrid retriever.",
    tags={"medrag", "search"},
    # meta={"version": "1.0", "author": "dvrk_dvys"}  # Custom metadata
)
def hybrid_search_tool(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve top documents from ES+Qdrant hybrid retriever.
    Returns a list of {id,title,text,rrf_score,source_type}.
    """
    hits = hybrid_search(query, top_k=top_k)
    return [
        {
            "id": h.id,
            "title": h.title,
            "text": h.text,
            "rrf_score": getattr(h, "rrf_score", 0.0),
            "source_type": getattr(h, "source_type", None),
        }
        for h in hits
    ]


@mcp.tool
def search_wikipedia(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve explanatory context from Wikipedia (vector/HTTP impl is up to you).
    Return the same shape as hybrid_search for easy fusion.
    Query Wikipedia vector database with gap-specific strategy
    """
    # TODO:
    return []


@mcp.tool
def simple_response_ok(query: str) -> str:
    """
    Hint to the LLM: this query is general/non-medical; answer directly w/o retrieval.
    """
    # TODO:
    pass


# TODO: @mcp_tool
def intelligent_merge_tool(medical_results, wikipedia_results, intent_data) -> Dict:
    """Combine and rerank results from multiple sources"""
    pass


# TODO: @mcp_tool
def user_intent_classifier(query: str, conversation_history: List[str] = None) -> Dict:
    """
    Determines user intent and expertise level for response customization

    Args:
        query: Current user query
        conversation_history: Previous messages for context

    Returns:
        {
            "intent_type": "definition|explanation|diagnosis|education|research",
            "user_type": "patient|student|healthcare_provider|researcher",
            "complexity_preference": "simple|intermediate|technical",
            "query_category": "symptom|condition|treatment|mechanism|drug"
        }
    """
    # Language pattern analysis
    # Medical terminology usage
    # Question structure analysis
    # Context from conversation
    pass


if __name__ == "__main__":
    # Runs an MCP server over stdin/stdout (no ports)
    mcp.run()


# User Query → Intent Classification → Medical Search → Gap Detection → [Wikipedia Search] → Merge → Format Response
# IF user_intent_classifier.user_type == "patient" AND knowledge_gap_detector.has_gap:
#    Use wikipedia_search_tool with gap_type="definition"
#    Apply simple language formatting
# ELIF user_intent_classifier.user_type == "student" AND knowledge_gap_detector.gap_type == "mechanism":
#    Use wikipedia_search_tool for foundational concepts
#    Blend with technical medical content


def get_mcp_client():
    """Get MCP client based on environment"""
    mcp_url = os.getenv("MCP_SERVER_URL", "./app/llm/mediRAG_tools_server.py")
    return Client(mcp_url)


async def initialize_tools():
    """Initialize MCP client and get tools"""
    client = get_mcp_client()
    try:
        async with client:
            tools = await client.list_tools()
            print(
                f"✅ Connected to MCP server. Available tools: {[tool.name for tool in tools]}"
            )
            return tools
    except Exception as e:
        print(f"❌ Failed to connect to MCP server: {e}")
        print("🔄 Running without tools...")
        return None
