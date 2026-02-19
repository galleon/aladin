"""Tool definitions for the LangGraph tool-calling agent platform.

This module provides LangChain @tool functions that wrap existing backend
capabilities (retrieval, translation, web ingestion) as well as an MCP
client for dynamically loading external tools.
"""

from .search_knowledge_base import search_knowledge_base
from .translate_text import translate_text
from .ingest_url import ingest_url
from .mcp_client import MCPClientWrapper

# Registry of built-in tool names to their LangChain tool objects
BUILTIN_TOOLS = {
    "search_knowledge_base": search_knowledge_base,
    "translate_text": translate_text,
    "ingest_url": ingest_url,
}


def get_tools_by_names(tool_names: list[str]) -> list:
    """Resolve a list of tool name strings to LangChain tool objects.

    Unknown names are silently skipped (they may be MCP-provided tools
    resolved separately).
    """
    return [BUILTIN_TOOLS[name] for name in tool_names if name in BUILTIN_TOOLS]


__all__ = [
    "search_knowledge_base",
    "translate_text",
    "ingest_url",
    "MCPClientWrapper",
    "BUILTIN_TOOLS",
    "get_tools_by_names",
]
