"""MCP (Model Context Protocol) client wrapper.

Connects to an external MCP server and converts discovered MCP tools
into standard LangChain ``Tool`` objects so they can be used by the
LangGraph agent.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from langchain_core.tools import Tool

logger = structlog.get_logger()


class MCPClientWrapper:
    """Wrapper around the ``mcp`` Python SDK that discovers remote tools and
    exposes them as LangChain ``Tool`` instances.

    Usage::

        wrapper = MCPClientWrapper(server_url="http://localhost:8080/sse")
        tools = await wrapper.get_tools()
    """

    def __init__(self, server_url: str):
        self.server_url = server_url

    async def get_tools(self) -> list[Tool]:
        """Connect to the MCP server, list available tools, and return them as
        LangChain ``Tool`` objects."""
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        tools: list[Tool] = []

        try:
            async with sse_client(url=self.server_url) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.list_tools()

                    for mcp_tool in result.tools:
                        tool = self._make_langchain_tool(session, mcp_tool)
                        tools.append(tool)

            logger.info(
                "Loaded MCP tools",
                server_url=self.server_url,
                tool_count=len(tools),
                tool_names=[t.name for t in tools],
            )
        except Exception as e:
            logger.error(
                "Failed to load MCP tools",
                server_url=self.server_url,
                error=str(e),
            )

        return tools

    def _make_langchain_tool(self, session: Any, mcp_tool: Any) -> Tool:
        """Convert a single MCP tool descriptor into a LangChain Tool."""
        tool_name = mcp_tool.name
        tool_description = mcp_tool.description or f"MCP tool: {tool_name}"
        server_url = self.server_url

        def _call_mcp_tool(input_text: str) -> str:
            """Invoke the MCP tool synchronously by bridging into asyncio."""
            return _run_async(_invoke_mcp(server_url, tool_name, input_text))

        return Tool(
            name=tool_name,
            description=tool_description,
            func=_call_mcp_tool,
        )


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from a sync context, handling running loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


async def _invoke_mcp(server_url: str, tool_name: str, input_text: str) -> str:
    """Open a fresh SSE connection to the MCP server and call a single tool."""
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    try:
        async with sse_client(url=server_url) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, {"input": input_text})
                # result.content is a list of content blocks
                parts = []
                for block in result.content:
                    if hasattr(block, "text"):
                        parts.append(block.text)
                return "\n".join(parts) if parts else str(result)
    except Exception as e:
        logger.error("MCP tool invocation failed", tool=tool_name, error=str(e))
        return f"MCP tool error: {e}"
