"""
Phase 4 — MCP Manager
Single entry point for all MCP tools.
The hybrid retriever calls mcp_manager.retrieve_all(query) to get
documents from GitHub + Filesystem in one shot.

Place at: app/mcp/mcp_manager.py

Usage in hybrid_retriever.py:
    from app.mcp.mcp_manager import MCPManager
    mcp = MCPManager()
    docs = await mcp.retrieve_all(query)

Toggle on/off with USE_MCP=true/false in .env
"""

import asyncio
import logging
from typing import List

from app.mcp.mcp_base import MCPDocument
from app.mcp.github_mcp import GitHubMCPTool
from app.mcp.filesystem_mcp import FilesystemMCPTool

logger = logging.getLogger(__name__)


class MCPManager:
    """
    Initialises all registered MCP tools and runs them in parallel.
    Add new tools to _tools list — nothing else changes.
    """

    def __init__(self):
        self._tools = [
            GitHubMCPTool(),
            FilesystemMCPTool(),
        ]
        logger.info(
            "MCPManager ready | tools=%s",
            [t.name for t in self._tools],
        )

    async def retrieve_all(
        self, query: str, max_per_tool: int = 5
    ) -> List[MCPDocument]:
        """
        Run all tools in parallel, merge results.
        Never raises — failed tools return [].
        """
        tasks = [tool.retrieve(query, max_per_tool) for tool in self._tools]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        docs: List[MCPDocument] = []
        for tool, result in zip(self._tools, results):
            if isinstance(result, Exception):
                logger.warning("MCPManager: %s raised %s", tool.name, result)
            else:
                docs.extend(result)
                logger.info("MCPManager: %s returned %d docs", tool.name, len(result))

        return docs