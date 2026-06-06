"""
Phase 4 — MCP Base Interface
All MCP tools implement this interface so the hybrid retriever can call them uniformly.
Toggle any tool on/off via USE_MCP=true/false in .env
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class MCPDocument:
    """Standardised document returned by any MCP tool."""
    title: str
    content: str
    url: str           # file path, GitHub URL, or API endpoint
    source: str        # "github", "filesystem", etc.
    score: float = 1.0


class BaseMCPTool(ABC):
    """
    Every MCP tool must implement retrieve().
    The hybrid retriever calls this identically for all tools.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier shown in logs and response messages."""
        ...

    @abstractmethod
    async def retrieve(self, query: str, max_results: int = 5) -> List[MCPDocument]:
        """
        Fetch documents relevant to query.
        Must never raise — return [] on any error.
        """
        ...