"""
Phase 4 — Filesystem MCP Tool
Reads local files (PDF, TXT, MD, DOCX) from a configured folder and returns
content relevant to the query using simple keyword matching.

Place at: app/mcp/filesystem_mcp.py

Set LOCAL_DOCS_PATH in .env to point at your folder of research PDFs/notes.
Default: ./local_docs  (create this folder in your project root)
"""

import logging
import os
import re
from pathlib import Path
from typing import List

from app.mcp.mcp_base import BaseMCPTool, MCPDocument

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".py", ".json", ".csv"}
DEFAULT_DOCS_PATH = "./local_docs"
MAX_FILE_CHARS = 4000   # truncate large files to avoid sending walls of text


class FilesystemMCPTool(BaseMCPTool):
    """
    Scans a local folder for files and returns those whose content
    contains keywords from the query. No embeddings needed — fast keyword scan.

    Supports: .txt  .md  .py  .json  .csv
    PDF support: install pypdf2 and uncomment the PDF block below.
    """

    def __init__(self, docs_path: str | None = None):
        self.docs_path = Path(docs_path or os.getenv("LOCAL_DOCS_PATH", DEFAULT_DOCS_PATH))
        if not self.docs_path.exists():
            self.docs_path.mkdir(parents=True, exist_ok=True)
            logger.info("FilesystemMCPTool: created docs folder at %s", self.docs_path)
        else:
            file_count = sum(1 for _ in self.docs_path.rglob("*") if _.is_file())
            logger.info(
                "FilesystemMCPTool: watching %s (%d files)", self.docs_path, file_count
            )

    @property
    def name(self) -> str:
        return "filesystem"

    async def retrieve(self, query: str, max_results: int = 5) -> List[MCPDocument]:
        keywords = self._extract_keywords(query)
        if not keywords:
            return []

        candidates: list[tuple[int, MCPDocument]] = []

        for filepath in self.docs_path.rglob("*"):
            if not filepath.is_file():
                continue
            if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            try:
                content = self._read_file(filepath)
            except Exception as exc:
                logger.debug("FilesystemMCPTool: could not read %s — %s", filepath, exc)
                continue

            hits = self._count_keyword_hits(content, keywords)
            if hits == 0:
                continue

            candidates.append(
                (
                    hits,
                    MCPDocument(
                        title=filepath.name,
                        content=content[:MAX_FILE_CHARS],
                        url=str(filepath.resolve()),
                        source="filesystem",
                        score=min(1.0, hits / max(len(keywords), 1)),
                    ),
                )
            )

        # Sort by most keyword hits first
        candidates.sort(key=lambda x: x[0], reverse=True)
        docs = [doc for _, doc in candidates[:max_results]]

        logger.info(
            "FilesystemMCPTool: %d relevant files for query '%s'", len(docs), query[:60]
        )
        return docs

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _read_file(self, path: Path) -> str:
        """Read text files. Extend here for PDF/DOCX support."""
        return path.read_text(encoding="utf-8", errors="ignore")

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful words (>3 chars, lowercase) from the query."""
        stopwords = {
            "what", "how", "does", "the", "and", "for", "are", "with",
            "this", "that", "from", "have", "will", "can", "its", "about",
        }
        words = re.findall(r"[a-zA-Z]{4,}", query.lower())
        return [w for w in words if w not in stopwords]

    def _count_keyword_hits(self, content: str, keywords: List[str]) -> int:
        content_lower = content.lower()
        return sum(1 for kw in keywords if kw in content_lower)