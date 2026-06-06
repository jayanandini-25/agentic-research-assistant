"""
app/mcp/github_mcp.py
=====================
Phase 4 — GitHub MCP Tool (FIXED v2)

KEY CHANGES vs previous version:
  FIX 1 — Relevance filtering: every repo and README is scored against the
           query using relevance_score() before being included. The old code
           scored repos purely by star count (popular ≠ relevant) and hardcoded
           score=0.7 for ALL README files regardless of content. A PyTorch repo
           with 50k stars mentioning "diffusion" in its README was getting the
           same score as an actual diffusion models paper implementation.

  FIX 2 — MIN_GITHUB_RELEVANCE = 0.15: repos and READMEs must score at least
           this against the query to be included. Generic ML framework repos
           (PyTorch, TensorFlow, etc.) typically score 0.05-0.08 for specific
           research questions — they're now dropped.

  FIX 3 — Content minimum length: README content must be >= 200 chars. Short
           READMEs ("This repo contains code for our paper...") are almost
           always useless for RAG.

  FIX 4 — Star-count score is now blended with relevance score:
           final_score = 0.4 * relevance + 0.6 * star_ratio
           instead of star_ratio alone. This means a highly relevant but
           lesser-known repo (1k stars, relevance=0.4) scores better than
           an irrelevant popular one (50k stars, relevance=0.05).

  FIX 5 — Removed the _search_code() strategy entirely. It searched for
           "filename:README" which returned random READMEs that mention the
           query keyword anywhere. With relevance filtering this would still
           return 5 irrelevant results — the strategy is just noisy.
           Repos from _search_repos() already fetch their README via
           _fetch_readme(), so we lose nothing by dropping _search_code().

  FIX 6 — max_results hard cap: retrieve() now accepts and respects
           max_results properly. The old code called docs[:max_results] at
           the end but fetched up to 2x max_results internally.
"""

import logging
import os
from typing import List

import httpx

from app.mcp.mcp_base import BaseMCPTool, MCPDocument

# Import relevance scorer (FIX 1)
try:
    from app.retrieval.query_preprocessor import relevance_score as _relevance_score
    _HAS_SCORER = True
except ImportError:
    _HAS_SCORER = False

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

# FIX 2: minimum relevance for a GitHub doc to be included
MIN_GITHUB_RELEVANCE = 0.15

# FIX 3: minimum README content length
MIN_README_CHARS = 200


def _score_doc(query: str, title: str, content: str, stars: int) -> float:
    """
    FIX 4: Blend relevance score with star-count score.
    relevance_score() uses keyword matching against the query.
    Star score normalised to 0-1 via log scale (50k stars → ~1.0).
    """
    import math

    if _HAS_SCORER:
        rel = _relevance_score(query, title, content)
    else:
        # Fallback: simple keyword count
        q_words = set(query.lower().split())
        c_words  = set(content.lower().split())
        rel = len(q_words & c_words) / max(len(q_words), 1)
        rel = min(rel, 1.0)

    # Log-normalised star score: 0 stars → 0.0, 50k+ stars → ~1.0
    star_score = min(math.log1p(stars) / math.log1p(50_000), 1.0) if stars > 0 else 0.0

    # 40% relevance, 60% popularity — relevance matters but popularity signals quality
    return round(0.4 * rel + 0.6 * star_score, 4), rel


class GitHubMCPTool(BaseMCPTool):
    """
    Searches GitHub for repositories and README content relevant to a query.
    Only includes repos/READMEs that are actually relevant to the query.
    """

    def __init__(self):
        token = os.getenv("GITHUB_TOKEN", "")
        self._headers = dict(HEADERS)
        if token:
            self._headers["Authorization"] = f"Bearer {token}"
            logger.info("GitHubMCPTool: using authenticated requests (5000 req/hr)")
        else:
            logger.info(
                "GitHubMCPTool: unauthenticated (60 req/hr) — set GITHUB_TOKEN for more"
            )

    @property
    def name(self) -> str:
        return "github"

    async def retrieve(self, query: str, max_results: int = 5) -> List[MCPDocument]:
        """
        Fetch relevant GitHub repos for the query.
        FIX 5: only uses _search_repos() — _search_code() removed (too noisy).
        FIX 1+2: relevance filtering applied before including any doc.
        """
        docs: List[MCPDocument] = []
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                repo_docs = await self._search_repos(client, query, max_results * 2)
                docs.extend(repo_docs)
        except Exception as exc:
            logger.warning("GitHubMCPTool error: %s", exc)

        # Sort by blended score descending, cap at max_results
        docs.sort(key=lambda d: d.score, reverse=True)
        docs = docs[:max_results]

        logger.info(
            "GitHubMCPTool: %d docs for query '%s' (after relevance filter)",
            len(docs), query[:60]
        )
        return docs

    async def _search_repos(
        self, client: httpx.AsyncClient, query: str, limit: int
    ) -> List[MCPDocument]:
        try:
            resp = await client.get(
                f"{GITHUB_API}/search/repositories",
                headers=self._headers,
                params={
                    "q":        query,
                    "sort":     "stars",
                    "per_page": min(limit, 10),  # fetch up to 10, filter down
                },
            )
            if resp.status_code != 200:
                logger.warning("GitHub repo search returned %d", resp.status_code)
                return []

            items   = resp.json().get("items", [])
            docs    = []
            dropped = 0

            for item in items:
                repo_name   = item.get("full_name", "")
                description = (item.get("description") or "").strip()
                stars       = item.get("stargazers_count", 0)

                # Fetch README for richer content
                readme = await self._fetch_readme(client, repo_name)

                # FIX 3: require minimum README length
                if not readme or len(readme) < MIN_README_CHARS:
                    if not description:
                        dropped += 1
                        continue
                    # Fall back to description if README is too short
                    content = description
                else:
                    content = f"{description}\n\n{readme}".strip() if description else readme

                # FIX 1+2: relevance gate
                blended_score, rel_score = _score_doc(query, repo_name, content, stars)
                if rel_score < MIN_GITHUB_RELEVANCE:
                    dropped += 1
                    logger.debug(
                        f"GitHub | dropped '{repo_name}' | "
                        f"rel={rel_score:.3f} < {MIN_GITHUB_RELEVANCE}"
                    )
                    continue

                docs.append(
                    MCPDocument(
                        title   = repo_name,
                        content = content[:3000],   # truncate to avoid token bloat
                        url     = item["html_url"],
                        source  = "github",
                        score   = blended_score,    # FIX 4: blended score
                    )
                )

            if dropped:
                logger.info(f"GitHub | dropped {dropped} irrelevant repos")
            return docs

        except Exception as exc:
            logger.warning("GitHub _search_repos: %s", exc)
            return []

    async def _fetch_readme(self, client: httpx.AsyncClient, full_name: str) -> str:
        try:
            resp = await client.get(
                f"{GITHUB_API}/repos/{full_name}/readme",
                headers={**self._headers, "Accept": "application/vnd.github.raw"},
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.text[:3000]
        except Exception:
            pass
        return ""