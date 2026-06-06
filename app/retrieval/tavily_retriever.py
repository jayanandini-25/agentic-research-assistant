from __future__ import annotations

"""
app/retrieval/tavily_retriever.py  (fixed)

Key fixes
---------
1. Tavily already handles full questions well (it's a real search engine),
   so we keep passing the full question.
2. BUT: we add a relevance post-filter so off-topic results are dropped.
3. Increased max_results to 7 so even after filtering we get enough docs.
"""

import certifi
import os
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()



import asyncio
from typing import List, Dict

from core.logger import setup_logger
from config.settings import get_settings
from app.retrieval.query_preprocessor import relevance_score

settings = get_settings()
logger   = setup_logger(__name__)


class TavilyRetriever:
    """
    Live web search via Tavily.
    Passes the full question (Tavily is a semantic search engine).
    Post-filters results by relevance to the original question.
    """

    def __init__(self):
        if not settings.tavily_api_key:
            logger.warning("Tavily API key not set — TavilyRetriever disabled")
            self.enabled = False
        else:
            self.enabled = True
        logger.info(f"TavilyRetriever initialized | enabled={self.enabled}")

    async def retrieve(self, question: str, max_results: int = 7) -> List[Dict]:
        if not self.enabled:
            return []

        logger.info(f"Tavily searching: '{question[:80]}'")
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=settings.tavily_api_key)

            loop     = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.search(
                    query=question,
                    search_depth="advanced",
                    max_results=max_results,
                    include_raw_content=True,
                ),
            )

            results: List[Dict] = []
            for r in response.get("results", []):
                content = r.get("raw_content") or r.get("content", "")
                if not content:
                    continue

                title = r.get("title", "")
                score = relevance_score(question, title, content)

                # Tavily is generally high quality — use a low threshold
                if score < 0.10:
                    logger.debug(f"Tavily dropped (score={score:.2f}): '{title[:60]}'")
                    continue

                results.append({
                    "source":  "tavily",
                    "title":   title,
                    "url":     r.get("url", ""),
                    "content": content[:3000],
                    "score":   score,
                })

            logger.info(f"Tavily returned {len(results)} relevant results (from raw API results)")
            return results

        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                logger.warning("Tavily rate limited")
            elif "401" in err or "invalid" in err.lower():
                logger.error("Tavily API key invalid")
                self.enabled = False
            else:
                logger.error(f"Tavily failed: {e}")
            return []