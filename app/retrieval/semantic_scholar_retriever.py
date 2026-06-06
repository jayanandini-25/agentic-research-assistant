"""
app/retrieval/semantic_scholar_retriever.py  (fixed)

Key fixes
---------
1. Uses preprocessor — sends keyword phrase, not full question sentence.
2. Post-filter: relevance scored against original question.
3. Removed the "skip after 3 consecutive failures" logic that was silently
   suppressing this source.  Failures are logged but don't permanently disable.
"""

from __future__ import annotations

import asyncio
from typing import List, Dict

import httpx

from core.logger import setup_logger
from config.settings import get_settings
from app.retrieval.query_preprocessor import build_source_queries, relevance_score

settings = get_settings()
logger   = setup_logger(__name__)

SS_API = "https://api.semanticscholar.org/graph/v1/paper/search"


class SemanticScholarRetriever:

    def __init__(self):
        self.api_key = getattr(settings, "semantic_scholar_api_key", "")
        mode = "WITH key" if self.api_key else "WITHOUT key (rate-limited)"
        logger.info(f"SemanticScholarRetriever initialized {mode}")

    async def retrieve(self, question: str, max_results: int = 5) -> List[Dict]:
        search_phrase = build_source_queries(question)["semantic_scholar"]
        logger.info(f"SemanticScholar | phrase='{search_phrase}' (from: '{question[:60]}')")

        headers = {"User-Agent": "ResearchAssistant/1.0"}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        try:
            async with httpx.AsyncClient(timeout=25) as client:
                resp = await client.get(
                    SS_API,
                    params={
                        "query":  search_phrase,
                        "limit":  max_results * 2,   # fetch extra, filter down
                        "fields": "title,abstract,authors,year,externalIds,openAccessPdf",
                    },
                    headers=headers,
                )

                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("retry-after", 30))
                    logger.warning(f"SemanticScholar rate limited — retry after {retry_after}s")
                    if retry_after <= 15:
                        await asyncio.sleep(retry_after)
                        resp = await client.get(
                            SS_API,
                            params={
                                "query":  search_phrase,
                                "limit":  max_results,
                                "fields": "title,abstract,authors,year,externalIds,openAccessPdf",
                            },
                            headers=headers,
                        )
                        if resp.status_code == 429:
                            return []
                    else:
                        return []

                resp.raise_for_status()
                data = resp.json()

            results: List[Dict] = []
            for paper in data.get("data", []):
                abstract = paper.get("abstract") or ""
                if not abstract:
                    continue   # skip papers without abstracts — can't verify relevance

                pdf_info = paper.get("openAccessPdf") or {}
                ext_ids  = paper.get("externalIds") or {}
                url = (
                    pdf_info.get("url")
                    or (f"https://arxiv.org/abs/{ext_ids['ArXiv']}" if "ArXiv" in ext_ids else "")
                    or f"https://www.semanticscholar.org/paper/{paper.get('paperId', '')}"
                )
                authors = ", ".join(
                    a.get("name", "") for a in paper.get("authors", [])[:3]
                )
                title = paper.get("title", "")
                content = (
                    f"{title}\n\nAuthors: {authors}\n"
                    f"Year: {paper.get('year', 'N/A')}\n\nAbstract: {abstract}"
                )

                score = relevance_score(question, title, content)
                if score < 0.20:
                    logger.debug(f"SS dropped (score={score:.2f}): '{title[:70]}'")
                    continue

                results.append({
                    "source":  "semantic_scholar",
                    "title":   title,
                    "url":     url,
                    "content": content,
                    "score":   score,
                })

                if len(results) >= max_results:
                    break

            logger.info(f"SemanticScholar returned {len(results)} relevant results")
            return results

        except httpx.TimeoutException:
            logger.warning("SemanticScholar timed out")
            return []
        except Exception as e:
            logger.error(f"SemanticScholar failed: {e}")
            return []