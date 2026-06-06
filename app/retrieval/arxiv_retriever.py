"""
app/retrieval/arxiv_retriever.py  (FIXED v2)

Root causes of repeated rate limiting
---------------------------------------
1. The old version used a flat 3 s inter-request sleep.  When 12 questions
   are queued sequentially and each ArXiv call takes ~2 s of network time,
   the effective gap between API calls is only 1 s (sleep fires AFTER the
   previous call finishes, not as a wall-clock 3 s gap).  ArXiv's documented
   rate limit is 3 requests/second but in practice they throttle heavier
   automated traffic far sooner.

2. On a 429 the old code returned [] immediately — wasting the slot.

Fixes
-----
1. True wall-clock rate limiting: record the absolute time of the last
   request and sleep the REMAINING time, regardless of how long the
   previous HTTP call took.

2. On 429: exponential back-off with jitter (3 s, 6 s, 12 s) up to
   MAX_RETRIES before giving up.

3. Request gap raised from 3 s to 4 s (ArXiv guidelines say ≥3 s;
   4 s gives a safety margin and matches HybridRetriever's 4 s loop delay).

4. Fallback query logic unchanged — still tries ti+abs first, then all:.

5. Relevance threshold unchanged at 0.20.
"""

from __future__ import annotations

import asyncio
import random
import xml.etree.ElementTree as ET
from typing import List, Dict

import httpx

from core.logger import setup_logger
from app.retrieval.query_preprocessor import build_source_queries, relevance_score

logger = setup_logger(__name__)

ARXIV_API   = "https://export.arxiv.org/api/query"
ARXIV_NS    = {"atom": "http://www.w3.org/2005/Atom"}
_REQUEST_GAP = 4.0   # seconds between requests (ArXiv policy: ≥3 s)
_MAX_RETRIES = 3     # on 429, retry this many times with back-off


class ArxivRetriever:

    # FIX 3A: Circuit breaker thresholds
    _CIRCUIT_BREAKER_THRESHOLD = 2  # consecutive 429s before opening circuit
    _CIRCUIT_COOLDOWN_SECONDS = 20.0  # recoverable cooldown instead of session lockout

    def __init__(self):
        self._last_request_time: float = 0.0
        # FIX 3A: recoverable circuit breaker state
        self._consecutive_rate_limits: int = 0
        self._circuit_open: bool = False
        self._circuit_open_until: float = 0.0
        self._circuit_warning_logged: bool = False
        logger.info(
            "ArxivRetriever initialized | circuit_breaker_threshold=2 | "
            f"cooldown={self._CIRCUIT_COOLDOWN_SECONDS}s"
        )

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    async def _wait_rate_limit(self) -> None:
        """Sleep until at least _REQUEST_GAP seconds since the last call."""
        loop    = asyncio.get_event_loop()
        now     = loop.time()
        elapsed = now - self._last_request_time
        if elapsed < _REQUEST_GAP:
            await asyncio.sleep(_REQUEST_GAP - elapsed)
        self._last_request_time = asyncio.get_event_loop().time()

    # ------------------------------------------------------------------
    # Query builder
    # ------------------------------------------------------------------

    def _build_arxiv_query(self, phrase: str) -> str:
        tokens = phrase.split()
        if not tokens:
            return f"all:{phrase}"
        if len(tokens) <= 4:
            quoted = f'"{phrase}"'
            return f"ti:{quoted} OR abs:{quoted}"
        core = " ".join(tokens[:4])
        return f'ti:"{core}" OR abs:"{phrase}"'

    # ------------------------------------------------------------------
    # HTTP fetch with retry
    # ------------------------------------------------------------------

    async def _fetch(
        self,
        client: httpx.AsyncClient,
        query_str: str,
        max_results: int,
    ) -> List[ET.Element]:
        loop = asyncio.get_event_loop()
        now = loop.time()

        # FIX 3A: recoverable circuit breaker — if cooldown elapsed, allow a probe
        if self._circuit_open:
            if now < self._circuit_open_until:
                if not self._circuit_warning_logged:
                    logger.warning(
                        "ArXiv circuit breaker OPEN — temporarily skipping requests "
                        f"(cooldown {self._CIRCUIT_COOLDOWN_SECONDS}s, trip after "
                        f"{self._CIRCUIT_BREAKER_THRESHOLD} consecutive 429s)"
                    )
                    self._circuit_warning_logged = True
                return []

            logger.info("ArXiv circuit breaker HALF-OPEN — allowing recovery probe")
            self._circuit_open = False
            self._circuit_warning_logged = False
            self._consecutive_rate_limits = 0

        params = {
            "search_query": query_str,
            "start":        0,
            "max_results":  max_results,
            "sortBy":       "relevance",
            "sortOrder":    "descending",
        }
        headers = {"User-Agent": "ResearchAssistant/1.0"}

        for attempt in range(1, _MAX_RETRIES + 1):
            await self._wait_rate_limit()
            try:
                # FIX 3B: 15s timeout for first attempt, 35s for subsequent
                timeout = 15 if attempt == 1 else 35
                resp = await client.get(
                    ARXIV_API,
                    params=params,
                    headers=headers,
                    timeout=timeout,
                )
                if resp.status_code == 429:
                    # FIX 3A: Track consecutive rate limits for circuit breaker
                    self._consecutive_rate_limits += 1
                    if self._consecutive_rate_limits >= self._CIRCUIT_BREAKER_THRESHOLD:
                        self._circuit_open = True
                        self._circuit_open_until = loop.time() + self._CIRCUIT_COOLDOWN_SECONDS
                        self._circuit_warning_logged = False
                        logger.warning(
                            f"ArXiv circuit breaker TRIPPED after {self._consecutive_rate_limits} "
                            f"consecutive 429s — cooling down for "
                            f"{self._CIRCUIT_COOLDOWN_SECONDS:.0f}s"
                        )
                        return []

                    # Exponential back-off: 3 s, 6 s, 12 s + jitter
                    wait = (2 ** attempt) * 1.5 + random.uniform(0, 1.0)
                    logger.warning(
                        f"ArXiv rate limited (attempt {attempt}/{_MAX_RETRIES}) "
                        f"— backing off {wait:.1f}s | consecutive_429s={self._consecutive_rate_limits}"
                    )
                    self._last_request_time = loop.time()
                    await asyncio.sleep(wait)
                    continue

                resp.raise_for_status()
                # Success — reset consecutive counter and close circuit
                self._consecutive_rate_limits = 0
                self._circuit_open = False
                self._circuit_open_until = 0.0
                self._circuit_warning_logged = False
                root = ET.fromstring(resp.text)
                return root.findall("atom:entry", ARXIV_NS)

            except httpx.TimeoutException:
                logger.warning(f"ArXiv timeout (attempt {attempt}/{_MAX_RETRIES}, timeout={timeout}s)")
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(2.0 * attempt)
                continue
            except Exception as e:
                logger.debug(f"ArXiv fetch error: {e}")
                return []

        logger.warning("ArXiv rate limited — giving up after retries")
        return []

    # ------------------------------------------------------------------
    # Entry parser
    # ------------------------------------------------------------------

    def _parse_entry(self, entry: ET.Element) -> Dict:
        def txt(tag: str) -> str:
            el = entry.find(tag, ARXIV_NS)
            return el.text.strip() if el is not None and el.text else ""

        title    = txt("atom:title").replace("\n", " ")
        abstract = txt("atom:summary").replace("\n", " ")
        url      = txt("atom:id")
        pub_date = txt("atom:published")[:10]
        authors  = ", ".join(
            a.find("atom:name", ARXIV_NS).text
            for a in entry.findall("atom:author", ARXIV_NS)[:3]
            if a.find("atom:name", ARXIV_NS) is not None
        )
        return {
            "source":    "arxiv",
            "title":     title,
            "url":       url,
            "content":   (
                f"{title}\n\nAuthors: {authors}\nPublished: {pub_date}"
                f"\n\nAbstract: {abstract}"
            ),
            "score":     0.85,
            "published": pub_date,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def retrieve(self, question: str, max_results: int = 4) -> List[Dict]:
        search_phrase = build_source_queries(question)["arxiv"]
        logger.info(f"ArXiv | phrase='{search_phrase}' (from: '{question[:60]}')")

        try:
            async with httpx.AsyncClient(timeout=35) as client:
                # Primary: targeted title+abstract search
                query_str = self._build_arxiv_query(search_phrase)
                entries   = await self._fetch(client, query_str, max_results * 3)

                # Fallback: broad search if nothing returned
                if not entries:
                    logger.debug("ArXiv: no results for targeted query — trying broad")
                    entries = await self._fetch(
                        client, f"all:{search_phrase}", max_results * 2
                    )

            results: List[Dict] = []
            for entry in entries:
                if len(results) >= max_results:
                    break
                doc   = self._parse_entry(entry)
                score = relevance_score(question, doc["title"], doc["content"])
                if score < 0.20:
                    logger.debug(
                        f"ArXiv dropped (score={score:.2f}): '{doc['title'][:70]}'"
                    )
                    continue
                doc["score"] = score
                results.append(doc)

            logger.info(f"ArXiv returned {len(results)} relevant results")
            return results

        except httpx.TimeoutException:
            logger.warning("ArXiv timed out")
            return []
        except Exception as e:
            logger.error(f"ArXiv failed: {e}")
            return []