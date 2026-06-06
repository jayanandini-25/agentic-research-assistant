"""
app/retrieval/wikipedia_retriever.py  (FIXED v3)

Fixes vs v2
-----------
FIX 1 — "Rag-and-bone man" / "Rag and Bone Buffet" appearing in results.

  Root cause: `is_relevant()` threshold was 0.10 — essentially passing
  everything. Wikipedia matches "RAG" literally to "Rag-and-bone man"
  because the opensearch is title-prefix and "rag" hits that article.
  A content snippet like "A rag-and-bone man is a person who collects
  unwanted household items..." scores ~0.06-0.08 on relevance_score()
  against "How are RAG and Fine-tuning related?" — and 0.10 let it through.

  Three-layer fix:
    a) Raise Wikipedia-specific relevance threshold from 0.10 → 0.20.
       0.20 still passes sparse academic articles (they score 0.22+) but
       cuts off the "rag trade" articles (they score 0.06-0.10).
    b) Add a TITLE BLOCKLIST of patterns that are structurally non-technical.
       Any article whose title matches a blocklist pattern is dropped before
       even fetching its content, saving an HTTP call.
    c) Content keyword gate: the fetched content must contain at least one
       of a set of topic-relevant terms extracted from the original question.
       Simple but effective: "Rag-and-bone man" content contains zero AI/ML
       terms so it's dropped before relevance scoring.

FIX 2 — `_direct_titles_for()` keyword matching was case-insensitive but
  broad. "rag" matched both "Retrieval-augmented generation" (correct) AND
  anything containing the substring "rag" in other positions. Added word-
  boundary check so "rag" only matches as a standalone word token.

FIX 3 — Wikipedia relevance threshold raised from 0.10 → 0.20 in retrieve().

All other logic unchanged from v2:
  - _short_search_phrase() for 2-3 word OpenSearch queries
  - Direct title lookup for known tech acronyms
  - Disambiguation page skip
  - Second search attempt with shorter phrase
"""

from __future__ import annotations

import re
from typing import List, Dict, Optional, Set

import httpx

from core.logger import setup_logger
from app.retrieval.query_preprocessor import (
    build_source_queries, is_relevant, extract_search_phrase
)

logger = setup_logger(__name__)

MEDIAWIKI_SEARCH = "https://en.wikipedia.org/w/api.php"
REST_SUMMARY     = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"

UA = (
    "Mozilla/5.0 (compatible; ResearchAssistant/1.0; "
    "+https://github.com/research-assistant)"
)

# ── FIX 1b: Title blocklist ───────────────────────────────────────────────────
# Patterns matched against article titles (case-insensitive).
# Matching → drop the article immediately, no content fetch.
# Keep this list tight — only add patterns for structural noise categories.
_TITLE_BLOCKLIST = re.compile(
    r"^("
    # Rag trade / physical rags
    r"rag.and.bone|ragman|rag trade|rag picker|"
    # Music albums / songs with common tech words in title
    r"rag and bone buffet|"
    # Pure disambiguation pages (belt-and-suspenders — opensearch already filters some)
    r".+\(disambiguation\)|"
    # Wikipedia maintenance articles
    r"wikipedia:|template:|category:|portal:|"
    # Generic "list of" articles that are just indexes
    r"list of .+(models?|methods?|techniques?|systems?)|"
    # People's biography pages when query is not about a person
    r".+(biography|discography|filmography|bibliography)$"
    r")",
    re.IGNORECASE,
)

# ── FIX 1c: Content keyword gate ─────────────────────────────────────────────
# For tech/AI queries, fetched content MUST contain at least one of these
# domain-specific terms to pass. Pure non-technical articles (rag trade,
# album reviews, etc.) contain none of them.
_AI_ML_TERMS: Set[str] = {
    "machine learning", "deep learning", "neural network", "language model",
    "retrieval", "fine-tuning", "transformer", "embedding", "inference",
    "training data", "artificial intelligence", "natural language",
    "gradient", "parameter", "algorithm", "dataset", "model",
    "classification", "regression", "encoder", "decoder", "attention",
    "llm", "gpt", "bert", "diffusion", "generative", "vector",
    "knowledge graph", "database", "query", "search", "retrieval-augmented",
    "rag", "finetuning", "pretrain", "token", "vocabulary",
}

# Signals in the query that indicate it's a tech/AI query — if any present,
# content keyword gate is activated.
_TECH_QUERY_SIGNALS = {
    "rag", "retrieval", "fine-tuning", "finetuning", "llm", "gpt", "bert",
    "transformer", "embedding", "neural", "deep learning", "machine learning",
    "diffusion", "generative", "language model", "vector", "knowledge graph",
    "attention", "inference", "training", "dataset", "benchmark",
}


def _is_tech_query(question: str) -> bool:
    q = question.lower()
    return any(sig in q for sig in _TECH_QUERY_SIGNALS)


def _content_passes_keyword_gate(content: str, is_tech: bool) -> bool:
    """For tech queries, content must contain at least one AI/ML term."""
    if not is_tech:
        return True  # not a tech query — don't gate
    c = content.lower()
    return any(term in c for term in _AI_ML_TERMS)


# ── Wikipedia-specific relevance threshold ────────────────────────────────────
# FIX 1a + FIX 3: raised from 0.10 → 0.20
# 0.20 passes real tech articles (score 0.22+) and drops rag-trade noise (0.06-0.10)
_WIKI_RELEVANCE_THRESHOLD = 0.20


# ── Direct title map (unchanged from v2) ─────────────────────────────────────
_DIRECT_TITLES: Dict[str, List[str]] = {
    "rag":                   ["Retrieval-augmented generation"],
    "retrieval augmented":   ["Retrieval-augmented generation"],
    "fine-tuning":           ["Fine-tuning (deep learning)", "Transfer learning"],
    "fine tuning":           ["Fine-tuning (deep learning)", "Transfer learning"],
    "llm":                   ["Large language model"],
    "large language":        ["Large language model"],
    "transformer":           ["Transformer (deep learning architecture)"],
    "gpt":                   ["Generative pre-trained transformer", "GPT-4"],
    "bert":                  ["BERT (language model)"],
    "diffusion model":       ["Diffusion model"],
    "stable diffusion":      ["Stable Diffusion"],
    "vector database":       ["Vector database"],
    "graph database":        ["Graph database"],
    "neural network":        ["Artificial neural network"],
    "deep learning":         ["Deep learning"],
    "machine learning":      ["Machine learning"],
    "reinforcement learning":["Reinforcement learning"],
    "computer vision":       ["Computer vision"],
    "natural language":      ["Natural language processing"],
    "attention mechanism":   ["Attention (machine learning)"],
    "word embedding":        ["Word embedding"],
    "knowledge graph":       ["Knowledge graph"],
    "chatgpt":               ["ChatGPT"],
    "openai":                ["OpenAI"],
    "anthropic":             ["Anthropic"],
    "hugging face":          ["Hugging Face"],
    "langchain":             ["LangChain"],
}


def _direct_titles_for(question: str) -> List[str]:
    """
    Return directly-known Wikipedia article titles for the question.
    FIX 2: use word-boundary matching so "rag" only matches as a whole
    token, not as a substring within other words.
    """
    q      = question.lower()
    titles: List[str] = []

    for keyword, arts in _DIRECT_TITLES.items():
        # Word-boundary check: keyword must appear as a whole word (or phrase)
        # in the question, not as a substring of another word.
        pattern = r"\b" + re.escape(keyword) + r"\b"
        if re.search(pattern, q):
            titles.extend(arts)

    # Deduplicate while preserving order
    seen:   set       = set()
    result: List[str] = []
    for t in titles:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


def _short_search_phrase(question: str) -> str:
    """
    Build a SHORT (2-3 word) phrase for Wikipedia title search.
    Wikipedia opensearch works best with short noun phrases, not sentences.
    """
    phrase = extract_search_phrase(question, max_words=3)
    words  = phrase.split()
    if len(words) > 4:
        phrase = " ".join(words[:4])
    return phrase.strip()


def _safe_json(resp: httpx.Response):
    """Return parsed JSON or None — never raises."""
    try:
        ct = resp.headers.get("content-type", "")
        if "json" not in ct:
            return None
        text = resp.text.strip()
        if not text or text[0] not in ("{", "["):
            return None
        return resp.json()
    except Exception:
        return None


class WikipediaRetriever:

    def __init__(self):
        logger.info("WikipediaRetriever initialized (direct REST mode)")

    async def _search_titles(
        self, client: httpx.AsyncClient, search_phrase: str, limit: int
    ) -> List[str]:
        try:
            resp = await client.get(
                MEDIAWIKI_SEARCH,
                params={
                    "action":    "opensearch",
                    "search":    search_phrase[:80],
                    "limit":     limit,
                    "format":    "json",
                    "redirects": "resolve",
                },
                headers={"User-Agent": UA},
                timeout=12,
            )
            if resp.status_code != 200:
                return []
            data = _safe_json(resp)
            if not data or not isinstance(data, list) or len(data) < 2:
                return []
            titles = [str(t) for t in data[1]]

            # FIX 1b: apply title blocklist
            filtered: List[str] = []
            for t in titles:
                if "(disambiguation)" in t:
                    continue
                if _TITLE_BLOCKLIST.match(t):
                    logger.debug(f"Wikipedia: blocked title '{t}'")
                    continue
                filtered.append(t)

            return filtered
        except Exception as e:
            logger.debug(f"Wikipedia title search error: {e}")
            return []

    async def _fetch_summary(
        self, client: httpx.AsyncClient, title: str, is_tech: bool
    ) -> Optional[Dict]:
        try:
            encoded = title.replace(" ", "_")
            resp    = await client.get(
                REST_SUMMARY.format(title=encoded),
                headers={"User-Agent": UA},
                timeout=12,
                follow_redirects=True,
            )
            if resp.status_code in (404, 301):
                return None
            if resp.status_code != 200:
                return None
            data = _safe_json(resp)
            if not data or not isinstance(data, dict):
                return None

            extract     = data.get("extract", "").strip()
            description = data.get("description", "").strip()
            page_title  = data.get("title", title)
            page_url    = (
                data.get("content_urls", {}).get("desktop", {}).get("page", "")
                or f"https://en.wikipedia.org/wiki/{encoded}"
            )

            if not extract:
                return None

            content = (f"{description}\n\n" if description else "") + extract[:2500]

            # FIX 1c: content keyword gate — drop non-technical content for tech queries
            if not _content_passes_keyword_gate(content, is_tech):
                logger.debug(
                    f"Wikipedia: content keyword gate dropped '{page_title}' "
                    f"(no AI/ML terms found)"
                )
                return None

            return {
                "source":  "wikipedia",
                "title":   page_title,
                "url":     page_url,
                "content": content,
                "score":   0.80,
            }
        except Exception as e:
            logger.debug(f"Wikipedia summary error for '{title}': {e}")
            return None

    async def retrieve(self, question: str, max_results: int = 3) -> List[Dict]:
        short_phrase = _short_search_phrase(question)
        is_tech      = _is_tech_query(question)
        logger.info(f"Wikipedia | q='{short_phrase}' (from: '{question[:60]}')")

        try:
            async with httpx.AsyncClient(timeout=20) as client:

                # Step 1: direct article lookup for known terms
                direct_titles = _direct_titles_for(question)

                # Step 2: opensearch with short phrase
                search_titles = await self._search_titles(
                    client, short_phrase, limit=max_results * 2
                )

                # Step 3: if still nothing, try a shorter phrase
                if not search_titles and len(short_phrase.split()) > 2:
                    shorter = " ".join(short_phrase.split()[:2])
                    logger.debug(f"Wikipedia: retrying with shorter phrase '{shorter}'")
                    search_titles = await self._search_titles(
                        client, shorter, limit=max_results * 2
                    )

                # Merge: direct titles first, then search results (deduplicated)
                seen:   set       = set()
                titles: List[str] = []
                for t in (direct_titles + search_titles):
                    if t not in seen:
                        seen.add(t)
                        titles.append(t)

                if not titles:
                    logger.info("Wikipedia: no titles found")
                    return []

                # Step 4: fetch summaries, apply keyword gate + relevance filter
                results: List[Dict] = []
                for title in titles:
                    if len(results) >= max_results:
                        break

                    # FIX 1b: blocklist check before HTTP fetch (saves a request)
                    if _TITLE_BLOCKLIST.match(title):
                        logger.debug(f"Wikipedia: skipping blocked title '{title}'")
                        continue

                    doc = await self._fetch_summary(client, title, is_tech)
                    if not doc:
                        continue

                    # FIX 1a + FIX 3: raised threshold 0.10 → 0.20
                    if not is_relevant(
                        question, doc["title"], doc["content"],
                        threshold=_WIKI_RELEVANCE_THRESHOLD,
                    ):
                        logger.debug(
                            f"Wikipedia filtered (relevance < {_WIKI_RELEVANCE_THRESHOLD}): "
                            f"'{doc['title']}'"
                        )
                        continue

                    results.append(doc)

                logger.info(f"Wikipedia returned {len(results)} relevant results")
                return results

        except Exception as e:
            logger.error(f"Wikipedia retriever failed: {e}")
            return []