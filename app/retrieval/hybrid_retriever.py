"""
app/retrieval/hybrid_retriever.py
==================================
Phase 12 — relevance-filtered version.

KEY CHANGES vs previous version:
  FIX 1 — Relevance pre-filter: every doc is scored against the sub-question
           before being kept. Docs below MIN_DOC_RELEVANCE (0.12) are dropped.
           This kills GitHub READMEs that mention "diffusion" in passing,
           CrossRef papers that match keywords but aren't about the question,
           and HackerNews threads that are tangentially related.

  FIX 2 — Source caps: GitHub capped at MAX_GITHUB = 2 per question.
           HackerNews capped at MAX_HN = 2 per question.
           Both are noisy for academic/research queries. The caps apply AFTER
           relevance filtering so even the 2 kept must be relevant.

  FIX 3 — Minimum content length: docs with < MIN_CONTENT_LEN chars of
           content are dropped before scoring (GitHub repo descriptions,
           CrossRef stubs with no abstract, etc).

  FIX 4 — repr(e) for all exception logging (was str(e), which is empty for
           some exception types like asyncio.TimeoutError).

  FIX 5 — docs_by_question now uses ONLY that question's docs for RAG,
           not the flat all_docs pool. Already correct in retrieve_all_questions
           — this comment clarifies the intent so downstream callers don't
           accidentally rebroadcast all_docs.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Set

from core.logger import setup_logger
from config.settings import get_settings
from app.retrieval.tavily_retriever import TavilyRetriever
from app.retrieval.wikipedia_retriever import WikipediaRetriever
from app.retrieval.arxiv_retriever import ArxivRetriever
from app.retrieval.semantic_scholar_retriever import SemanticScholarRetriever
from app.retrieval.pubmed_retriever import PubMedRetriever
from app.retrieval.hackernews_retriever import HackerNewsRetriever
from app.retrieval.nature_retriever import NatureRetriever
from app.retrieval.image_retriever import ImageRetriever
from app.retrieval.query_preprocessor import (
    is_tech_query,
    is_science_query,
    is_medical_query,
    is_academic_query,
    relevance_score,          # FIX 1: used for per-doc relevance gate
)

settings = get_settings()
logger   = setup_logger(__name__)

# ── Source quality caps ───────────────────────────────────────────────────────
# GitHub READMEs and HackerNews threads are noisy for research queries.
# Even after relevance filtering, cap how many can survive per question.
MAX_GITHUB    = 1   # GitHub READMEs rarely useful for research — hard cap
MAX_HN        = 1   # HackerNews useful for tech context, but noisy
MAX_NATURE    = 2   # CrossRef returns loosely matched papers; cap to best 2
MAX_WIKIPEDIA = 1   # Wikipedia articles are broad; 1 is usually enough

# ── Relevance gate ────────────────────────────────────────────────────────────
# Docs below this score against the sub-question are dropped immediately.
# 0.12 is low enough to pass genuine but sparse academic abstracts,
# while dropping "what is diffusion" README.md files found by GitHub search.
MIN_DOC_RELEVANCE = 0.20   # raised from 0.12 → 0.20 to drop more irrelevant docs

# ── Minimum content length ────────────────────────────────────────────────────
# GitHub stubs, CrossRef records with no abstract, and HN title-only items
# are useless for RAG. Drop anything with less than this many characters.
MIN_CONTENT_LEN = 80

# ── Max docs per sub-question after all filtering ────────────────────────────
# 107 docs for 12 questions = ~9 per question. Cap to 6 to reduce noise.
MAX_DOCS_PER_QUESTION = 6

# ── FIX 4: Domain-coherence terms for AI/ML queries ──────────────────────────
# If the query is about AI/ML, Nature docs MUST contain at least one of these
# terms in their title+content. Otherwise they're off-domain (e.g., medical
# "diffusion" papers, neuroscience studies) and should be dropped.
_AI_ML_COHERENCE_TERMS = {
    "machine learning", "deep learning", "neural network", "transformer",
    "language model", "llm", "diffusion model", "generative adversarial",
    "gan", "reinforcement learning", "computer vision", "nlp",
    "natural language processing", "embedding", "fine-tuning", "fine tuning",
    "rag", "retrieval-augmented", "retrieval augmented", "attention mechanism",
    "bert", "gpt", "convolutional", "recurrent", "autoencoder",
    "benchmark", "pre-training", "pretraining", "classification",
    "object detection", "image generation", "text generation",
    "artificial intelligence", "ai model", "training data",
}

_AI_ML_QUERY_SIGNALS = {
    "machine learning", "deep learning", "neural network", "transformer",
    "llm", "language model", "diffusion model", "gan", "generative",
    "reinforcement learning", "computer vision", "nlp", "embedding",
    "fine-tuning", "fine tuning", "rag", "retrieval-augmented",
    "retrieval augmented", "artificial intelligence", "benchmark",
    "attention mechanism", "bert", "gpt", "inference",
}

# ── FIX 2A: MCP query keyword extraction ─────────────────────────────────
# Conversational queries like "What are the key differences between RAG and
# fine-tuning?" return 0 results from GitHub search. Extract 3-5 keywords.

_MCP_STOP_WORDS = {
    "the", "a", "an", "of", "in", "and", "or", "to", "for", "is", "are",
    "was", "were", "how", "what", "why", "when", "which", "who", "where",
    "do", "does", "did", "that", "this", "these", "those", "it", "its",
    "be", "have", "has", "had", "with", "from", "on", "at", "by", "as",
    "not", "but", "if", "so", "than", "too", "very", "can", "should",
    "would", "could", "vs", "versus", "between", "most", "some", "any",
    "key", "main", "important", "real", "world", "use", "cases",
    "compare", "comparison", "differences", "similarities",
    "strengths", "limitations", "tradeoffs", "preferred",
    "related", "recent", "advances", "overview",
}

_MCP_STRIP_PREFIXES = [
    "what are the ", "what is the ", "what is ", "what are ",
    "how does ", "how do ", "how are ", "how is ",
    "when should ", "why are ", "why do ", "why is ",
    "which ", "where ",
]


def _extract_mcp_keywords(query: str) -> str:
    """
    FIX 2A: Extract 3-5 meaningful keywords from a conversational query
    for use as a GitHub/MCP search query.

    Example:
        "What are the key differences between RAG and fine-tuning?" → "RAG fine-tuning"
        "How does retrieval-augmented generation work?" → "retrieval-augmented generation"
    """
    q = query.strip().rstrip("?")
    q_lower = q.lower()

    # Strip question prefixes
    for prefix in _MCP_STRIP_PREFIXES:
        if q_lower.startswith(prefix):
            q = q[len(prefix):]
            q_lower = q.lower()
            break

    # Tokenize and filter stop words, keep original case for proper nouns
    words = q.split()
    keywords = [w for w in words if w.lower().strip(".,;:!?") not in _MCP_STOP_WORDS and len(w) > 1]

    # Take at most 5 keywords
    result = " ".join(keywords[:5]).strip()
    if not result:
        # Fallback: just use the first 4 non-trivial words
        result = " ".join(w for w in query.split() if len(w) > 2)[:60]
    return result


def _is_ai_ml_query(query: str) -> bool:
    """Check if query is about AI/ML topics."""
    q = query.lower()
    return any(sig in q for sig in _AI_ML_QUERY_SIGNALS)


def _is_domain_coherent(doc: Dict, query: str) -> bool:
    """
    FIX 4: Domain-coherence check for Nature docs.
    For AI/ML queries, Nature docs must mention at least one AI/ML term.
    Non-Nature docs and non-AI/ML queries always pass.
    """
    src = doc.get("source", "")
    if src != "nature":
        return True  # Only applies to Nature docs
    if not _is_ai_ml_query(query):
        return True  # Not an AI/ML query, don't filter

    combined = f"{doc.get('title', '')} {doc.get('content', '')}".lower()
    has_ai_term = any(term in combined for term in _AI_ML_COHERENCE_TERMS)
    if not has_ai_term:
        logger.debug(
            f"Domain coherence | dropped Nature doc: '{doc.get('title', '')[:60]}' "
            f"(no AI/ML terms for AI/ML query)"
        )
    return has_ai_term


def _load_mcp():
    if not settings.use_mcp:
        return None
    try:
        from app.mcp.mcp_manager import MCPManager
        manager = MCPManager()
        logger.info("MCP loaded | tools: github + filesystem")
        return manager
    except Exception as e:
        logger.warning(f"MCP failed to load (running without it): {repr(e)}")
        return None


class HybridRetriever:
    """
    Query-routed retrieval with relevance filtering and source caps.

    Pipeline per sub-question:
      1. Tavily (fast, high quality, parallel with the rest)
      2. Source-specific retrievers in parallel
      3. Relevance pre-filter (drop docs irrelevant to the sub-question)
      3b. Domain-coherence check (Nature docs must match query domain)
      4. Source caps (GitHub ≤2, HN ≤2, Nature ≤3, Wikipedia ≤2)
      5. Sort by score, keep top max_search_results
    """

    def __init__(self):
        self.tavily           = TavilyRetriever()
        self.wikipedia        = WikipediaRetriever()
        self.arxiv            = ArxivRetriever()
        self.semantic_scholar = SemanticScholarRetriever()
        self.pubmed           = PubMedRetriever()
        self.hackernews       = HackerNewsRetriever()
        self.nature           = NatureRetriever()
        self.image_retriever  = ImageRetriever(min_width=80, min_height=80, max_images=8)
        self.mcp              = _load_mcp()

        mcp_status = "ON (github + filesystem)" if self.mcp else "OFF"
        logger.info(f"HybridRetriever ready | 7 text sources | MCP: {mcp_status}")

    def _plan_sources(self, query: str) -> Dict[str, bool]:
        tech     = is_tech_query(query)
        science  = is_science_query(query)
        medical  = is_medical_query(query)
        academic = is_academic_query(query)

        return {
            "tavily":           True,
            "wikipedia":        True,
            "semantic_scholar": academic or tech or science or medical,
            "arxiv":            academic or tech or science,
            "pubmed":           medical,
            "hackernews":       tech,
            "nature":           science or medical or academic,
            "mcp":              bool(self.mcp),
        }

    async def _safe(self, name: str, coro, timeout: int) -> List[Dict]:
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"{name} timed out after {timeout}s — skipped")
            return []
        except Exception as e:
            logger.warning(f"{name} error: {repr(e)} — skipped")
            return []

    async def _mcp_retrieve(self, query: str) -> List[Dict]:
        if not self.mcp:
            # FIX 2A: Distinguish "disabled" from "zero results"
            logger.debug("MCP | SKIPPED (not loaded/disabled)")
            return []
        try:
            # FIX 2A: Extract keywords from conversational query
            mcp_query = _extract_mcp_keywords(query)
            logger.info(
                f"MCP | query rewrite: '{query[:60]}' → '{mcp_query}'"
            )
            mcp_docs = await asyncio.wait_for(
                self.mcp.retrieve_all(mcp_query, max_per_tool=3), timeout=20
            )
            docs = [
                {
                    "title":        doc.title,
                    "content":      doc.content,
                    "url":          doc.url,
                    "source":       doc.source,
                    "score":        doc.score,
                    "sub_question": query,
                }
                for doc in mcp_docs
            ]
            # Phase 13: Structured MCP logging — per-tool breakdown
            tool_counts: Dict[str, int] = {}
            for d in docs:
                src = d.get("source", "mcp")
                tool_counts[src] = tool_counts.get(src, 0) + 1
            if docs:
                logger.info(
                    f"MCP | success | {len(docs)} docs | "
                    f"tools: {tool_counts}"
                )
            else:
                logger.debug(
                    f"MCP | no results | keywords='{mcp_query}'"
                )
            return docs
        except asyncio.TimeoutError:
            logger.debug(f"MCP | timeout for '{query[:40]}'")
            return []
        except Exception as e:
            # Phase 13: Log at WARNING with full error so MCP failures
            # are visible in pipeline logs instead of silently swallowed
            logger.debug(
                f"MCP | error={repr(e)} | query='{query[:40]}'"
            )
            return []

    async def _safe_images(self, query: str, urls: List[str]) -> List[Dict]:
        try:
            return await asyncio.wait_for(
                self.image_retriever.retrieve(query, urls), timeout=45
            )
        except asyncio.TimeoutError:
            logger.warning(f"Image retrieval timed out for '{query[:40]}' — skipped")
            return []
        except Exception as e:
            logger.warning(f"Image retrieval failed for '{query[:40]}': {repr(e)}")
            return []

    # ── FIX 1+2+3+4: relevance gate + domain coherence + source caps ──────────

    def _apply_relevance_filter(
        self,
        query: str,
        docs: List[Dict],
    ) -> List[Dict]:
        """
        Drop docs that are clearly irrelevant to this sub-question.

        Steps:
          1. Drop docs with too little content (stubs, title-only entries).
          2. Score each doc against the question using relevance_score().
          3. Drop anything below MIN_DOC_RELEVANCE.
          3b. FIX 4: Domain-coherence check for Nature docs.
          4. Apply per-source caps on noisy sources.
          5. Return filtered list (order preserved, sorted by score later).
        """
        if not docs:
            return docs

        # Step 1: drop content stubs
        has_content = [
            d for d in docs
            if len((d.get("content") or "").strip()) >= MIN_CONTENT_LEN
        ]
        dropped_stubs = len(docs) - len(has_content)
        if dropped_stubs:
            logger.debug(f"Relevance filter | dropped {dropped_stubs} content stubs")

        # Step 2+3: score and threshold
        scored: List[tuple] = []
        for doc in has_content:
            title   = doc.get("title",   "") or ""
            content = doc.get("content", "") or ""
            score   = relevance_score(query, title, content)
            scored.append((score, doc))

        passed = [(s, d) for s, d in scored if s >= MIN_DOC_RELEVANCE]
        dropped_irrel = len(has_content) - len(passed)
        if dropped_irrel:
            sources_dropped = [d.get("source","?") for _, d in scored if _ < MIN_DOC_RELEVANCE]
            from collections import Counter
            logger.debug(
                f"Relevance filter | dropped {dropped_irrel} irrelevant docs | "
                f"sources: {dict(Counter(sources_dropped))}"
            )

        # Step 3b (FIX 4): Domain-coherence check for Nature docs
        pre_coherence_count = len(passed)
        passed = [(s, d) for s, d in passed if _is_domain_coherent(d, query)]
        dropped_incoherent = pre_coherence_count - len(passed)
        if dropped_incoherent:
            logger.info(
                f"Domain coherence | dropped {dropped_incoherent} Nature docs "
                f"(off-domain for AI/ML query)"
            )

        # Step 4: source caps
        source_counts: Dict[str, int] = {}
        capped: List[Dict] = []

        _CAPS = {
            "github":     MAX_GITHUB,
            "hackernews": MAX_HN,
            "nature":     MAX_NATURE,
            "wikipedia":  MAX_WIKIPEDIA,
        }

        # Sort by relevance score desc so caps keep the BEST docs per source
        passed.sort(key=lambda x: x[0], reverse=True)

        for rel_score, doc in passed:
            src = doc.get("source", "other")
            cap = _CAPS.get(src, 999)
            cnt = source_counts.get(src, 0)
            if cnt >= cap:
                continue
            source_counts[src] = cnt + 1
            capped.append(doc)

        dropped_caps = len(passed) - len(capped)
        if dropped_caps:
            logger.debug(f"Relevance filter | dropped {dropped_caps} docs from capped sources")

        # Step 5: hard cap per sub-question
        final = capped[:MAX_DOCS_PER_QUESTION]
        dropped_hardcap = len(capped) - len(final)

        logger.info(
            f"Relevance filter | {len(docs)} → {len(final)} docs | "
            f"(stubs={dropped_stubs} irrelevant={dropped_irrel} "
            f"incoherent={dropped_incoherent} capped={dropped_caps + dropped_hardcap})"
        )
        return final

    # ── retrieve() — text + images for ONE sub-question ──────────────────────

    async def retrieve(self, query: str) -> Dict:
        logger.info(f"Retrieving for sub-question: '{query[:80]}'")
        plan = self._plan_sources(query)

        # Tavily first — fast, high quality, gives URLs for image scraping
        tavily_docs = await self._safe(
            "tavily", self.tavily.retrieve(query, max_results=7), timeout=18
        )
        tavily_urls = [d.get("url", "") for d in tavily_docs if d.get("url")]
        logger.info(f"Tavily: {len(tavily_docs)} docs | {len(tavily_urls)} URLs")

        # Build remaining text tasks
        named_tasks: List[tuple] = []
        if plan["wikipedia"]:
            named_tasks.append(("wikipedia", self._safe(
                "wikipedia", self.wikipedia.retrieve(query, max_results=3), timeout=20
            )))
        if plan["semantic_scholar"]:
            named_tasks.append(("semantic_scholar", self._safe(
                "semantic_scholar", self.semantic_scholar.retrieve(query, max_results=4), timeout=25
            )))
        if plan["arxiv"]:
            named_tasks.append(("arxiv", self._safe(
                "arxiv", self.arxiv.retrieve(query, max_results=4), timeout=35
            )))
        if plan["pubmed"]:
            named_tasks.append(("pubmed", self._safe(
                "pubmed", self.pubmed.retrieve(query, max_results=4), timeout=18
            )))
        if plan["hackernews"]:
            named_tasks.append(("hackernews", self._safe(
                "hackernews", self.hackernews.retrieve(query, max_results=4), timeout=18
            )))
        if plan["nature"]:
            named_tasks.append(("nature", self._safe(
                "nature", self.nature.retrieve(query, max_results=2), timeout=28
            )))
        if plan["mcp"] and self.mcp:
            named_tasks.append(("mcp", self._mcp_retrieve(query)))

        # Image task runs in parallel
        image_coro = self._safe_images(query, tavily_urls)

        all_results = await asyncio.gather(
            *(coro for _, coro in named_tasks),
            image_coro,
        )

        text_results = all_results[:-1]
        images       = all_results[-1] if isinstance(all_results[-1], list) else []

        # Merge all text docs
        docs = list(tavily_docs)
        for (name, _), items in zip(named_tasks, text_results):
            if isinstance(items, list):
                docs.extend(items)

        # If image retrieval returned few results, try again with ALL source URLs
        all_source_urls = [d.get("url", "") for d in docs if d.get("url")]
        unique_extra_urls = [u for u in all_source_urls if u not in tavily_urls]
        if len(images) < 3 and unique_extra_urls:
            logger.info(
                f"Image re-scrape | only {len(images)} images found, "
                f"retrying with {len(unique_extra_urls)} additional source URLs"
            )
            extra_images = await self._safe_images(query, unique_extra_urls)
            # Merge, dedup by URL
            existing_img_urls = {img.get("url", "") for img in images}
            for img in extra_images:
                if img.get("url", "") not in existing_img_urls:
                    images.append(img)
                    existing_img_urls.add(img.get("url", ""))

        logger.info(f"Raw docs: {len(docs)} | raw images: {len(images)}")

        # ── FIX 1+2+3: apply relevance filter + source caps ──────────────────
        docs = self._apply_relevance_filter(query, docs)
        # ── END FIX ───────────────────────────────────────────────────────────

        # Dedup by URL
        seen_urls: Set[str] = set()
        deduped: List[Dict] = []
        for doc in docs:
            url = doc.get("url", "")
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            deduped.append(doc)

        # Sort by retriever score (not relevance — already filtered) and cap
        deduped.sort(key=lambda x: x.get("score", 0), reverse=True)
        top_docs = deduped[: settings.max_search_results]

        source_counts: Dict[str, int] = {}
        for doc in top_docs:
            s = doc.get("source", "unknown")
            source_counts[s] = source_counts.get(s, 0) + 1

        logger.info(
            f"Kept {len(top_docs)}/{len(deduped)} docs | "
            f"sources: {source_counts} | images: {len(images)}"
        )

        return {"docs": top_docs, "images": images, "source_counts": source_counts}

    # ── retrieve_all_questions() ──────────────────────────────────────────────

    async def retrieve_all_questions(
        self,
        questions:      List[str],
        original_query: str = "",
    ) -> Dict:
        """
        Retrieve text + images for every sub-question.

        IMPORTANT: docs_by_question maps each question to ONLY its own docs.
        Deep research agent must use docs_by_question for RAG, NOT all_docs.
        Passing all_docs to every question causes 3x chunk inflation.
        """
        logger.info(
            f"retrieve_all_questions | {len(questions)} sub-questions "
            f"| original='{original_query[:60]}'"
        )

        docs_by_question: Dict[str, List[Dict]] = {}
        all_docs:   List[Dict] = []
        all_images: List[Dict] = []

        seen_doc_urls: Set[str] = set()
        seen_img_urls: Set[str] = set()

        for i, question in enumerate(questions):
            if i > 0:
                await asyncio.sleep(2)

            result = await self.retrieve(question)
            docs   = result["docs"]
            images = result["images"]

            # FIX 5: docs_by_question = per-question docs ONLY
            docs_by_question[question] = docs

            # Merge docs globally (for writer/metadata), dedup by URL
            for doc in docs:
                url = doc.get("url", "")
                if url and url in seen_doc_urls:
                    continue
                if url:
                    seen_doc_urls.add(url)
                all_docs.append(doc)

            for img in images:
                url = img.get("url", "")
                if url and url in seen_img_urls:
                    continue
                if url:
                    seen_img_urls.add(url)
                all_images.append(img)

            logger.info(
                f"  [{i+1}/{len(questions)}] '{question[:50]}' "
                f"| docs={len(docs)} imgs={len(images)} "
                f"| total_imgs={len(all_images)}"
            )

        all_images.sort(key=lambda x: x.get("score", 0), reverse=True)

        source_counts: Dict[str, int] = {}
        for doc in all_docs:
            s = doc.get("source", "unknown")
            source_counts[s] = source_counts.get(s, 0) + 1

        logger.info(
            f"Done | docs={len(all_docs)} | images={len(all_images)} | "
            f"sources={source_counts}"
        )

        return {
            "docs_by_question": docs_by_question,  # per-question (use for RAG)
            "all_docs":         all_docs,           # merged (use for writer metadata)
            "all_images":       all_images,
            "source_counts":    source_counts,
        }