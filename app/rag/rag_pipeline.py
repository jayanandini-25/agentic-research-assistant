import re
import math
import hashlib
import logging
from typing import List, Dict, Tuple, Optional, Set

import numpy as np

try:
    from core.logger import setup_logger
    logger = setup_logger(__name__)
except ModuleNotFoundError:
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

try:
    from app.retrieval.query_preprocessor import relevance_score as _qs_relevance
    _HAS_RELEVANCE_SCORER = True
except ImportError:
    _HAS_RELEVANCE_SCORER = False

# ── Embedding model (singleton) ───────────────────────────────────────────────

try:
    from sentence_transformers import SentenceTransformer
    _MODEL: Optional[SentenceTransformer] = None

    def _get_model() -> SentenceTransformer:
        global _MODEL
        if _MODEL is None:
            logger.info("Loading embedding model (first run: ~22MB download)...")
            _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            logger.info("Embedding model loaded")
        return _MODEL

    EMBEDDINGS_AVAILABLE = True

except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("sentence-transformers not installed — using keyword fallback")

# ── Tunable constants ─────────────────────────────────────────────────────────

# FIX R1 (CRITICAL): all-MiniLM-L6-v2 has a 256-token hard limit.
# 400 words ≈ 530 tokens → silent truncation → crushed similarity scores.
# 160 words ≈ 210 tokens → safely within the model's limit.
CHUNK_SIZE         = 160    # FIX R1: was 400 — truncation was killing similarity scores
CHUNK_OVERLAP      = 30     # FIX R2: was 80 — smaller chunks need less overlap

MIN_SCORE          = 0.42   # cuts noisy tail chunks (was 0.32 before Phase 12)
FALLBACK_SCORE     = 0.35   # tighter fallback (was 0.20 before Phase 12)
TOP_K              = 6      # max chunks returned per question
MMR_LAMBDA         = 0.7    # MMR trade-off: relevance vs diversity
MIN_CONTENT_CHARS  = 80     # docs shorter than this are skipped
LOW_CONF_THRESHOLD = 0.22   # chunks below this get low_confidence=True

# FIX B: doc-level relevance gate — docs below this against the question
# are dropped before chunking. Low so we don't drop sparse academic abstracts.
DOC_RELEVANCE_GATE = 0.10

# ── Query expansion ───────────────────────────────────────────────────────────

_EXPANSIONS = {
    "difference":   "{q} key differences comparison contrast",
    "compare":      "{q} comparison similarities differences tradeoffs",
    " vs ":         "{q} comparison versus differences advantages disadvantages",
    "tradeoff":     "{q} tradeoffs advantages disadvantages when to use",
    "similarit":    "{q} similarities shared properties common features overlap",
    "what is":      "{q} definition overview introduction explanation",
    "what are":     "{q} definition overview examples explanation",
    "how do":       "{q} mechanism process steps method explanation",
    "how does":     "{q} mechanism process steps method explanation",
    "latency":      "{q} inference latency speed throughput benchmark",
    "scale":        "{q} scalability distributed training deployment",
    "benchmark":    "{q} performance benchmark evaluation results metrics",
    "limitation":   "{q} limitations challenges weaknesses drawbacks failure modes",
    "out-of-vocab": "{q} OOV handling subword tokenization vocabulary",
    "strength":     "{q} strengths advantages capabilities benefits superiority",
    "preferred":    "{q} when to use best use case selection criteria recommendation",
    "when should":  "{q} when to use best use case selection criteria recommendation",
    "memory":       "{q} memory footprint parameters model size GPU",
    "training":     "{q} training dynamics convergence stability loss",
    "dynamics":     "{q} training dynamics convergence stability instability",
    # FIX 3A: abstract comparison triggers that previously had poor embeddings
    "related":      "{q} relationship connection overlap complementary",
    "relationship": "{q} relationship connection overlap complementary",
    "use case":     "{q} applications deployment production use case real-world",
}

# FIX 3A: extract entity names from comparison questions
_ENTITY_SPLIT_PATTERNS = [
    r"between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
    r"of\s+(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\?|$)",
    r"(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\?|$)",
    r"(.+?)\s+and\s+(.+?)(?:\s+(?:in|on|for|with)\b|\?|$)",
]

def _extract_entities_from_question(question: str) -> list:
    """Extract entity names from a comparison question for query expansion."""
    import re as _re
    q = question.strip().rstrip("?")
    for pattern in _ENTITY_SPLIT_PATTERNS:
        m = _re.search(pattern, q, _re.IGNORECASE)
        if m:
            entities = [g.strip() for g in m.groups() if g and g.strip()]
            entities = [e for e in entities if len(e) > 1 and e.lower() not in _STOP]
            if entities:
                return entities
    return []


def _expand_query(question: str) -> str:
    q_lower = question.lower()
    for trigger, template in _EXPANSIONS.items():
        if trigger in q_lower:
            expanded = template.format(q=question)
            # FIX 3A: inject entity names for comparison questions
            entities = _extract_entities_from_question(question)
            if entities:
                expanded += " " + " ".join(entities)
            return expanded
    if len(question.split()) < 6:
        return f"{question} explanation overview definition details"
    return question

# ── Content quality pre-filter ────────────────────────────────────────────────

_JUNK = re.compile(
    r"(cookie policy|privacy policy|terms of service|subscribe now|"
    r"click here|advertisement|all rights reserved|"
    r"javascript is disabled|enable javascript|sign up for|newsletter)",
    re.IGNORECASE,
)

def _is_quality(text: str) -> bool:
    if len(text) < MIN_CONTENT_CHARS:
        return False
    if len(_JUNK.findall(text)) >= 2:
        return False
    alpha = sum(1 for c in text if c.isalpha())
    if alpha / max(len(text), 1) < 0.4:
        return False
    return True

# ── FIX B: document-level relevance gate ─────────────────────────────────────

def _doc_passes_relevance_gate(question: str, doc: Dict) -> bool:
    """
    Return True if the doc is relevant enough to the question to be chunked.
    Uses relevance_score() from query_preprocessor — same scorer as the retriever.
    Falls back to True (pass everything) if the scorer isn't available.
    """
    if not _HAS_RELEVANCE_SCORER:
        return True
    title   = doc.get("title",   "") or ""
    content = doc.get("content", "") or ""
    score   = _qs_relevance(question, title, content)
    return score >= DOC_RELEVANCE_GATE

# ── Sentence-boundary chunking ────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    text = re.sub(r'(?<=[.!?])\s+(?=[A-Z])', '\n', text)
    return [s.strip() for s in text.split('\n') if s.strip()]

def _chunk_doc(
    content: str, title: str, url: str, source: str,
    chunk_size: int, overlap: int,
) -> List[Dict]:
    sentences  = _split_sentences(content)
    chunks: List[Dict] = []
    buf_words: List[str] = []
    buf_sents: List[str] = []

    def _flush():
        if buf_words:
            text = " ".join(buf_words)
            chunks.append({
                "content": text,
                "title":   title,
                "url":     url,
                "source":  source,
                "_hash":   hashlib.md5(text.encode()).hexdigest()[:12],
            })

    def _trim_to_overlap():
        nonlocal buf_words, buf_sents
        keep_sents: List[str] = []
        count = 0
        for s in reversed(buf_sents):
            w = len(s.split())
            if count + w > overlap:
                break
            keep_sents.insert(0, s)
            count += w
        buf_sents[:] = keep_sents
        buf_words[:] = " ".join(buf_sents).split()

    for sent in sentences:
        sw = sent.split()
        if buf_words and len(buf_words) + len(sw) > chunk_size:
            _flush()
            _trim_to_overlap()
        buf_words.extend(sw)
        buf_sents.append(sent)

    _flush()
    return chunks

# ── Embedding-based scoring ───────────────────────────────────────────────────

def _score_embeddings(
    question: str,
    chunks: List[Dict],
) -> List[Tuple[Dict, float, np.ndarray]]:
    model    = _get_model()
    texts    = [c["content"] for c in chunks]
    expanded = _expand_query(question)

    q_vec  = model.encode(expanded, convert_to_numpy=True, normalize_embeddings=True)
    c_vecs = model.encode(
        texts,
        convert_to_numpy     = True,
        normalize_embeddings = True,
        batch_size           = 64,
        show_progress_bar    = False,
    )
    return [(chunk, float(np.dot(q_vec, cv)), cv)
            for chunk, cv in zip(chunks, c_vecs)]

# ── Keyword fallback ──────────────────────────────────────────────────────────

_STOP = {
    "the","a","an","of","in","and","or","to","for","is","are","was","were",
    "it","this","that","with","on","at","by","from","as","be","has","have",
    "not","do","does","did","how","what","when","why","which","who","where",
}

def _score_keywords(
    question: str,
    chunks: List[Dict],
) -> List[Tuple[Dict, float, None]]:
    q_words = [w.lower() for w in question.split() if w.lower() not in _STOP]
    if not q_words:
        return [(c, 0.5, None) for c in chunks]

    doc_freq: Dict[str, int] = {}
    cw_list: List[Set[str]] = []
    for chunk in chunks:
        words = set(w.lower() for w in chunk["content"].split())
        cw_list.append(words)
        for w in words:
            doc_freq[w] = doc_freq.get(w, 0) + 1

    N = len(chunks)
    scored = []
    for chunk, cw in zip(chunks, cw_list):
        total = max(len(chunk["content"].split()), 1)
        score = 0.0
        for qw in q_words:
            if qw in cw:
                tf  = chunk["content"].lower().count(qw) / total
                idf = math.log((N + 1) / (doc_freq.get(qw, 0) + 1)) + 1.0
                score += tf * idf
        max_s = len(q_words) * 0.1 * (math.log(N + 1) + 1)
        norm  = min(score / max(max_s, 1e-9), 1.0)
        scored.append((chunk, norm, None))
    return scored

# ── MMR reranking ─────────────────────────────────────────────────────────────

def _mmr_select(
    scored: List[Tuple[Dict, float, Optional[np.ndarray]]],
    top_k: int,
    lmbda: float = MMR_LAMBDA,
) -> List[Tuple[Dict, float]]:
    if not scored:
        return []
    if scored[0][2] is None:
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(c, s) for c, s, _ in scored[:top_k]]

    selected: List[Tuple[Dict, float, np.ndarray]] = []
    candidates = list(scored)

    while candidates and len(selected) < top_k:
        if not selected:
            best = max(candidates, key=lambda x: x[1])
        else:
            sel_vecs = np.stack([v for _, _, v in selected])

            def _mmr_score(item: Tuple[Dict, float, np.ndarray]) -> float:
                _, rel, vec = item
                max_sim = float((sel_vecs @ vec).max())
                return lmbda * rel - (1 - lmbda) * max_sim

            best = max(candidates, key=_mmr_score)

        selected.append(best)
        candidates.remove(best)

    return [(c, s) for c, s, _ in selected]

# ── RAGPipeline class ─────────────────────────────────────────────────────────

class RAGPipeline:

    def __init__(
        self,
        chunk_size    : int   = CHUNK_SIZE,
        chunk_overlap : int   = CHUNK_OVERLAP,
        min_score     : float = MIN_SCORE,
        top_k         : int   = TOP_K,
        mmr_lambda    : float = MMR_LAMBDA,
    ):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_score     = min_score
        self.top_k         = top_k
        self.mmr_lambda    = mmr_lambda
        mode = "embeddings+MMR" if EMBEDDINGS_AVAILABLE else "keyword-fallback"
        logger.info(
            f"RAGPipeline initialized | mode={mode} | top_k={top_k} | "
            f"min_score={min_score} | chunk_size={chunk_size} (≈{chunk_size*1.3:.0f} tokens) | "
            f"fallback_score={FALLBACK_SCORE} | doc_gate={DOC_RELEVANCE_GATE}"
        )

    def filter_docs(
        self,
        question  : str,
        docs      : List[Dict],
        min_score : Optional[float] = None,
        top_k     : Optional[int]   = None,
    ) -> List[Dict]:
        """
        Filter one question's docs down to top-K relevant chunks.

        Steps:
          1. Quality pre-filter (remove junk, short content)
          2. FIX B: Document-level relevance gate (drop irrelevant docs)
          3. Sentence-aware chunking (within-question dedup)
          4. Embedding-based scoring vs expanded query
          5. Primary threshold filter (min_score)
          6. Fallback tiers if needed
          7. MMR for diversity
        """
        effective_min_score = min_score if min_score is not None else self.min_score
        effective_top_k     = top_k     if top_k     is not None else self.top_k

        if not docs:
            return []

        # Step 1: Quality pre-filter
        clean = [d for d in docs if _is_quality((d.get("content") or "").strip())]
        if not clean:
            logger.warning(f"RAG | all docs failed quality filter for '{question[:50]}'")
            return []

        # ── FIX B: Document-level relevance gate ─────────────────────────────
        # Score docs once so we can both gate and produce a sensible fallback.
        doc_relevance: List[Tuple[Dict, float]] = []
        if _HAS_RELEVANCE_SCORER:
            for d in clean:
                title = d.get("title", "") or ""
                content = d.get("content", "") or ""
                doc_relevance.append((d, _qs_relevance(question, title, content)))
        else:
            doc_relevance = [(d, 1.0) for d in clean]

        relevant_docs = [d for d, score in doc_relevance if score >= DOC_RELEVANCE_GATE]
        dropped_gate  = len(clean) - len(relevant_docs)
        if dropped_gate:
            logger.info(
                f"RAG | doc-gate dropped {dropped_gate}/{len(clean)} docs "
                f"for '{question[:50]}'"
            )
        if not relevant_docs:
            # Fallback: keep the top few docs by relevance rather than
            # restoring every clean doc, which would reintroduce noise.
            ranked = sorted(doc_relevance, key=lambda item: item[1], reverse=True)
            relevant_docs = [d for d, score in ranked[:3]]
            best_score = ranked[0][1] if ranked else 0.0
            logger.warning(
                "RAG | doc-gate dropped ALL docs — using top-3 quality-filtered "
                f"fallback docs (best_score={best_score:.3f})"
            )
        # ── END FIX B ────────────────────────────────────────────────────────

        # Step 3: Chunk with within-question dedup
        seen_hashes: Set[str] = set()
        all_chunks: List[Dict] = []
        for doc in relevant_docs:
            for chunk in _chunk_doc(
                content    = (doc.get("content") or "").strip(),
                title      = doc.get("title",  ""),
                url        = doc.get("url",    ""),
                source     = doc.get("source", ""),
                chunk_size = self.chunk_size,
                overlap    = self.chunk_overlap,
            ):
                if chunk["_hash"] not in seen_hashes:
                    seen_hashes.add(chunk["_hash"])
                    all_chunks.append(chunk)

        if not all_chunks:
            return []

        logger.info(
            f"RAG | '{question[:60]}' | "
            f"{len(docs)} raw → {len(relevant_docs)} relevant docs "
            f"→ {len(all_chunks)} unique chunks | "
            f"min_score={effective_min_score} top_k={effective_top_k}"
        )

        # Step 4: Score chunks vs query
        try:
            scored = (
                _score_embeddings(question, all_chunks)
                if EMBEDDINGS_AVAILABLE
                else _score_keywords(question, all_chunks)
            )
        except Exception as e:
            logger.error(f"RAG | scoring error: {e} — keyword fallback")
            scored = _score_keywords(question, all_chunks)

        if scored:
            all_scores = sorted([s for _, s, _ in scored], reverse=True)
            logger.debug(
                f"RAG | score dist: max={all_scores[0]:.3f} "
                f"median={all_scores[len(all_scores)//2]:.3f} "
                f"min={all_scores[-1]:.3f}"
            )

        # Step 5: Primary threshold filter
        primary = [(c, s, v) for c, s, v in scored if s >= effective_min_score]
        used_fallback = False

        # FIX 3B: Single-chunk guardrail
        # If only 1 chunk passes primary threshold, force-include the next 2
        # best-scoring chunks to prevent single-source hallucination.
        if 0 < len(primary) <= 1:
            remaining = sorted(
                [(c, s, v) for c, s, v in scored if s < effective_min_score],
                key=lambda x: x[1], reverse=True,
            )
            supplement = remaining[:2]
            if supplement:
                logger.warning(
                    f"RAG | FIX 3B | only {len(primary)} chunk(s) above threshold "
                    f"for '{question[:50]}' — adding {len(supplement)} supplementary "
                    f"chunks (scores: {[round(s,3) for _,s,_ in supplement]})"
                )
                primary.extend(supplement)

        # Step 6: Fallback tiers
        if not primary:
            used_fallback = True
            logger.warning(
                f"RAG | 0 chunks ≥ {effective_min_score} for '{question[:50]}' "
                f"— trying fallback={FALLBACK_SCORE}"
            )
            primary = [(c, s, v) for c, s, v in scored if s >= FALLBACK_SCORE]

        if not primary:
            used_fallback = True
            logger.warning(f"RAG | fallback empty — taking top-3, flagged low_confidence")
            primary = sorted(scored, key=lambda x: x[1], reverse=True)[:3]

        # Step 7: MMR
        top = _mmr_select(primary, top_k=effective_top_k, lmbda=self.mmr_lambda)

        # Build output
        result = []
        for chunk, score in top:
            out = {k: v for k, v in chunk.items() if k != "_hash"}
            out["rag_score"]      = round(score, 4)
            out["low_confidence"] = score < LOW_CONF_THRESHOLD
            if used_fallback:
                out["_used_fallback"] = True
            result.append(out)

        logger.info(
            f"RAG | kept {len(result)}/{len(all_chunks)} chunks "
            f"| scores: {[round(s, 3) for _, s in top]}"
        )
        return result

    def filter_all_questions(
        self,
        docs_by_question : Dict[str, List[Dict]],
        min_score        : Optional[float] = None,
        top_k            : Optional[int]   = None,
    ) -> Dict[str, List[Dict]]:
        """
        Run filter_docs independently for each sub-question.

        FIX C + FIX H-D: Cross-question chunk deduplication.
        If the exact same chunk (same content hash) passes for multiple
        questions, it only gets kept for the first MAX_CROSS_QUESTION_USES
        questions. This stops a single Wikipedia paragraph from filling
        many question slots, while still allowing shared foundational
        content for comparison queries where 2 questions legitimately
        need the same chunk.
        """
        MAX_CROSS_QUESTION_USES = 2  # allow chunk in up to 2 questions

        filtered_map: Dict[str, List[Dict]] = {}
        total_in  = 0
        total_out = 0

        # FIX C + FIX H-D: track hash usage counts across all questions
        global_hash_counts: Dict[str, int] = {}

        for question, docs in docs_by_question.items():
            raw_filtered = self.filter_docs(
                question  = question,
                docs      = docs,
                min_score = min_score,
                top_k     = top_k,
            )

            # FIX C + FIX H-D: cross-question dedup with soft limit
            deduped_filtered = []
            cross_dupes = 0
            for chunk in raw_filtered:
                # Reconstruct hash from content (filter_docs strips _hash from output)
                h = hashlib.md5((chunk.get("content","")).encode()).hexdigest()[:12]
                count = global_hash_counts.get(h, 0)
                if count >= MAX_CROSS_QUESTION_USES:
                    cross_dupes += 1
                    continue
                global_hash_counts[h] = count + 1
                deduped_filtered.append(chunk)

            if cross_dupes:
                logger.info(
                    f"RAG | cross-question dedup: dropped {cross_dupes} duplicate chunks "
                    f"for '{question[:50]}'"
                )

            filtered_map[question] = deduped_filtered
            total_in  += len(docs)
            total_out += len(deduped_filtered)

        low_conf = sum(
            1 for chunks in filtered_map.values()
            for c in chunks if c.get("low_confidence")
        )

        logger.info(
            f"RAG complete | {total_in} raw docs → {total_out} filtered chunks "
            f"across {len(docs_by_question)} questions | "
            f"low_confidence_chunks={low_conf}"
        )
        return filtered_map