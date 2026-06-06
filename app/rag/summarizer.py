"""
app/rag/summarizer.py
=====================
Summarization Agent — Phase 12 fixed version.

FIXES IN THIS VERSION vs the previous:
  FIX 1 — Retry fires on non-empty issues[] regardless of numeric quality score.
           BEFORE: retry condition was `if quality >= MIN_QUALITY_SCORE: break`
           meaning Q6 with quality=0.70 (above 0.60 threshold) + issues=
           ["Contains failure language: i don't know"] would pass on attempt 1
           and never retry. The "i don't know" summary entered the report.
           AFTER: retry condition is now:
               if quality >= MIN_QUALITY_SCORE and not issues: break
           Any summary with non-empty issues[] is ALWAYS retried regardless
           of its numeric score.

  FIX 2 — issues[] stored in the output dict.
           BEFORE: issues were logged but not included in the returned dict,
           so verifier_node's `issues` field check always saw [].
           AFTER: output dict includes "issues": issues so nodes.py
           verifier_node can read and tag low_confidence correctly.

  FIX 3 — LOW_QUALITY_FLAG threshold added (0.80).
           Summaries that pass the numeric threshold BUT have a score below
           0.80 are flagged as borderline so the verifier can inspect them.
           This catches the 0.70 case without blocking it.

FIX 4 — MIN_SOURCE_COUNT now actually enforced in self-eval.
           Summaries grounded in fewer than 2 distinct sources are penalized
           and marked as structural so retrying does not waste attempts.

All other fixes from the previous version retained:
  - 429 / quota backoff (_llm_call with exponential backoff)
  - Self-eval token savings (skip LLM for clear pass/fail)
  - CONCURRENCY=1 to avoid 429 cascade
  - INTER_QUESTION_DELAY between questions
  - Chunk fallback when all LLM calls fail
  - repr(e) in error logs
"""

import asyncio
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

try:
    from openai import AsyncOpenAI
    from config.settings import get_settings
    from core.logger import setup_logger
    settings = get_settings()
    logger   = setup_logger(__name__)
except ModuleNotFoundError:
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    settings = None

# ── Config ────────────────────────────────────────────────────────────────────

MAX_CONTEXT_CHARS    = 10_000   # ~2500 tokens — safe for 8B models
MIN_QUALITY_SCORE    = 0.60     # minimum acceptable quality score
LOW_QUALITY_FLAG     = 0.80     # below this → flagged borderline even if passing
MAX_RETRIES          = 2
MIN_SUMMARY_WORDS    = 80
CONCURRENCY          = 1        # sequential to avoid 429 cascade
INTER_QUESTION_DELAY = 2.0      # seconds between questions

# FIX 6A: Stricter quality gate thresholds
MIN_QUALITY_WORD_COUNT = 100    # summaries with fewer words are penalized
MIN_SOURCE_COUNT       = 2      # summaries grounded in <2 sources are penalized
MAX_COSINE_CAP_QUALITY = 0.70   # FIX 6B: quality capped here if top chunk cosine < 0.50
COSINE_THRESHOLD       = 0.50   # FIX 6B: below this → cap quality

# FIX H-A: Structural issue prefix — issues the LLM cannot fix by retrying.
# The retry loop should NOT count these when deciding whether to retry.
_STRUCTURAL_PREFIX = "[STRUCTURAL] "

def _is_retryable_issue(issue: str) -> bool:
    """Return True if the issue can potentially be fixed by regenerating the summary."""
    return not issue.startswith(_STRUCTURAL_PREFIX)

# FIX 6C: Known factual constraints (ground truths for AI/ML domain)
# Summaries contradicting any of these receive a penalty.
_FACTUAL_CONSTRAINTS = [
    # Format: (contradiction_pattern, correction_note)
    ("rag replaces fine-tuning", "RAG and fine-tuning are complementary, not replacements"),
    ("rag eliminates hallucination", "RAG reduces but does not eliminate hallucinations"),
    ("transformers were invented by google in 2018", "Transformers were introduced in 2017"),
    ("gpt-4 is open source", "GPT-4 is proprietary/closed-source"),
    ("bert is a generative model", "BERT is a discriminative/masked language model"),
    ("diffusion models are a type of gan", "Diffusion models are distinct from GANs"),
    ("fine-tuning requires no data", "Fine-tuning requires task-specific training data"),
    ("rag does not need a retriever", "RAG by definition requires a retrieval component"),
]

# 429 retry config
_QUOTA_RETRY_DELAYS = [5, 15, 45]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return (
        "429" in msg
        or "quota" in msg
        or "rate_limit" in msg
        or "too_many_tokens" in msg
        or "token_quota" in msg
        or "queue_exceeded" in msg
        or "high traffic" in msg
    )


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SummaryResult:
    question       : str
    summary        : str
    sources        : List[str]
    quality_score  : float
    attempts       : int
    word_count     : int
    status         : str
    issues         : List[str] = field(default_factory=list)   # FIX 2
    had_low_conf   : bool = False


# ── SummarizerAgent ───────────────────────────────────────────────────────────

class SummarizerAgent:
    """
    Generates a focused, grounded answer for each sub-question
    using RAG-filtered chunks as context.

    Loop per question:
      1. Build context from top chunks
      2. Call LLM with 429 backoff → generate summary
      3. Self-evaluate (rule-based; LLM only for borderline scores)
      4. FIX 1: retry if quality < threshold OR issues[] non-empty
      5. If all LLM calls fail → build answer from raw chunks (fallback)
      6. Return best result regardless of score
    """

    def __init__(self):
        if settings is None:
            raise RuntimeError("Settings not available — cannot init SummarizerAgent")

        self.client = AsyncOpenAI(
            api_key  = settings.openai_api_key,
            base_url = getattr(settings, "openai_base_url", None) or None,
        )
        self.model      = settings.openai_model
        self.fast_model = (
            getattr(settings, "openai_fast_model", None)
            or settings.openai_model
        )
        logger.info(
            f"SummarizerAgent initialized | model={self.model} | "
            f"eval_model={self.fast_model} | max_retries={MAX_RETRIES} | "
            f"min_quality={MIN_QUALITY_SCORE}"
        )

    # ── Context builder ───────────────────────────────────────────────────────

    def _build_context(
        self,
        chunks   : List[Dict],
        max_chars: int = MAX_CONTEXT_CHARS,
    ) -> Tuple[str, List[str], bool]:
        """Returns (context_text, source_urls, had_low_confidence)."""
        sorted_chunks = sorted(
            chunks,
            key     = lambda c: c.get("rag_score", c.get("score", 0)),
            reverse = True,
        )

        parts: List[str]   = []
        sources: List[str] = []
        total_chars  = 0
        had_low_conf = False

        for chunk in sorted_chunks:
            text  = (chunk.get("content") or "").strip()
            title = chunk.get("title", "Unknown source")
            url   = chunk.get("url",   "")
            score = chunk.get("rag_score", chunk.get("score", 0.0))

            if not text:
                continue
            if total_chars + len(text) > max_chars:
                break

            if chunk.get("low_confidence"):
                had_low_conf = True

            parts.append(f"[Source: {title} | Relevance: {score:.2f}]\n{text}")
            total_chars += len(text)

            if url and url not in sources:
                sources.append(url)

        return "\n\n---\n\n".join(parts), sources, had_low_conf

    # ── Chunk fallback (no LLM) ───────────────────────────────────────────────

    def _chunk_fallback(self, question: str, chunks: List[Dict]) -> Tuple[str, List[str]]:
        """Build a readable answer from raw chunk text when all LLM calls fail."""
        sorted_chunks = sorted(
            chunks,
            key     = lambda c: c.get("rag_score", c.get("score", 0)),
            reverse = True,
        )[:3]

        parts   = []
        sources = []
        for chunk in sorted_chunks:
            text = (chunk.get("content") or "").strip()
            if text:
                parts.append(text[:600])
            url = chunk.get("url", "")
            if url and url not in sources:
                sources.append(url)

        if not parts:
            return f"No summarized content available for: {question}", []

        combined = "\n\n".join(parts)
        prefix   = "[Auto-extracted from sources — LLM quota exceeded]\n\n"
        return prefix + combined, sources

    # ── LLM call with 429 backoff ─────────────────────────────────────────────

    async def _llm_call(self, prompt: str, label: str = "LLM") -> str:
        """Call the LLM with exponential backoff on rate-limit errors."""
        last_exc: Exception | None = None

        for delay in [0] + _QUOTA_RETRY_DELAYS:
            if delay:
                logger.warning(
                    f"SummarizerAgent | {label} | 429/quota — waiting {delay}s before retry"
                )
                await asyncio.sleep(delay)
            try:
                response = await self.client.chat.completions.create(
                    model       = self.model,
                    messages    = [{"role": "user", "content": prompt}],
                    max_tokens  = 450,
                    temperature = 0.15,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if _is_rate_limit_error(e):
                    last_exc = e
                    logger.warning(
                        f"SummarizerAgent | {label} | rate-limit: {repr(e)}"
                    )
                    continue
                logger.error(f"SummarizerAgent | {label} | error: {repr(e)}")
                raise

        raise last_exc  # type: ignore[misc]

    # ── Generate summary ──────────────────────────────────────────────────────

    async def _generate(
        self,
        question : str,
        chunks   : List[Dict],
        retry    : bool = False,
    ) -> Tuple[str, List[str], bool]:
        """Returns (summary_text, source_urls, had_low_confidence)."""
        max_chars = MAX_CONTEXT_CHARS if not retry else int(MAX_CONTEXT_CHARS * 1.4)
        context, sources, had_low_conf = self._build_context(chunks, max_chars)

        if not context:
            return "", [], False

        if not retry:
            prompt = (
                f"You are a research analyst. Using ONLY the sources below, "
                f"answer this research question in a clear, factual paragraph "
                f"(aim for 150–220 words).\n\n"
                f"Research Question: {question}\n\n"
                f"Sources:\n{context}\n\n"
                f"Instructions:\n"
                f"- Answer directly and specifically\n"
                f"- Use ONLY the provided sources — no outside knowledge\n"
                f"- Include key facts, numbers, or findings from the sources\n"
                f"- End with the most important insight or conclusion\n"
                f"- Do NOT say 'I don't know' or 'I cannot answer' — use whatever "
                f"  relevant information IS in the sources\n\n"
                f"Answer:"
            )
        else:
            prompt = (
                f"You are a research analyst. Your previous answer for the question below "
                f"was not specific enough or contained uncertainty language. "
                f"Use ALL the source excerpts provided to write a better answer.\n\n"
                f"Research Question: {question}\n\n"
                f"Source excerpts (use ALL of them):\n{context}\n\n"
                f"Write a comprehensive, factual answer (150–220 words) that:\n"
                f"- Directly addresses every aspect of the question\n"
                f"- Cites specific facts, numbers, or examples from the sources\n"
                f"- Does NOT add information not in the sources\n"
                f"- Does NOT use phrases like 'I don't know', 'I cannot', "
                f"  'no information available' — synthesise what IS there\n\n"
                f"Answer:"
            )

        label = f"generate attempt={'retry' if retry else '1'} | '{question[:40]}'"
        text  = await self._llm_call(prompt, label=label)
        return text, sources, had_low_conf

    # ── Self-evaluation (token-efficient) ─────────────────────────────────────

    async def _evaluate(
        self,
        question  : str,
        summary   : str,
        chunks    : List[Dict] = None,
    ) -> Tuple[float, List[str]]:
        """
        Returns (quality_score 0.0-1.0, list_of_issues).

        FIX 1 context: issues[] being non-empty is what triggers retry
        in the outer loop, regardless of the numeric score.

        FIX 6A: Multi-factor quality checks (word count, source count, specificity).
        FIX 6B: Quality capped at 0.70 if top chunk cosine < 0.50.
        FIX 6C: Factual constraint check.

        LLM call skipped when:
          - rule-based score >= 0.85  (clear pass — save tokens)
          - rule-based score <= 0.40  (clear fail — LLM won't help, retry faster)
        """
        issues: List[str] = []
        score  = 1.0

        # Rule 1: Minimum length
        word_count = len(summary.split())
        if word_count < MIN_SUMMARY_WORDS:
            issues.append(f"Too short: {word_count} words (min {MIN_SUMMARY_WORDS})")
            score -= 0.3

        # FIX 6A: Stricter word count check
        if word_count < MIN_QUALITY_WORD_COUNT and word_count >= MIN_SUMMARY_WORDS:
            issues.append(
                f"Below quality threshold: {word_count} words (preferred >= {MIN_QUALITY_WORD_COUNT})"
            )
            score -= 0.1

        # Rule 2: Failure/hedging language — comprehensive list
        failure_phrases = [
            "i don't have",
            "i cannot",
            "no information",
            "not found",
            "unable to",
            "i'm sorry",
            "as an ai",
            "i don't know",         # ← was missing in original, caused Q6 bug
            "cannot answer",
            "no relevant",
            "no sources",
            "not available",
            "llm quota",
            "quota exceeded",
            "i am unable",
            "i lack",
            "insufficient information",
        ]
        s_lower = summary.lower()
        hits = [p for p in failure_phrases if p in s_lower]
        if hits:
            issues.append(f"Contains failure language: {', '.join(hits[:2])}")
            score -= 0.3

        # FIX 6A: Specificity check — vague, generic summaries are penalized
        vague_patterns = [
            "various studies", "many researchers", "it is widely known",
            "there are many", "generally speaking", "in general terms",
            "numerous approaches", "the field has seen",
        ]
        vague_hits = [p for p in vague_patterns if p in s_lower]
        if len(vague_hits) >= 2:
            issues.append(f"Too vague/generic: uses {len(vague_hits)} hedge phrases")
            score -= 0.1

        # FIX 6A: Minimum distinct source count — summaries backed by too few
        # unique sources are structurally weak and should not be retried blindly.
        if chunks:
            unique_sources = {
                (c.get("url") or c.get("source_url") or c.get("source") or c.get("title") or "").strip().lower()
                for c in chunks
                if (c.get("url") or c.get("source_url") or c.get("source") or c.get("title"))
            }
            unique_sources.discard("")
            if len(unique_sources) < MIN_SOURCE_COUNT:
                issues.append(
                    f"{_STRUCTURAL_PREFIX}Too few distinct sources: {len(unique_sources)} "
                    f"(min {MIN_SOURCE_COUNT})"
                )
                score -= 0.15

        # FIX 6C: Factual constraint check
        for contradiction, correction in _FACTUAL_CONSTRAINTS:
            if contradiction in s_lower:
                issues.append(f"Factual error: '{contradiction}' — {correction}")
                score -= 0.3
                logger.warning(
                    f"SummarizerAgent | factual violation: '{contradiction}' in summary "
                    f"for '{question[:40]}'"
                )

        # FIX 6B: Cap quality if top chunk cosine score is low
        # FIX H-A: Mark as STRUCTURAL — retrying won't change the chunks.
        if chunks:
            top_cosine = max(
                (c.get("rag_score", c.get("score", 0.0)) for c in chunks),
                default=0.0,
            )
            if top_cosine < COSINE_THRESHOLD:
                if score > MAX_COSINE_CAP_QUALITY:
                    issues.append(
                        f"{_STRUCTURAL_PREFIX}Quality capped at {MAX_COSINE_CAP_QUALITY:.2f}: "
                        f"top chunk cosine={top_cosine:.2f} < {COSINE_THRESHOLD}"
                    )
                    score = min(score, MAX_COSINE_CAP_QUALITY)
                    logger.info(
                        f"SummarizerAgent | cosine cap applied | top_cosine={top_cosine:.2f} "
                        f"→ quality capped at {MAX_COSINE_CAP_QUALITY}"
                    )

        rule_score = round(max(0.0, min(1.0, score)), 2)

        # Borderline quality flag: pass threshold but still not fully confident.
        if MIN_QUALITY_SCORE <= rule_score < LOW_QUALITY_FLAG:
            issues.append(
                f"{_STRUCTURAL_PREFIX}Borderline quality: {rule_score:.2f} < {LOW_QUALITY_FLAG:.2f}"
            )

        # Skip LLM eval for clear pass or clear fail
        if rule_score >= 0.85 or rule_score <= 0.40:
            return rule_score, issues

        # Borderline: run LLM relevance check
        try:
            eval_prompt = (
                f"Rate how well this answer addresses the question.\n"
                f"Question: {question}\n"
                f"Answer: {summary[:600]}\n\n"
                f"Reply with ONE number from 0 to 10 only. No other text."
            )
            resp = await self.client.chat.completions.create(
                model       = self.fast_model,
                messages    = [{"role": "user", "content": eval_prompt}],
                max_tokens  = 10,
                temperature = 0.0,
            )
            raw  = resp.choices[0].message.content.strip()
            nums = [float(x) for x in raw.replace("/", " ").split() if _is_number(x)]
            if nums:
                relevance = min(nums[0], 10.0) / 10.0
                score = min(rule_score, 0.4 + relevance * 0.6)
                if relevance < 0.5:
                    issues.append(f"Low relevance to question (score: {relevance:.1f}/1.0)")
        except Exception as e:
            logger.debug(f"Self-eval LLM call failed: {repr(e)} — rule-based score used")

        return round(max(0.0, min(1.0, score)), 2), issues

    # ── Single-question agent loop ────────────────────────────────────────────

    async def run(self, question: str, chunks: List[Dict]) -> SummaryResult:
        """Run summarization for one question. Returns SummaryResult."""
        logger.info(
            f"SummarizerAgent | question='{question[:60]}' | chunks={len(chunks)}"
        )

        if not chunks:
            logger.warning(f"SummarizerAgent | no chunks for '{question[:60]}'")
            return SummaryResult(
                question      = question,
                summary       = "No relevant information found for this sub-question.",
                sources       = [],
                quality_score = 0.0,
                attempts      = 0,
                word_count    = 0,
                status        = "no_data",
                issues        = ["No chunks available"],
                had_low_conf  = False,
            )

        best_summary  = ""
        best_score    = 0.0
        best_sources: List[str]  = []
        best_issues:  List[str]  = []
        had_low_conf  = False
        final_status  = "failed"
        quota_failed  = False
        attempts_made = 0

        for attempt in range(1, MAX_RETRIES + 1):
            attempts_made = attempt
            logger.info(
                f"SummarizerAgent | attempt {attempt}/{MAX_RETRIES} | "
                f"broader={'yes' if attempt > 1 else 'no'}"
            )
            try:
                summary, sources, low_conf = await self._generate(
                    question, chunks, retry=(attempt > 1)
                )
                had_low_conf = had_low_conf or low_conf

                if not summary:
                    continue

                quality, issues = await self._evaluate(question, summary, chunks=chunks)

                logger.info(
                    f"SummarizerAgent | attempt {attempt} | quality={quality:.2f} | "
                    f"words={len(summary.split())} | issues={issues}"
                )

                if quality > best_score:
                    best_score   = quality
                    best_summary = summary
                    best_sources = sources
                    best_issues  = issues

                # ── FIX 1 + FIX H-A: retry condition ─────────────────────────
                # Only retry on RETRYABLE issues (failure language, too short, etc.)
                # STRUCTURAL issues (cosine cap, low source count) cannot be
                # fixed by regenerating the summary — don't waste an LLM call.
                retryable_issues = [i for i in issues if _is_retryable_issue(i)]
                if quality >= MIN_QUALITY_SCORE and not retryable_issues:
                    final_status = "done"
                    if issues:  # has only structural issues
                        final_status = "done_with_caveats"
                    logger.info(
                        f"SummarizerAgent | DONE | '{question[:60]}' | "
                        f"quality={quality:.2f} | attempts={attempt}"
                        + (f" | structural_issues={len(issues) - len(retryable_issues)}" if issues else "")
                    )
                    break
                elif quality >= MIN_QUALITY_SCORE and retryable_issues:
                    # Score is acceptable but has retryable issues — retry
                    logger.info(
                        f"SummarizerAgent | quality={quality:.2f} passes threshold "
                        f"but retryable issues present {retryable_issues} — retrying"
                    )
                    # Don't break — continue to next attempt
                # else: quality below threshold — also continues to next attempt

            except Exception as e:
                logger.error(
                    f"SummarizerAgent | attempt {attempt} failed: {repr(e)}"
                )
                if _is_rate_limit_error(e):
                    quota_failed = True
                if attempt >= MAX_RETRIES:
                    break

        # After all attempts: determine final status
        retryable_remaining = [i for i in best_issues if _is_retryable_issue(i)]
        structural_only = best_issues and not retryable_remaining
        borderline_quality = bool(best_summary and MIN_QUALITY_SCORE <= best_score < LOW_QUALITY_FLAG)

        if best_summary and best_score >= MIN_QUALITY_SCORE and not best_issues and not borderline_quality:
            final_status = "done"
        elif best_summary and best_score >= MIN_QUALITY_SCORE and (structural_only or borderline_quality) and not retryable_remaining:
            # Passed threshold but only structural/borderline issues remain (can't be fixed by retry)
            final_status = "done_with_caveats"
            logger.info(
                f"SummarizerAgent | DONE_WITH_CAVEATS | '{question[:60]}' | "
                f"quality={best_score:.2f} | structural_issues={best_issues}"
            )
        elif best_summary and best_score >= MIN_QUALITY_SCORE and best_issues:
            # Best we could do — passed threshold but retryable issues remain
            final_status = "done_with_issues"
            logger.warning(
                f"SummarizerAgent | DONE_WITH_ISSUES | '{question[:60]}' | "
                f"quality={best_score:.2f} | issues={best_issues}"
            )
        elif best_summary:
            final_status = "below_threshold"

        # If LLM failed entirely, build answer from raw chunks
        if not best_summary:
            logger.warning(
                f"SummarizerAgent | all LLM attempts failed for '{question[:60]}' "
                f"— using chunk fallback"
            )
            fallback_text, fallback_sources = self._chunk_fallback(question, chunks)
            best_summary  = fallback_text
            best_sources  = fallback_sources
            best_score    = 0.2
            best_issues   = ["LLM unavailable — auto-extracted from chunks"]
            final_status  = "quota_fallback" if quota_failed else "llm_failed"

        had_low_conf = had_low_conf or (best_summary and best_score < LOW_QUALITY_FLAG)

        return SummaryResult(
            question      = question,
            summary       = best_summary,
            sources       = best_sources,
            quality_score = best_score,
            attempts      = attempts_made,
            word_count    = len(best_summary.split()),
            status        = final_status,
            issues        = best_issues,   # FIX 2: included in result
            had_low_conf  = had_low_conf,
        )

    # ── Run all questions ─────────────────────────────────────────────────────

    async def run_all(self, filtered_map: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Run summarization for all sub-questions sequentially (CONCURRENCY=1)
        to prevent 429 cascade on quota-limited APIs.
        """
        logger.info(
            f"SummarizerAgent | starting {len(filtered_map)} questions | "
            f"concurrency={CONCURRENCY} | max_retries={MAX_RETRIES} | "
            f"min_quality={MIN_QUALITY_SCORE}"
        )

        sem = asyncio.Semaphore(CONCURRENCY)

        async def _run_one(idx: int, question: str, chunks: List[Dict]) -> Dict:
            async with sem:
                if idx > 1:
                    await asyncio.sleep(INTER_QUESTION_DELAY)
                logger.info(
                    f"SummarizerAgent | [{idx}/{len(filtered_map)}] '{question[:60]}'"
                )
                result = await self.run(question, chunks)
                # FIX 2: issues[] included in output dict
                return {
                    "question":      result.question,
                    "summary":       result.summary,
                    "sources":       result.sources,
                    "quality_score": result.quality_score,
                    "attempts":      result.attempts,
                    "word_count":    result.word_count,
                    "status":        result.status,
                    "issues":        result.issues,       # FIX 2
                    "had_low_conf":  result.had_low_conf,
                }

        tasks = [
            _run_one(i, q, chunks)
            for i, (q, chunks) in enumerate(filtered_map.items(), 1)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        clean_results = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                q = list(filtered_map.keys())[i]
                logger.error(f"SummarizerAgent | question {i} gather failed: {repr(res)}")
                clean_results.append({
                    "question":      q,
                    "summary":       "Summarization failed due to an error.",
                    "sources":       [],
                    "quality_score": 0.0,
                    "attempts":      MAX_RETRIES,
                    "word_count":    0,
                    "status":        "failed",
                    "issues":        ["gather exception"],
                    "had_low_conf":  False,
                })
            else:
                clean_results.append(res)

        avg_quality = sum(r["quality_score"] for r in clean_results) / max(len(clean_results), 1)
        avg_words   = sum(r["word_count"]    for r in clean_results) / max(len(clean_results), 1)
        retried     = sum(1 for r in clean_results if r["attempts"] > 1)
        fallbacks   = sum(1 for r in clean_results if r["status"] in ("quota_fallback", "llm_failed"))
        with_issues = sum(1 for r in clean_results if r.get("issues"))

        logger.info(
            f"SummarizerAgent complete | summaries={len(clean_results)} | "
            f"avg_quality={avg_quality:.2f} | avg_words={avg_words:.0f} | "
            f"retried={retried}/{len(filtered_map)} | fallbacks={fallbacks} | "
            f"with_issues={with_issues}"
        )
        return clean_results

    # ── Backward-compat aliases ───────────────────────────────────────────────

    async def summarize_all(self, filtered_map: Dict[str, List[Dict]]) -> List[Dict]:
        return await self.run_all(filtered_map)

    async def summarize_question(self, question: str, chunks: List[Dict]) -> Dict:
        result = await self.run(question, chunks)
        return {
            "question":      result.question,
            "summary":       result.summary,
            "sources":       result.sources,
            "quality_score": result.quality_score,
            "attempts":      result.attempts,
            "word_count":    result.word_count,
            "status":        result.status,
            "issues":        result.issues,
            "had_low_conf":  result.had_low_conf,
        }


# Backward-compat alias
Summarizer = SummarizerAgent