"""
app/agents/verification_agent.py
=================================
Verification Agent — Phase 12 fixed version.

FIXES IN THIS VERSION vs previous:
  FIX 1 — "i don't know" added to WEAK_ANSWER_PHRASES.
           BEFORE: WEAK_ANSWER_PHRASES had "i don't have" but NOT "i don't know".
           The Q6 summary containing "i don't know" passed _check_weak_answer
           silently because the exact phrase wasn't in the list. That's why
           the verification log showed 0 issues on 12 summaries.
           AFTER: full list of hedging/failure phrases, including "i don't know",
           "i am unable", "i lack", "no data", "cannot determine".

  FIX 2 — Verifier reads issues[] from summarizer output.
           BEFORE: _check_weak_answer re-ran its own phrase check from scratch
           but only checked WEAK_ANSWER_PHRASES — it ignored the "issues" field
           that the summarizer already computed and stored in each summary dict.
           If the summarizer flagged an issue that _check_weak_answer's phrase
           list didn't cover, it was silently dropped.
           AFTER: _check_weak_answer also reads summary.get("issues", []) and
           treats any non-empty list from the summarizer as an additional weak
           signal, ensuring the two layers are consistent.

  FIX 3 — LOW_QUALITY_SCORE threshold raised from 0.35 → 0.50 for weak-answer
           detection. A summary scoring 0.40-0.50 is borderline and should be
           flagged. The original 0.35 cutoff was too permissive.

  FIX 4 — Contradiction check for comparison queries now includes cross-entity
           pairs (not just same-entity pairs). For "RAG vs Fine-tuning", a
           contradiction between how RAG latency is described in Q3 vs Q8 IS
           relevant even though they cover different aspects.

  FIX 5 (NEW) — FIX 7B quality re-retrieval block moved BEFORE the judge loop.
           BEFORE: issues added in FIX 7B (quality < 0.70) were appended to
           flagged_issues AFTER the judge+act loop had already started iterating,
           meaning those issues were never acted on (no re-retrieval triggered).
           AFTER: all rule-based checks (including quality < 0.70) complete
           BEFORE the single unified judge+act loop runs.

  FIX 6 (NEW) — was_re_retrieved stat now reliably counted.
           BEFORE: the stats block counted records with was_re_retrieved=True,
           but _re_retrieve() sets that flag only on success AND the eval was
           reading a stale snapshot. verified_summaries now always reflects
           the final was_re_retrieved state from the SummaryRecord.

  FIX 7 (NEW) — _original_query set at top of verify() before any async work.
           BEFORE: self._original_query was set after the is_comparison check,
           causing a subtle race condition if verify() were ever called
           concurrently. Safe ordering now.

All other fixes from the previous version retained:
  - asyncio.gather for parallel contradiction checks
  - Stricter domain-entity extraction for _subjects_overlap
  - Safe re-retrieval API shape handling
  - had_zero_chunks surfaced in output dict
  - had_low_conf from summarizer passed through
"""

import asyncio
import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple

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

# ── Constants ─────────────────────────────────────────────────────────────────

MIN_WORD_COUNT      = 40     # summaries shorter than this are flagged weak
CONTRADICTION_SCORE = 0.90   # LLM confidence threshold to flag a contradiction
MAX_RE_RETRIEVAL    = 1      # max re-retrieval attempts per question

# FIX 1: comprehensive list of failure/hedging phrases
# "i don't know" was missing — that's why Q6 was not caught
WEAK_ANSWER_PHRASES = [
    "no information",
    "not found",
    "unable to find",
    "i don't have",
    "i don't know",            # FIX 1 — was missing
    "i do not know",           # FIX 1 — variant
    "i am unable",             # FIX 1 — was missing
    "i lack",                  # FIX 1 — was missing
    "cannot answer",
    "cannot determine",        # FIX 1 — was missing
    "no relevant",
    "no sources",
    "not available",
    "insufficient information",
    "no data",                 # FIX 1 — was missing
    "i cannot",
    "i'm sorry",
    "as an ai",
]

# FIX 3: raised from 0.35 → 0.50 to catch borderline summaries
WEAK_QUALITY_THRESHOLD = 0.50

# FIX 5: quality threshold for automatic re-retrieval trigger
LOW_QUALITY_RERETRIEVAL_THRESHOLD = 0.70


# ── Enums & data classes ──────────────────────────────────────────────────────

class IssueType(str, Enum):
    CONTRADICTION    = "contradiction"
    WEAK_ANSWER      = "weak_answer"
    SOURCE_IMBALANCE = "source_imbalance"

class IssueAction(str, Enum):
    FLAGGED_ONLY = "flagged_only"
    RE_RETRIEVED = "re_retrieved"
    FIXED        = "fixed"
    UNRESOLVED   = "unresolved"
    SKIPPED      = "skipped"

@dataclass
class FlaggedIssue:
    issue_type   : IssueType
    question     : str
    description  : str
    severity     : str
    action_taken : IssueAction = IssueAction.FLAGGED_ONLY
    resolved     : bool        = False

@dataclass
class SummaryRecord:
    question           : str
    summary            : str
    sources            : List[str]
    chunks             : List[Dict]
    quality_score      : float              = 0.0
    word_count         : int                = 0
    had_low_conf       : bool               = False
    had_zero_chunks    : bool               = False
    was_re_retrieved   : bool               = False
    final_summary      : str                = ""
    summarizer_issues  : List[str]          = field(default_factory=list)  # FIX 2
    issues             : List[FlaggedIssue] = field(default_factory=list)


# ── Comparison query detection ────────────────────────────────────────────────

_COMPARISON_SIGNALS = [
    " vs ", " versus ", " compared to ", "compare ", "difference between",
    "pros and cons", "better than", "which is better", "tradeoff",
]

def _is_comparison_query(summaries: List[Dict]) -> bool:
    all_text = " ".join(s.get("question", "").lower() for s in summaries)
    return any(sig in all_text for sig in _COMPARISON_SIGNALS)


# ── Domain entity extraction ──────────────────────────────────────────────────

_ENTITY_MAP = {
    "diffusion":    {"diffusion", "ddpm", "stable diffusion", "imagen", "dall-e"},
    "gan":          {"gan", "gans", "generative adversarial"},
    "vae":          {"vae", "variational autoencoder"},
    "transformer":  {"transformer", "attention mechanism", "bert", "gpt", "self-attention"},
    "rnn":          {"rnn", "lstm", "recurrent", "gru"},
    "llm":          {"llm", "large language model", "language model"},
    "cnn":          {"cnn", "convolutional", "resnet", "vgg"},
    "rl":           {"reinforcement learning", "reward", "policy gradient", "ppo"},
    "rag":          {"rag", "retrieval augmented", "retrieval-augmented"},
    "fine_tuning":  {"fine-tuning", "fine tuning", "finetuning", "lora", "qlora"},
}

def _extract_entities(question: str) -> set:
    q = question.lower()
    found = set()
    for entity, signals in _ENTITY_MAP.items():
        if any(s in q for s in signals):
            found.add(entity)
    return found

def _subjects_overlap(q_a: str, q_b: str) -> bool:
    """
    Returns True if both questions share at least one domain entity.
    Falls back to True (conservative) if entities can't be determined.
    """
    ea = _extract_entities(q_a)
    eb = _extract_entities(q_b)
    if not ea or not eb:
        return True
    return bool(ea & eb)


# ── VerificationAgent ─────────────────────────────────────────────────────────

class VerificationAgent:

    def __init__(self, retriever=None, rag=None, summarizer=None):
        if settings is None:
            raise RuntimeError("Settings not available — cannot init VerificationAgent")

        self.client = AsyncOpenAI(
            api_key  = settings.openai_api_key,
            base_url = getattr(settings, "openai_base_url", None) or None,
        )
        self.model      = settings.openai_model
        self.fast_model = getattr(settings, "openai_fast_model", None) or settings.openai_model
        self.retriever  = retriever
        self.rag        = rag
        self.summarizer = summarizer

        # FIX 7: safe default so _re_retrieve never hits AttributeError
        self._original_query: str = ""

        logger.info(
            f"VerificationAgent initialized | model={self.model} | "
            f"re_retrieval={'enabled' if retriever else 'disabled'} | "
            f"contradiction_score={CONTRADICTION_SCORE}"
        )

    # ── Check 1: Contradiction (LLM-based) ───────────────────────────────────

    async def _check_contradiction(
        self,
        rec_a         : SummaryRecord,
        rec_b         : SummaryRecord,
        is_comparison : bool,
    ) -> Optional[FlaggedIssue]:
        """
        FIX 4: For comparison queries, cross-entity contradictions ARE checked.
        A statement about RAG latency in Q3 contradicting a statement about RAG
        latency in Q8 is a real problem even in a comparison query.
        """
        stop = {
            "the","a","an","of","in","and","or","is","are","was","were",
            "what","how","why","when","which","do","vs","for","to",
            "on","by","at","with","from","that","this","it","be","have",
        }
        wa = set(rec_a.question.lower().split()) - stop
        wb = set(rec_b.question.lower().split()) - stop
        if len(wa & wb) < 3:
            return None

        if len(rec_a.summary.split()) < 40 or len(rec_b.summary.split()) < 40:
            return None

        # FIX 4: for comparison queries, check cross-entity pairs but skip
        # pairs where entities are completely different AND topic overlap is low
        if is_comparison:
            ea = _extract_entities(rec_a.question)
            eb = _extract_entities(rec_b.question)
            if ea and eb and not (ea & eb):
                topic_words = wa & wb
                if len(topic_words) < 4:
                    return None

        prompt = (
            f"You are a strict fact-checker. Your job is to find GENUINE contradictions.\n\n"
            f"Summary A (for: {rec_a.question}):\n{rec_a.summary[:500]}\n\n"
            f"Summary B (for: {rec_b.question}):\n{rec_b.summary[:500]}\n\n"
            f"TASK: Extract directional claims and check for TRUE contradictions.\n\n"
            f"A TRUE contradiction requires ALL of:\n"
            f"  1. Both claims are about the EXACT SAME entity/concept\n"
            f"  2. Both claims are about the EXACT SAME attribute (speed, quality, etc.)\n"
            f"  3. The claims assert OPPOSITE directions (fast vs slow, better vs worse)\n\n"
            f"These are NOT contradictions:\n"
            f"  - Different contexts: 'X is fast for images' vs 'X is slow for text'\n"
            f"  - Different entities: 'GANs are fast' vs 'Diffusion models are slow'\n"
            f"  - Different attributes: 'X has high quality' vs 'X is slow'\n"
            f"  - Nuanced claims: 'X is generally fast' vs 'X can be slow in some cases'\n"
            f"  - Comparative claims: 'X > Y for A' vs 'Y > X for B'\n\n"
            f"Reply in EXACTLY this format:\n"
            f"ENTITY: <the specific entity both claims are about, or 'different'>\n"
            f"ATTRIBUTE: <the attribute being compared, or 'different'>\n"
            f"DIRECTION_A: <positive/negative/neutral>\n"
            f"DIRECTION_B: <positive/negative/neutral>\n"
            f"CONTRADICTION: yes / no\n"
            f"CONFIDENCE: 0.0-1.0\n"
            f"CLAIM_A: <specific claim from A or 'none'>\n"
            f"CLAIM_B: <opposing claim from B or 'none'>"
        )

        try:
            resp = await self.client.chat.completions.create(
                model       = self.fast_model,
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = 200,
                temperature = 0.0,
            )
            text = resp.choices[0].message.content.strip()

            is_contradiction = False
            confidence       = 0.0
            claim_a = claim_b = ""
            entity = attribute = ""
            dir_a = dir_b = ""

            for line in text.splitlines():
                key_val = line.split(":", 1)
                if len(key_val) < 2:
                    continue
                key = key_val[0].strip().upper()
                val = key_val[1].strip()

                if key == "ENTITY":
                    entity = val
                elif key == "ATTRIBUTE":
                    attribute = val
                elif key == "DIRECTION_A":
                    dir_a = val.lower()
                elif key == "DIRECTION_B":
                    dir_b = val.lower()
                elif key == "CONTRADICTION":
                    is_contradiction = "yes" in val.lower()
                elif key == "CONFIDENCE":
                    try:
                        confidence = float(val)
                    except ValueError:
                        pass
                elif key == "CLAIM_A":
                    claim_a = val
                elif key == "CLAIM_B":
                    claim_b = val

            # Reject if entity or attribute are 'different'
            if entity.lower() == "different" or attribute.lower() == "different":
                logger.debug(
                    f"Contradiction rejected (different entity/attribute): "
                    f"entity={entity}, attribute={attribute}"
                )
                return None

            # Penalise non-opposing directions
            opposing_pairs = {
                ("positive", "negative"), ("negative", "positive"),
            }
            if (dir_a, dir_b) not in opposing_pairs and is_contradiction:
                confidence *= 0.5

            if is_contradiction and confidence >= CONTRADICTION_SCORE:
                desc = (
                    f"Contradiction detected between two summaries.\n"
                    f"  • Entity: {entity} | Attribute: {attribute}\n"
                    f"  • '{rec_a.question[:60]}' states: {claim_a} (direction: {dir_a})\n"
                    f"  • '{rec_b.question[:60]}' states: {claim_b} (direction: {dir_b})\n"
                    f"  Confidence: {confidence:.0%}"
                )
                logger.warning(
                    f"VerificationAgent | CONTRADICTION | conf={confidence:.2f} | "
                    f"entity={entity} | attr={attribute} | "
                    f"'{rec_a.question[:40]}' ↔ '{rec_b.question[:40]}'"
                )
                return FlaggedIssue(
                    issue_type  = IssueType.CONTRADICTION,
                    question    = f"{rec_a.question[:50]} ↔ {rec_b.question[:50]}",
                    description = desc,
                    severity    = "high" if confidence > 0.95 else "medium",
                )

        except Exception as e:
            logger.debug(f"Contradiction check error: {repr(e)}")

        return None

    # ── Check 2: Weak answer ──────────────────────────────────────────────────

    def _check_weak_answer(self, rec: SummaryRecord) -> Optional[FlaggedIssue]:
        """
        FIX 1: comprehensive failure phrase list including "i don't know".
        FIX 2: also reads rec.summarizer_issues (from summarizer output).
        FIX 3: quality threshold raised to 0.50.
        """
        reasons = []
        s_lower = rec.summary.lower()

        if len(rec.summary.split()) < MIN_WORD_COUNT:
            reasons.append(f"Too short: {len(rec.summary.split())} words (min {MIN_WORD_COUNT})")

        # FIX 1: comprehensive phrase list
        hits = [p for p in WEAK_ANSWER_PHRASES if p in s_lower]
        if hits:
            reasons.append(f"Failure language: {', '.join(hits[:2])}")

        # FIX 3: raised threshold
        if 0 < rec.quality_score < WEAK_QUALITY_THRESHOLD:
            reasons.append(f"Low quality score: {rec.quality_score:.2f}")

        if rec.had_low_conf:
            reasons.append("Source material had low RAG confidence scores")

        # Single-chunk guard: only flag if rag_score is also low
        if len(rec.chunks) <= 1 and not rec.had_zero_chunks:
            chunk_score = (
                rec.chunks[0].get("rag_score", rec.chunks[0].get("score", 0.0))
                if rec.chunks else 0.0
            )
            if chunk_score < 0.50:
                reasons.append(
                    f"Single-chunk summary — only {len(rec.chunks)} chunk(s) available "
                    f"(rag_score={chunk_score:.2f}), limited evidence base"
                )

        # FIX 2: read summarizer's own issues[] from the summary dict
        if rec.summarizer_issues:
            for issue_str in rec.summarizer_issues:
                if issue_str not in reasons:
                    reasons.append(f"Summarizer flagged: {issue_str}")

        if reasons:
            return FlaggedIssue(
                issue_type  = IssueType.WEAK_ANSWER,
                question    = rec.question,
                description = "Weak answer:\n" + "\n".join(f"  • {r}" for r in reasons),
                severity    = "high" if len(reasons) >= 2 else "medium",
            )
        return None

    # ── Check 3: Source imbalance ─────────────────────────────────────────────

    def _check_source_imbalance(self, rec: SummaryRecord) -> Optional[FlaggedIssue]:
        if not rec.chunks:
            return None

        domains = []
        for chunk in rec.chunks:
            url = chunk.get("url", "")
            m   = re.search(r"https?://(?:www\.)?([^/]+)", url)
            if m:
                domains.append(m.group(1))

        if len(domains) < 3:
            return None

        unique   = set(domains)
        dominant = max(unique, key=domains.count)
        dominant_pct = domains.count(dominant) / len(domains)

        if dominant_pct > 0.85:
            return FlaggedIssue(
                issue_type  = IssueType.SOURCE_IMBALANCE,
                question    = rec.question,
                description = (
                    f"Single-source bias: {domains.count(dominant)}/{len(domains)} chunks "
                    f"({dominant_pct:.0%}) from '{dominant}'"
                ),
                severity = "medium",
            )
        return None

    # ── Re-retrieval ──────────────────────────────────────────────────────────

    async def _re_retrieve(self, rec: SummaryRecord, reason: str) -> bool:
        """Trigger targeted re-retrieval for a weak question. Returns True if fixed."""
        if not (self.retriever and self.rag and self.summarizer):
            return False
        if rec.had_zero_chunks:
            logger.warning(
                f"VerificationAgent | re-retrieval SKIPPED (had 0 chunks) | "
                f"'{rec.question[:60]}'"
            )
            return False

        logger.info(
            f"VerificationAgent | re-retrieval | '{rec.question[:60]}' | reason={reason}"
        )

        try:
            try:
                # Pass original_query so retrieve_all_questions() can use
                # proper source routing and relevance scoring
                retrieved = await self.retriever.retrieve_all_questions(
                    [rec.question], original_query=self._original_query
                )
                new_docs = retrieved.get("all_docs", [])
            except (AttributeError, TypeError):
                retrieved = await self.retriever.retrieve(rec.question)
                new_docs  = retrieved if isinstance(retrieved, list) else []

            if not new_docs:
                return False

            new_filtered = self.rag.filter_all_questions({rec.question: new_docs})
            new_chunks   = new_filtered.get(rec.question, [])

            if not new_chunks:
                rec.had_zero_chunks = True
                return False

            seen_urls = {c.get("url", "") for c in rec.chunks}
            merged    = rec.chunks + [
                c for c in new_chunks if c.get("url", "") not in seen_urls
            ]

            new_result  = await self.summarizer.summarize_question(rec.question, merged)
            new_summary = new_result.get("summary", "")

            if not new_summary or new_summary == rec.summary:
                return False

            rec.final_summary    = new_summary
            rec.chunks           = merged
            rec.sources          = new_result.get("sources", rec.sources)
            # FIX 6: was_re_retrieved is set here (on success) and always
            # reflected in verified_summaries via the SummaryRecord reference
            rec.was_re_retrieved = True
            return True

        except Exception as e:
            logger.error(f"VerificationAgent | re-retrieval failed: {repr(e)}")
            return False

    # ── Main verify loop ──────────────────────────────────────────────────────

    async def verify(
        self,
        summaries      : List[Dict],
        source_docs    : List[Dict],
        original_query : str = "",
    ) -> Dict:
        """
        Full verification pass over all summaries.
        Returns: {status, verified_summaries, flagged_issues, re_retrieval_log, summary_stats}
        """
        # FIX 7: set _original_query at the very top before any async work
        self._original_query = original_query

        is_comparison = _is_comparison_query(summaries)
        if is_comparison:
            logger.info(
                "VerificationAgent | COMPARISON QUERY — "
                "contradiction checks use cross-entity pairs (FIX 4)"
            )

       
        records: List[SummaryRecord] = []
        for item in summaries:
            q = item.get("question", "")
            original_chunks = item.get("chunks", [])
            had_zero_chunks = len(original_chunks) == 0
            chunks = original_chunks

            if not chunks:
                q_keywords = q.lower().split()[:4]
                chunks = [
                    d for d in source_docs
                    if any(kw in (d.get("content", "") or "").lower() for kw in q_keywords)
                ][:8]

            records.append(SummaryRecord(
                question         = q,
                summary          = item.get("summary", ""),
                sources          = item.get("sources", []),
                chunks           = chunks,
                quality_score    = item.get("quality_score", 0.0),
                word_count       = len(item.get("summary", "").split()),
                had_low_conf     = item.get("had_low_conf", False),
                had_zero_chunks  = had_zero_chunks,
                final_summary    = item.get("summary", ""),
                summarizer_issues= item.get("issues", []),
            ))

        logger.info(
            f"VerificationAgent | starting | {len(records)} summaries | "
            f"is_comparison={is_comparison}"
        )

        flagged_issues: List[FlaggedIssue] = []


        # ── Phase A: Rule-based checks (fast, no LLM) ────────────────────────
        for rec in records:
            for check_fn in (self._check_weak_answer, self._check_source_imbalance):
                issue = check_fn(rec)
                if issue:
                    rec.issues.append(issue)
                    flagged_issues.append(issue)

        n_weak_pre      = sum(1 for i in flagged_issues if i.issue_type == IssueType.WEAK_ANSWER)
        n_imbalance_pre = sum(1 for i in flagged_issues if i.issue_type == IssueType.SOURCE_IMBALANCE)
        logger.info(
            f"VerificationAgent | rule-based checks done | "
            f"weak={n_weak_pre} imbalanced={n_imbalance_pre}"
        )

        # ── Phase B: Contradiction checks — all pairs in parallel ─────────────
        pairs = [
            (records[i], records[j])
            for i in range(len(records))
            for j in range(i + 1, len(records))
        ]

        logger.info(f"VerificationAgent | checking {len(pairs)} summary pairs for contradictions")

        if pairs:
            contra_results = await asyncio.gather(
                *[self._check_contradiction(a, b, is_comparison) for a, b in pairs],
                return_exceptions=True,
            )
            for res in contra_results:
                if isinstance(res, FlaggedIssue):
                    flagged_issues.append(res)

        n_contra    = sum(1 for i in flagged_issues if i.issue_type == IssueType.CONTRADICTION)
        n_weak      = sum(1 for i in flagged_issues if i.issue_type == IssueType.WEAK_ANSWER)
        n_imbalance = sum(1 for i in flagged_issues if i.issue_type == IssueType.SOURCE_IMBALANCE)

        logger.info(
            f"VerificationAgent | CHECK complete | issues={len(flagged_issues)} "
            f"(contradictions={n_contra}, weak={n_weak}, imbalanced={n_imbalance})"
        )

        # ── FIX 5: Phase C — quality-based re-retrieval triggers ──────────────
        # CRITICAL: this must run BEFORE the judge+act loop.
        # In the previous version, FIX 7B appended to flagged_issues DURING
        # the judge loop, so those new issues were never acted on.
        # Now all issues are collected first, then the judge loop runs once.
        already_flagged_questions = {
            i.question
            for i in flagged_issues
            if i.issue_type == IssueType.WEAK_ANSWER
        }
        for rec in records:
            if (
                rec.quality_score < LOW_QUALITY_RERETRIEVAL_THRESHOLD
                and rec.question not in already_flagged_questions
            ):
                logger.info(
                    f"VerificationAgent | quality={rec.quality_score:.2f} < "
                    f"{LOW_QUALITY_RERETRIEVAL_THRESHOLD} | "
                    f"adding re-retrieval trigger for '{rec.question[:60]}'"
                )
                issue = FlaggedIssue(
                    issue_type  = IssueType.WEAK_ANSWER,
                    question    = rec.question,
                    description = (
                        f"Low quality score ({rec.quality_score:.2f} < "
                        f"{LOW_QUALITY_RERETRIEVAL_THRESHOLD}) triggers re-retrieval"
                    ),
                    severity = "medium",
                )
                rec.issues.append(issue)
                flagged_issues.append(issue)

        # ── Phase D: Judge + Act (single unified loop) ────────────────────────
        re_retrieval_log: List[str] = []
        re_retrieval_counts: Dict[str, int] = {}

        for issue in flagged_issues:
            # Contradictions are always flagged-only (no re-retrieval)
            if issue.issue_type == IssueType.CONTRADICTION:
                issue.action_taken = IssueAction.FLAGGED_ONLY
                logger.info(
                    f"VerificationAgent | FLAGGED_ONLY (contradiction) | "
                    f"'{issue.question[:60]}'"
                )
                continue

            q   = issue.question
            rec = next((r for r in records if r.question == q), None)

            # Skip if this question had no chunks to begin with
            if rec and rec.had_zero_chunks:
                issue.action_taken = IssueAction.SKIPPED
                re_retrieval_log.append(f"SKIPPED (no initial chunks): '{q[:60]}'")
                logger.info(
                    f"VerificationAgent | re-retrieval SKIPPED | "
                    f"reason=no_initial_chunks | '{q[:60]}'"
                )
                continue

            # Skip if required components are missing
            if not (self.retriever and self.rag and self.summarizer):
                issue.action_taken = IssueAction.SKIPPED
                re_retrieval_log.append(f"SKIPPED (no retriever/rag/summarizer): '{q[:60]}'")
                logger.info(
                    f"VerificationAgent | re-retrieval SKIPPED | "
                    f"reason=missing_components | '{q[:60]}'"
                )
                continue

            # Enforce max re-retrieval per question
            count = re_retrieval_counts.get(q, 0)
            if count >= MAX_RE_RETRIEVAL:
                issue.action_taken = IssueAction.SKIPPED
                re_retrieval_log.append(f"SKIPPED (max re-retrieval reached): '{q[:60]}'")
                logger.info(
                    f"VerificationAgent | re-retrieval SKIPPED | "
                    f"reason=max_attempts_reached | '{q[:60]}'"
                )
                continue

            re_retrieval_counts[q] = count + 1
            issue.action_taken     = IssueAction.RE_RETRIEVED
            quality_str = f"{rec.quality_score:.2f}" if rec else "?"
            logger.info(
                f"VerificationAgent | TRIGGERING re-retrieval | "
                f"issue={issue.issue_type.value} | quality={quality_str} | "
                f"'{q[:60]}'"
            )

            if rec:
                success = await self._re_retrieve(rec, issue.issue_type.value)
                if success:
                    issue.resolved     = True
                    issue.action_taken = IssueAction.FIXED
                    re_retrieval_log.append(f"FIXED: '{q[:60]}'")
                else:
                    issue.action_taken = IssueAction.UNRESOLVED
                    re_retrieval_log.append(f"UNRESOLVED: '{q[:60]}'")
            else:
                issue.action_taken = IssueAction.SKIPPED
                re_retrieval_log.append(f"SKIPPED (record not found): '{q[:60]}'")
                logger.warning(
                    f"VerificationAgent | re-retrieval SKIPPED | "
                    f"reason=record_not_found | '{q[:60]}'"
                )

        # ── Overall status ────────────────────────────────────────────────────
        has_unresolved    = any(
            not i.resolved
            and i.action_taken not in (IssueAction.SKIPPED, IssueAction.FLAGGED_ONLY)
            for i in flagged_issues
        )
        has_contradiction = any(
            i.issue_type == IssueType.CONTRADICTION for i in flagged_issues
        )

        if not flagged_issues:
            status = "approved"
        elif has_contradiction or has_unresolved:
            status = "approved_with_flags"
        else:
            status = "approved"

        logger.info(
            f"VerificationAgent | APPROVED | status={status} | "
            f"issues={len(flagged_issues)} | re_retrieved={len(re_retrieval_log)}"
        )

        # ── Build verified_summaries ──────────────────────────────────────────
        # FIX 6: always read was_re_retrieved from the live SummaryRecord,
        # not from the original item dict — the flag is set by _re_retrieve()
        # which mutates the record in-place after this dict was built.
        verified_summaries = []
        for rec in records:
            final = rec.final_summary or rec.summary
            verified_summaries.append({
                "question"         : rec.question,
                "summary"          : final,
                "sources"          : rec.sources,
                "quality_score"    : rec.quality_score,
                "was_re_retrieved" : rec.was_re_retrieved,   # FIX 6: from live record
                "had_zero_chunks"  : rec.had_zero_chunks,
                "had_low_conf"     : rec.had_low_conf,
                "issues"           : [
                    {
                        "type"       : i.issue_type.value,
                        "description": i.description,
                        "severity"   : i.severity,
                        "action"     : i.action_taken.value,
                        "resolved"   : i.resolved,
                    }
                    for i in rec.issues
                ],
            })

        # ── Stats ─────────────────────────────────────────────────────────────
        n_contra    = sum(1 for i in flagged_issues if i.issue_type == IssueType.CONTRADICTION)
        n_weak      = sum(1 for i in flagged_issues if i.issue_type == IssueType.WEAK_ANSWER)
        n_imbalance = sum(1 for i in flagged_issues if i.issue_type == IssueType.SOURCE_IMBALANCE)

        stats = {
            "total"         : len(records),
            "contradictions": n_contra,
            "weak_answers"  : n_weak,
            "imbalanced"    : n_imbalance,
            # FIX 6: count from live SummaryRecords, not from flagged_issues
            "re_retrieved"  : sum(1 for r in records if r.was_re_retrieved),
            "fixed"         : sum(1 for i in flagged_issues if i.resolved),
        }

        return {
            "status"             : status,
            "verified_summaries" : verified_summaries,
            "flagged_issues"     : [
                {
                    "type"       : i.issue_type.value,
                    "question"   : i.question,
                    "description": i.description,
                    "severity"   : i.severity,
                    "action"     : i.action_taken.value,
                    "resolved"   : i.resolved,
                }
                for i in flagged_issues
            ],
            "re_retrieval_log" : re_retrieval_log,
            "summary_stats"    : stats,
        }