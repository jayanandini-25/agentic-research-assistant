"""
Deep Research Agent — Phase 12 (Gap-Targeted)
=====================================================================
File location:  app/agents/deep_research_agent.py

KEY CHANGES vs previous version:

  FIX 1 — run() now accepts coverage_gaps: List[str]
           When coverage_gaps is provided (from depth_judge_node),
           root questions are generated specifically to fill those
           gaps instead of general exploration of the topic.

  FIX 2 — _generate_gap_questions() added.
           Generates ROOT_QUESTIONS questions specifically targeting
           the uncovered dimensions identified by depth_judge_node.
           Falls back to _generate_questions() if gaps is empty.

  FIX 3 — _identify_subtopic() now avoids sub-topics that overlap
           with coverage_gaps already being researched at root level,
           preventing double-research of the same gap.

  FIX 4 — Child node questions are also gap-aware when coverage_gaps
           remain after root research. After root pipeline runs,
           re-check which gaps are still uncovered and assign them
           to the child node.

  FIX 10A — Hard cap of MAX_CHILD_QUESTIONS (5) per child node.
  FIX 10B — Parent-child deduplication: child questions whose
            normalized tokens overlap >60% with any parent question
            are dropped before the child pipeline runs.
            Uses Jaccard similarity (intersection / union) so short
            child questions don't get artificially high overlap scores.

  FIX D1 (NEW) — Gap question parser less aggressive.
            BEFORE: rejected questions < 15 chars, and rejected any
            line ending with ":" regardless of length.
            AFTER: threshold lowered to 10 chars; colon-terminated
            lines only rejected if they are ≤ 4 words (pure labels).
            Valid questions like "How does X work: an overview?" now
            pass through correctly.

  FIX D2 (NEW) — Prefix stripping breaks after first match.
            BEFORE: the prefix-strip loop could run multiple passes
            on the same string (e.g. "Q: Gap: actual question" would
            strip "Q:" then "Gap:" is still not checked in same pass).
            AFTER: break after the first matching prefix is stripped.

  FIX D3 (NEW) — _check_remaining_gaps JSON parse is case-insensitive
            for the ```json fence and handles trailing whitespace.

  FIX D4 (NEW) — _run_pipeline_for_node uses pipeline default min_score
            (0.42) instead of overriding to 0.20. With CHUNK_SIZE fixed
            to 160 words (no more truncation), 0.20 lets junk through.
            The 0.20 override was a workaround for the old truncation bug.

  FIX D5 (NEW) — Jaccard overlap for _dedup_against_parent.
            BEFORE: overlap = intersection / len(child_tokens) — could
            produce misleading ratios for very short child questions.
            AFTER: Jaccard = intersection / union — symmetric and bounded.

All Phase 11 bug fixes retained:
  BUG FIX 1 — Thread-safe RAG settings (kwargs, no singleton mutation)
  BUG FIX 3 — Concurrent child expansion with asyncio.gather
  BUG FIX 4 — Skip truly empty no_data nodes
  BUG FIX 5 — Per-node synthesis budget (no brutal truncation)
  BUG FIX 6 — all_images passed through correctly
  BUG FIX 7 — retrieve_all_questions() receives original_user_query
  BUG FIX 8 — Root questions generated directly, not via planner.plan()
  BUG FIX 9 — docs_by_question from retriever (per-Q docs, not broadcast)
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from openai import AsyncOpenAI

from config.settings import get_settings
from core.logger import setup_logger

settings = get_settings()
logger   = setup_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

ROOT_QUESTIONS        = 3
MAX_SUBTOPICS         = 1
QUESTIONS_PER_CHILD   = 2
DENSITY_THRESHOLD     = 0.30
NODE_SYNTHESIS_BUDGET = 1200   # chars per node in synthesis prompt
MAX_CHILD_QUESTIONS   = 5      # FIX 10A: hard cap on child questions
_DEDUP_OVERLAP_RATIO  = 0.60   # FIX D5: Jaccard threshold for dedup


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ResearchNode:
    topic        : str
    depth        : int
    questions    : List[str]            = field(default_factory=list)
    summaries    : List[Dict]           = field(default_factory=list)
    docs         : List[Dict]           = field(default_factory=list)
    images       : List[Dict]           = field(default_factory=list)
    children     : List["ResearchNode"] = field(default_factory=list)
    dense_topics : List[str]            = field(default_factory=list)
    status       : str                  = "pending"


@dataclass
class DeepResearchState:
    query           : str
    root_node       : Optional[ResearchNode] = None
    all_summaries   : List[Dict]             = field(default_factory=list)
    all_images      : List[Dict]             = field(default_factory=list)
    expanded_topics : Set[str]               = field(default_factory=set)
    status          : str                    = "running"
    errors          : List[str]              = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class DeepResearchAgent:
    """
    Phase 12 — Gap-Targeted Deep Research Agent.

    When called with coverage_gaps, the agent researches specifically
    what the standard pipeline missed rather than exploring randomly.

    Flow (with gaps):
        Root : 3 Qs targeting coverage_gaps[0:3]
             → retrieve → RAG → summarize
            ↓ check remaining gaps
        Child: 2 Qs targeting remaining coverage_gaps (if any)
             → retrieve → RAG → summarize
        → synthesize report focused on the gaps

    Flow (without gaps / fallback):
        Same as before — general topic exploration.

    Max questions: 5 (3 root + 2 child)
    """

    def __init__(self, planner, retriever, rag, summarizer):
        self.planner    = planner
        self.retriever  = retriever
        self.rag        = rag
        self.summarizer = summarizer

        self.client = AsyncOpenAI(
            api_key  = settings.openai_api_key,
            base_url = getattr(settings, "openai_base_url", None) or None,
        )
        self.model      = settings.openai_model
        self.fast_model = getattr(settings, "openai_fast_model", settings.openai_model)

        logger.info(
            "DeepResearchAgent (Phase 12) initialized | "
            f"root_questions={ROOT_QUESTIONS} | "
            f"max_subtopics={MAX_SUBTOPICS} | "
            f"questions_per_child={QUESTIONS_PER_CHILD} | "
            f"max_total=5"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Topic density scoring
    # ─────────────────────────────────────────────────────────────────────────

    def _score_topic_density(self, topic: str, docs: List[Dict]) -> float:
        if not docs:
            return 0.0
        topic_words = [w for w in topic.lower().split() if len(w) > 3]
        if not topic_words:
            return 0.0
        hit_count = 0
        for doc in docs:
            content = (
                doc.get("content", "") + " "
                + doc.get("title",   "") + " "
                + doc.get("text",    "")
            ).lower()
            matches = sum(1 for w in topic_words if w in content)
            if matches >= max(1, len(topic_words) // 2):
                hit_count += 1
        return round(hit_count / len(docs), 3)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _parse_question_lines(raw_text: str, n_questions: int) -> List[str]:
        """
        Parse LLM output into a clean list of questions.

        Rules:
          - Minimum length: 10 chars
          - Strip leading list markers and common prefixes
          - Reject ONLY pure header labels: all-caps AND ≤4 words AND no "?"
          - Reject colon-terminated lines ONLY if they are ≤4 words
          - Break after first matching prefix
        """
        _PREFIXES = ("gap:", "dimension:", "question:", "q:", "topic:")
        questions: List[str] = []

        for ln in raw_text.splitlines():
            q = ln.strip().lstrip("0123456789.-) \t").strip()

            if not q or len(q) < 10:
                continue

            q_lower = q.lower()
            for prefix in _PREFIXES:
                if q_lower.startswith(prefix):
                    q = q[len(prefix):].strip()
                    break

            if not q or len(q) < 10:
                continue

            words = q.split()
            if q.isupper() and len(words) <= 4 and "?" not in q:
                continue

            if q.endswith(":") and len(words) <= 4:
                continue

            questions.append(q)

            if len(questions) >= n_questions:
                break

        return questions
    

    # ─────────────────────────────────────────────────────────────────────────
    # FIX 1 / FIX 2 — Gap-targeted question generation
    # ─────────────────────────────────────────────────────────────────────────

    async def _generate_gap_questions(
        self,
        query         : str,
        coverage_gaps : List[str],
        n_questions   : int,
    ) -> List[str]:
        """
        Generate questions specifically targeting coverage gaps.

        Each question is designed to retrieve information that fills
        one of the identified missing dimensions.
        """
        if not coverage_gaps:
            return await self._generate_questions(query, "", n_questions)

        # Use only the first n_questions gaps (one question per gap)
        gaps_to_cover = coverage_gaps[:n_questions]
        gaps_str      = "\n".join(f"- {g}" for g in gaps_to_cover)

        prompt = (
            f'Research topic: "{query}"\n\n'
            f"The standard research pipeline already covered this topic broadly "
            f"but the following specific dimensions are MISSING or insufficiently covered:\n"
            f"{gaps_str}\n\n"
            f"Generate exactly {n_questions} focused research questions, "
            f"one per missing dimension, that would retrieve the specific "
            f"information needed to fill these gaps.\n\n"
            f"Requirements:\n"
            f"- Each question must directly target one of the missing dimensions\n"
            f"- Questions must be specific and searchable (keyword-rich)\n"
            f"- Do NOT generate questions about already-well-covered aspects\n\n"
            f"Return ONLY the questions, one per line, no numbering or bullets."
        )

        try:
            resp = await self.client.chat.completions.create(
                model       = self.fast_model,
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = 150,
                temperature = 0.3,
            )
            raw_text  = resp.choices[0].message.content.strip()
            questions = self._parse_question_lines(raw_text, n_questions)

            logger.info(
                f"DeepResearchAgent | gap questions generated | "
                f"gaps={gaps_to_cover} | questions={questions}"
            )

            if not questions:
                raise ValueError("No valid questions parsed from LLM response")

            return questions

        except Exception as e:
            logger.error(
                f"DeepResearchAgent | _generate_gap_questions failed: {e} "
                f"— falling back to generic questions"
            )
            return await self._generate_questions(query, "", n_questions)

    # ─────────────────────────────────────────────────────────────────────────
    # FIX 4 — Check which gaps remain after root research
    # ─────────────────────────────────────────────────────────────────────────

    async def _check_remaining_gaps(
        self,
        coverage_gaps  : List[str],
        root_summaries : List[Dict],
    ) -> List[str]:
        """
        After root research runs, check which gaps are still uncovered.
        Returns the list of still-uncovered dimensions so the child node
        can target them.

        FIX D3: JSON fence stripping is case-insensitive and handles
        trailing whitespace before the closing fence.
        """
        if not coverage_gaps:
            return []

        if not root_summaries:
            return coverage_gaps

        full_text = "\n\n".join(
            f"[Q: {s.get('question','')}]\n{s.get('summary','')}"
            for s in root_summaries
            if s.get("summary", "").strip()
        )

        dims_json = json.dumps(coverage_gaps)
        prompt = (
            f"Deep research just ran to fill these coverage gaps:\n{dims_json}\n\n"
            f"New research summaries:\n---\n{full_text[:3000]}\n---\n\n"
            f"Which gaps are STILL not substantively covered after this new research?\n"
            f"Return ONLY a valid JSON array of the still-uncovered gap strings.\n"
            f"If all gaps are now covered, return: []"
        )

        try:
            resp = await self.client.chat.completions.create(
                model       = self.fast_model,
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = 150,
                temperature = 0.1,
            )
            raw = resp.choices[0].message.content.strip()

            # FIX D3: case-insensitive fence stripping, handle trailing whitespace
            if raw.startswith("```"):
                # Remove opening fence (```json, ```JSON, ``` etc.)
                first_newline = raw.find("\n")
                if first_newline != -1:
                    raw = raw[first_newline:].strip()
                # Remove closing fence if present
                if raw.endswith("```"):
                    raw = raw[:-3].strip()

            remaining = json.loads(raw.strip())
            logger.info(
                f"DeepResearchAgent | remaining gaps after root: {remaining}"
            )
            return remaining if isinstance(remaining, list) else []

        except Exception as e:
            logger.warning(
                f"DeepResearchAgent | _check_remaining_gaps failed: {e} — "
                f"assuming all gaps still open"
            )
            return coverage_gaps

    # ─────────────────────────────────────────────────────────────────────────
    # Identify sub-topic for fallback (no gaps provided)
    # ─────────────────────────────────────────────────────────────────────────

    async def _identify_subtopic(
        self,
        parent_topic  : str,
        summaries     : List[Dict],
        docs          : List[Dict],
        excluded      : Set[str],
        coverage_gaps : List[str] = None,
    ) -> Optional[Tuple[str, float]]:
        """
        Returns (topic, density) for the single best sub-topic found,
        or None if nothing qualifies.

        FIX 3: If coverage_gaps is provided, skip sub-topics that
        merely overlap with gaps already being researched at root level,
        preventing double-research of the same dimension.
        """
        if not summaries and not docs:
            return None

        if summaries:
            context = "\n\n".join(
                f"Q: {s.get('question', '')}\nA: {s.get('summary', '')[:200]}"
                for s in summaries[:4]
                if s.get("summary", "").strip()
            )
        else:
            context = "\n".join(
                f"- {d.get('title', '') or d.get('content', '')[:80]}"
                for d in docs[:8]
            )

        if not context.strip():
            return None

        excluded_set = set(e.lower() for e in excluded)
        # FIX 3: also exclude dimensions already targeted by gap questions
        if coverage_gaps:
            for gap in coverage_gaps:
                excluded_set.add(gap.lower())

        excluded_str = ", ".join(list(excluded_set)[:10]) if excluded_set else "none"

        prompt = (
            f'Analyze research about: "{parent_topic}"\n\n'
            f"Identify the single most important sub-topic that appears "
            f"repeatedly and deserves deeper investigation.\n"
            f"Do NOT suggest any of these (already covered): {excluded_str}\n\n"
            f"Content:\n{context[:1000]}\n\n"
            f"Reply with ONLY the sub-topic name, 2-5 words, nothing else."
        )

        try:
            resp = await self.client.chat.completions.create(
                model       = self.fast_model,
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = 20,
                temperature = 0.3,
            )
            raw = resp.choices[0].message.content.strip().lstrip("-•*123456789. ").strip()
        except Exception as e:
            logger.error(f"DeepResearchAgent | sub-topic LLM failed: {e}")
            return None

        if not raw or raw.lower() in excluded_set or len(raw.split()) > 6:
            return None

        density = self._score_topic_density(raw, docs) if docs else 0.0

        if density < DENSITY_THRESHOLD:
            logger.info(
                f"DeepResearchAgent | sub-topic '{raw}' density={density:.3f} "
                f"below threshold — skipping"
            )
            return None

        logger.info(
            f"DeepResearchAgent | sub-topic selected: '{raw}' | density={density:.3f}"
        )
        return (raw, density)

    # ─────────────────────────────────────────────────────────────────────────
    # Generic question generation (fallback)
    # ─────────────────────────────────────────────────────────────────────────

    async def _generate_questions(
        self,
        topic          : str,
        parent_context : str,
        n_questions    : int,
    ) -> List[str]:
        prompt = (
            f'Generate {n_questions} specific research questions about: "{topic}"\n'
            f'Context: "{parent_context}"\n\n'
            f"Requirements:\n"
            f'- Each question must specifically address "{topic}"\n'
            f"- Cover a different aspect per question\n"
            f"- Answerable from web/academic sources\n\n"
            f"Return ONLY the questions, one per line, no numbering."
        )
        try:
            resp = await self.client.chat.completions.create(
                model       = self.fast_model,
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = 120,
                temperature = 0.4,
            )
            raw_text  = resp.choices[0].message.content.strip()
            # Reuse the shared parser for consistency
            questions = self._parse_question_lines(raw_text, n_questions)
            return questions or [
                f"What are the key aspects of {topic}?",
                f"How does {topic} work in practice?",
            ][:n_questions]
        except Exception as e:
            logger.error(f"DeepResearchAgent | question gen failed for '{topic}': {e}")
            return [
                f"What are the key aspects of {topic}?",
                f"How does {topic} work in practice?",
            ][:n_questions]

    # ─────────────────────────────────────────────────────────────────────────
    # Run pipeline for one node
    # ─────────────────────────────────────────────────────────────────────────

    async def _run_pipeline_for_node(self, node: ResearchNode) -> None:
        """
        Retrieve → RAG → Summarize for a single node.

        FIX D4: No longer overrides min_score to 0.20. The 0.20 override
        was a workaround for chunk truncation (CHUNK_SIZE was 400 words,
        exceeding the embedding model's 256-token limit). Now that
        CHUNK_SIZE is fixed to 160 words, the pipeline's default (0.42)
        correctly filters low-quality chunks without letting junk through.

        BUG FIX 1: min_score/top_k as kwargs only (no singleton mutation).
        BUG FIX 7: node.topic passed as original_user_query.
        BUG FIX 9: use docs_by_question from retriever so each question
                   gets its own ~10 docs instead of all questions sharing
                   the flat all_docs pool.
        """
        if not node.questions:
            node.status = "failed"
            return

        logger.info(
            f"DeepResearchAgent | pipeline | topic='{node.topic}' | "
            f"depth={node.depth} | questions={len(node.questions)}"
        )

        try:
            retrieved  = await self.retriever.retrieve_all_questions(
                node.questions,
                node.topic,
            )
            all_docs   = retrieved.get("all_docs",         [])
            all_images = retrieved.get("all_images",       [])
            docs_by_q  = retrieved.get("docs_by_question", {})

            node.images.extend(all_images)

            if not all_docs:
                logger.warning(f"DeepResearchAgent | no docs for '{node.topic}'")
                node.status = "no_data"
                return

            node.docs = all_docs

            if not docs_by_q:
                logger.warning(
                    f"DeepResearchAgent | docs_by_question empty — "
                    f"falling back to broadcast ({len(all_docs)} docs)"
                )
                docs_by_q = {q: all_docs for q in node.questions}

            # FIX D4: removed min_score=0.20 override — use pipeline default (0.42)
            # The old override was a workaround for CHUNK_SIZE=400 truncation bug.
            filtered_map = self.rag.filter_all_questions(
                docs_by_q,
                top_k=6,
            )

            total_chunks = sum(len(v) for v in filtered_map.values())
            logger.info(
                f"DeepResearchAgent | RAG | topic='{node.topic}' | "
                f"docs={len(all_docs)} → chunks={total_chunks}"
            )

            node.summaries = await self.summarizer.run_all(filtered_map)
            node.status    = "done"

            logger.info(
                f"DeepResearchAgent | node done | topic='{node.topic}' | "
                f"docs={len(node.docs)} | summaries={len(node.summaries)}"
            )

        except Exception as e:
            logger.error(f"DeepResearchAgent | pipeline failed '{node.topic}': {e}")
            node.status = "failed"
            raise

    # ─────────────────────────────────────────────────────────────────────────
    # FIX D5 — Jaccard-based parent-child deduplication
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_question(q: str) -> set:
        """Extract meaningful lowercase tokens from a question."""
        tokens = re.findall(r"[a-z0-9]+", q.lower())
        stop = {
            "what", "how", "why", "when", "where", "which", "does", "is",
            "are", "the", "a", "an", "of", "in", "to", "for", "and", "or",
            "do", "can", "its", "it", "this", "that", "with", "on", "by",
            "be", "has", "have", "been",
        }
        return {t for t in tokens if len(t) > 2 and t not in stop}

    def _dedup_against_parent(
        self,
        child_questions  : List[str],
        parent_questions : List[str],
    ) -> List[str]:
        """
        FIX D5: Drop child questions whose Jaccard similarity with any
        parent question exceeds _DEDUP_OVERLAP_RATIO.

        BEFORE: overlap = intersection / len(child_tokens)
                → misleading for short child questions (e.g. a 3-token
                  child matching 2 of 3 tokens scores 0.67 even if the
                  parent has 20 tokens — clearly not a duplicate).

        AFTER: Jaccard = intersection / union
               → symmetric, bounded [0,1], not skewed by question length.
        """
        if not parent_questions:
            return child_questions

        parent_token_sets = [
            self._normalize_question(q) for q in parent_questions
        ]

        kept: List[str] = []
        for cq in child_questions:
            child_tokens = self._normalize_question(cq)
            if not child_tokens:
                kept.append(cq)
                continue

            is_dup = False
            for pt_set in parent_token_sets:
                if not pt_set:
                    continue
                intersection = child_tokens & pt_set
                union        = child_tokens | pt_set
                if not union:
                    continue
                # FIX D5: Jaccard similarity instead of asymmetric ratio
                jaccard = len(intersection) / len(union)
                if jaccard >= _DEDUP_OVERLAP_RATIO:
                    logger.info(
                        f"FIX 10B | dedup dropped child Q (overlap={jaccard:.0%}): "
                        f"'{cq[:60]}'"
                    )
                    is_dup = True
                    break
            if not is_dup:
                kept.append(cq)

        if not kept and child_questions:
            logger.warning(
                "FIX 10B | all child questions were duplicates of parent — "
                "keeping the first one as fallback"
            )
            kept = [child_questions[0]]

        return kept

    # ─────────────────────────────────────────────────────────────────────────
    # FIX 4 — Gap-aware child expansion
    # ─────────────────────────────────────────────────────────────────────────

    async def _expand_root(
        self,
        root          : ResearchNode,
        state         : DeepResearchState,
        coverage_gaps : List[str] = None,
    ) -> None:
        """
        Finds the child topic/questions for the second research pass.

        With coverage_gaps:
            Checks which gaps are STILL uncovered after root research.
            If any remain, generates child questions targeting them.

        Without coverage_gaps (fallback):
            Uses sub-topic density discovery (original behavior).
        """
        if root.status not in ("done", "no_data"):
            return

        if root.status == "no_data" and not root.docs and not root.summaries:
            return

        child_questions : List[str] = []
        child_topic     : str       = ""

        # ── Gap-aware path ────────────────────────────────────────────────────
        if coverage_gaps:
            remaining_gaps = await self._check_remaining_gaps(
                coverage_gaps  = coverage_gaps,
                root_summaries = root.summaries,
            )

            if remaining_gaps:
                child_topic     = f"{root.topic} — remaining gaps"
                child_questions = await self._generate_gap_questions(
                    query         = root.topic,
                    coverage_gaps = remaining_gaps,
                    n_questions   = QUESTIONS_PER_CHILD,
                )
                logger.info(
                    f"DeepResearchAgent | child targeting remaining gaps: "
                    f"{remaining_gaps}"
                )
            else:
                logger.info(
                    "DeepResearchAgent | all gaps covered after root research — "
                    "no child needed"
                )
                return

        # ── Fallback: sub-topic density discovery ─────────────────────────────
        else:
            result = await self._identify_subtopic(
                parent_topic  = root.topic,
                summaries     = root.summaries,
                docs          = root.docs,
                excluded      = state.expanded_topics,
                coverage_gaps = coverage_gaps or [],
            )

            if result is None:
                logger.info(
                    "DeepResearchAgent | no qualifying sub-topic — single level only"
                )
                return

            sub_topic, density = result

            if sub_topic.lower() in {e.lower() for e in state.expanded_topics}:
                logger.info(
                    f"DeepResearchAgent | '{sub_topic}' already expanded — skipping"
                )
                return

            child_topic = sub_topic
            state.expanded_topics.add(sub_topic.lower())
            root.dense_topics = [sub_topic]

            child_questions = await self._generate_questions(
                topic          = sub_topic,
                parent_context = root.topic,
                n_questions    = QUESTIONS_PER_CHILD,
            )
            logger.info(
                f"DeepResearchAgent | expanding child '{sub_topic}' | "
                f"density={density:.3f}"
            )

        # ── FIX 10A: Hard cap on child questions ──────────────────────────────
        if len(child_questions) > MAX_CHILD_QUESTIONS:
            logger.info(
                f"FIX 10A | capping child questions from {len(child_questions)} → "
                f"{MAX_CHILD_QUESTIONS}"
            )
            child_questions = child_questions[:MAX_CHILD_QUESTIONS]

        # ── FIX D5: Dedup child questions against parent (Jaccard) ───────────
        child_questions = self._dedup_against_parent(
            child_questions  = child_questions,
            parent_questions = root.questions,
        )

        if not child_questions:
            return

        # ── Run child pipeline ────────────────────────────────────────────────
        child           = ResearchNode(topic=child_topic, depth=1)
        child.questions = child_questions

        try:
            await self._run_pipeline_for_node(child)
        except Exception as e:
            logger.error(f"DeepResearchAgent | child '{child_topic}' failed: {e}")
            child.status = "failed"

        root.children.append(child)
        state.all_images.extend(child.images)

        for s in child.summaries:
            s["depth"]        = 1
            s["parent_topic"] = root.topic
            s["sub_topic"]    = child.topic

        state.all_summaries.extend(child.summaries)

    # ─────────────────────────────────────────────────────────────────────────
    # Build final report
    # ─────────────────────────────────────────────────────────────────────────

    async def _build_report(
        self,
        state         : DeepResearchState,
        coverage_gaps : List[str] = None,
    ) -> str:
        """BUG FIX 5: per-node budget so both root and child appear."""
        root = state.root_node
        if not root:
            return "Deep research failed — no data collected."

        parts: List[str] = []

        def collect(node: ResearchNode, level: int):
            prefix = "#" * (level + 2)
            parts.append(f"\n{prefix} {node.topic}\n")
            node_chars = 0
            for s in node.summaries:
                text = s.get("summary", "").strip()
                if not text or len(text.split()) < 10:
                    continue
                entry = f"**Q:** {s.get('question', '')}\n{text}\n"
                if node_chars + len(entry) > NODE_SYNTHESIS_BUDGET:
                    remaining = NODE_SYNTHESIS_BUDGET - node_chars
                    if remaining > 80:
                        parts.append(entry[:remaining] + "…\n")
                    break
                parts.append(entry)
                node_chars += len(entry)
            for child in node.children:
                collect(child, level + 1)

        collect(root, 0)
        full_text = "\n".join(parts)

        # Build gap-aware synthesis instruction
        if coverage_gaps:
            gaps_str = "\n".join(f"  - {g}" for g in coverage_gaps)
            gap_instruction = (
                f"3. Focuses specifically on filling these coverage gaps that were "
                f"missing from the initial research:\n{gaps_str}\n"
            )
        elif root.children:
            gap_instruction = (
                f"3. Dedicates a sub-section to the sub-topic "
                f"'{root.children[0].topic}'\n"
            )
        else:
            gap_instruction = ""

        synthesis_prompt = (
            f'You are an expert research analyst. Write a supplementary research '
            f'report on: "{state.query}"\n\n'
            f"This is a DEEP RESEARCH pass that fills gaps from the standard pipeline.\n\n"
            f"Research data:\n{full_text}\n\n"
            f"Write a report that:\n"
            f"1. Starts with a 2-sentence summary of what new information was found\n"
            f"2. Covers the core findings from this deep research pass\n"
            f"{gap_instruction}"
            f"4. Ends with Key Conclusions\n\n"
            f"Format: markdown, ## for sections, ### for sub-sections.\n"
            f"Academic prose — no bullet lists. Write now:"
        )

        for attempt in range(3):
            try:
                resp = await self.client.chat.completions.create(
                    model       = self.model,
                    messages    = [{"role": "user", "content": synthesis_prompt}],
                    max_tokens  = 1200,
                    temperature = 0.3,
                )
                synthesized = resp.choices[0].message.content.strip()
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "queue_exceeded" in err_str or "too_many_requests" in err_str:
                    wait = 5 * (2 ** attempt)
                    logger.warning(
                        f"DeepResearchAgent | 429 on synthesis attempt {attempt + 1} "
                        f"— retrying in {wait}s"
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"DeepResearchAgent | synthesis failed: {e}")
                    synthesized = self._fallback_report(state)
                    break
        else:
            logger.error("DeepResearchAgent | synthesis failed after all retries")
            synthesized = self._fallback_report(state)

        tree_map = self._build_tree_map(root)
        return f"# Deep Research Report: {state.query}\n\n{tree_map}\n\n{synthesized}"

    def _build_tree_map(self, root: ResearchNode) -> str:
        lines = ["## Research Tree\n```"]
        lines.append(
            f"[L0] {root.topic} "
            f"({len(root.summaries)} summaries, {len(root.docs)} docs)"
        )
        for i, child in enumerate(root.children):
            conn = "└── " if i == len(root.children) - 1 else "├── "
            lines.append(
                f"    {conn}[L1] {child.topic} "
                f"({len(child.summaries)} summaries, {len(child.docs)} docs)"
            )
        lines.append("```\n")
        return "\n".join(lines)

    def _fallback_report(self, state: DeepResearchState) -> str:
        parts: List[str] = []

        def render(node: ResearchNode):
            h     = "#" * (node.depth + 2)
            label = f"Deep Dive: {node.topic}" if node.depth > 0 else "Core Research"
            parts.append(f"\n{h} {label}\n")
            for s in node.summaries:
                txt = s.get("summary", "").strip()
                if txt and len(txt.split()) > 10:
                    parts.append(f"**{s.get('question', '')}**\n{txt}\n")
            for child in node.children:
                render(child)

        if state.root_node:
            render(state.root_node)
        return "\n".join(parts)

    # ─────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────────

    async def run(self, query: str, coverage_gaps: List[str] = None, **_kwargs) -> Dict:
        """
        FIX 1: Accepts coverage_gaps from depth_judge_node.

        When coverage_gaps is provided:
          - Root questions target the first ROOT_QUESTIONS gaps specifically
          - Child questions target any remaining gaps after root research
          - The synthesis prompt emphasizes filling the identified gaps

        When coverage_gaps is None or empty:
          - Falls back to general topic exploration (original behavior)
          - Sub-topic discovery used for child node

        BUG FIX 8: root questions generated via _generate_gap_questions()
        or _generate_questions() directly, never via self.planner.plan().
        """
        coverage_gaps = coverage_gaps or []

        logger.info(
            f"DeepResearchAgent | START | query='{query}' | "
            f"coverage_gaps={coverage_gaps}"
        )

        state = DeepResearchState(query=query)

        try:
            root_node = ResearchNode(topic=query, depth=0)
            state.root_node = root_node
            state.expanded_topics.add(query.lower())

            # ── Step 1: generate root questions ──────────────────────────────
            if coverage_gaps:
                logger.info(
                    f"DeepResearchAgent | generating {ROOT_QUESTIONS} gap-targeted "
                    f"root questions for gaps: {coverage_gaps[:ROOT_QUESTIONS]}"
                )
                root_questions = await self._generate_gap_questions(
                    query         = query,
                    coverage_gaps = coverage_gaps,
                    n_questions   = ROOT_QUESTIONS,
                )
            else:
                logger.info(
                    f"DeepResearchAgent | no coverage gaps — generating "
                    f"{ROOT_QUESTIONS} general root questions"
                )
                root_questions = await self._generate_questions(
                    topic          = query,
                    parent_context = "",
                    n_questions    = ROOT_QUESTIONS,
                )

            root_node.questions = root_questions
            logger.info(
                f"DeepResearchAgent | Root | questions={len(root_questions)} | "
                f"{root_questions}"
            )

            # ── Step 2: run root pipeline ─────────────────────────────────────
            await self._run_pipeline_for_node(root_node)

            state.all_images.extend(root_node.images)

            for s in root_node.summaries:
                s["depth"]        = 0
                s["parent_topic"] = None
                s["sub_topic"]    = query
            state.all_summaries.extend(root_node.summaries)

            # ── Step 3: expand to child (gap-aware) ───────────────────────────
            await self._expand_root(
                root          = root_node,
                state         = state,
                coverage_gaps = coverage_gaps,
            )

            # ── Step 4: synthesize ────────────────────────────────────────────
            final_report = await self._build_report(state, coverage_gaps=coverage_gaps)
            state.status = "done"

            # ── Stats ─────────────────────────────────────────────────────────
            total_questions = len(root_node.questions) + sum(
                len(c.questions) for c in root_node.children
            )
            total_docs = len(root_node.docs) + sum(
                len(c.docs) for c in root_node.children
            )
            depth_reached      = 1 if root_node.children else 0
            dense_topics_found = list(state.expanded_topics - {query.lower()})

            seen_urls: Set[str] = set()
            deduped_images: List[Dict] = []
            for img in state.all_images:
                url = img.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    deduped_images.append(img)

            logger.info(
                f"DeepResearchAgent | COMPLETE | "
                f"questions={total_questions} | docs={total_docs} | "
                f"summaries={len(state.all_summaries)} | "
                f"depth_reached={depth_reached} | "
                f"images={len(deduped_images)} | "
                f"gaps_targeted={coverage_gaps}"
            )

            return {
                "query"              : query,
                "final_report"       : final_report,
                "all_summaries"      : state.all_summaries,
                "all_images"         : deduped_images,
                "research_tree"      : _serialize_tree(root_node),
                "total_questions"    : total_questions,
                "total_docs"         : total_docs,
                "depth_reached"      : depth_reached,
                "dense_topics_found" : dense_topics_found,
                "coverage_gaps"      : coverage_gaps,
                "status"             : "done",
            }

        except Exception as e:
            logger.error(f"DeepResearchAgent | FAILED: {e}", exc_info=True)
            return {
                "query"              : query,
                "final_report"       : f"Deep research failed: {e}",
                "all_summaries"      : state.all_summaries,
                "all_images"         : state.all_images,
                "research_tree"      : {},
                "total_questions"    : 0,
                "total_docs"         : 0,
                "depth_reached"      : 0,
                "dense_topics_found" : [],
                "coverage_gaps"      : coverage_gaps,
                "status"             : "failed",
                "errors"             : [str(e)],
            }


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _serialize_tree(node: ResearchNode) -> Dict:
    return {
        "topic"        : node.topic,
        "depth"        : node.depth,
        "status"       : node.status,
        "questions"    : node.questions,
        "summary_count": len(node.summaries),
        "doc_count"    : len(node.docs),
        "dense_topics" : node.dense_topics,
        "children"     : [_serialize_tree(c) for c in node.children],
    }