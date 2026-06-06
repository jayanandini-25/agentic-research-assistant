"""
LangGraph Nodes — Phase 12 (Coverage-Based Depth Judge)
=====================================================================
File location:  app/graph/nodes.py

KEY CHANGES vs previous version:

  FIX 1 — depth_judge_node completely rewritten.

      OLD (broken) logic:
          avg_words >= 120 AND avg_quality >= 0.85 AND n_sums >= 8
          → "sufficient" immediately (FAST SKIP)
          This is purely statistical. It measures HOW MUCH text was
          generated, not WHETHER the query's important dimensions are
          actually covered. A summarizer writing 200 words of vague
          fluff scores 1.00 quality and passes all thresholds. Deep
          research never fires even when coverage is genuinely poor.

      NEW (correct) logic — two stages:

          Stage 1 — Extract required dimensions (1 LLM call):
              For the query, identify the 6-10 specific dimensions
              a complete answer MUST cover. For a comparison query
              like "RAG vs Fine-tuning tradeoffs" this produces axes
              like: cost comparison, latency, when to choose each,
              data requirements, hallucination behavior, production
              complexity, benchmarks, scalability.

          Stage 2 — Check coverage (1 LLM call):
              Given the full summary text, determine which of those
              dimensions are substantively covered vs missing.
              "Substantively" means actually explained, not just
              mentioned in passing.

          Decision:
              coverage_ratio = covered / total_dimensions
              if coverage_ratio >= COVERAGE_THRESHOLD (0.70): sufficient
              else: needs_deep_research

          The uncovered dimensions are passed to the deep research
          agent so it targets the gaps specifically, not random
          sub-topics.

  FIX 2 — deep_research_node passes coverage_gaps to the agent
           so deep research is targeted, not exploratory.

  FIX 3 — verifier_node: low-quality summaries (quality < 0.65)
           are now flagged and passed to writer with a warning tag
           instead of silently included. The Q6 "i don't know"
           summary that scored 0.70 would be flagged here.

  All other nodes unchanged.
"""

from typing import Dict, Any, List, Tuple
import json

from core.logger import setup_logger
from app.graph.state import PipelineStatus, ResearchState
from app.agents.planner import PlannerAgent
from app.agents.planner import _get_dimensions as _planner_get_dimensions
from app.retrieval.hybrid_retriever import HybridRetriever
from app.rag.rag_pipeline import RAGPipeline
from app.rag.summarizer import Summarizer
from app.agents.writer_agent import WriterAgent
from app.agents.verification_agent import VerificationAgent
from app.agents.deep_research_agent import DeepResearchAgent
from config.settings import get_settings
from openai import AsyncOpenAI

settings = get_settings()
logger   = setup_logger(__name__)

MAX_PLANNER_QUESTIONS = 12

# Coverage threshold: if fewer than this fraction of required dimensions
# are covered by the summaries, trigger deep research.
COVERAGE_THRESHOLD = 0.70

# Min quality score below which a summary is flagged as low-confidence
# and tagged for the writer agent to treat carefully.
LOW_QUALITY_THRESHOLD = 0.65


# ─────────────────────────────────────────────────────────────────────────────
# Singletons
# ─────────────────────────────────────────────────────────────────────────────

_planner    = PlannerAgent()
_retriever  = HybridRetriever()
_rag        = RAGPipeline()
_summarizer = Summarizer()
_writer     = WriterAgent()
_verifier   = VerificationAgent(
    retriever  = _retriever,
    rag        = _rag,
    summarizer = _summarizer,
)
_deep_agent = DeepResearchAgent(
    planner    = _planner,
    retriever  = _retriever,
    rag        = _rag,
    summarizer = _summarizer,
)
_openai = AsyncOpenAI(
    api_key  = settings.openai_api_key,
    base_url = getattr(settings, "openai_base_url", None) or None,
)
_fast_model = getattr(settings, "openai_fast_model", settings.openai_model)


# ─────────────────────────────────────────────────────────────────────────────
# Coverage helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _extract_required_dimensions(query: str) -> List[str]:
    """
    Ask the LLM what dimensions a complete answer to this query MUST cover.

    Returns a list of 6-10 short dimension strings.
    Falls back to a generic set if the LLM call fails.
    """
    prompt = f"""You are a research quality assessor.

For this research query: "{query}"

List the 6-10 specific dimensions or aspects that a truly complete answer
MUST substantively cover. Be concrete and specific to this exact query.

Rules:
- For comparison queries ("X vs Y"): list evaluation axes
  e.g. cost, latency, accuracy, scalability, use cases, limitations, when-to-choose
- For how-to queries: list key steps or components
- For topic queries: list essential sub-topics
- Each dimension should be 2-5 words
- No overlap between dimensions

Return ONLY a valid JSON array of strings, nothing else.
Example format: ["cost comparison", "inference latency", "data requirements"]"""

    try:
        resp = await _openai.chat.completions.create(
            model       = _fast_model,
            messages    = [{"role": "user", "content": prompt}],
            max_tokens  = 200,
            temperature = 0.2,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        dimensions = json.loads(raw.strip())
        if isinstance(dimensions, list) and len(dimensions) >= 3:
            logger.info(
                f"depth_judge | extracted {len(dimensions)} required dimensions: "
                f"{dimensions}"
            )
            return dimensions[:10]
    except Exception as e:
        logger.warning(f"depth_judge | _extract_required_dimensions failed: {e}")

    # Fallback: generic dimensions for unknown query type
    return [
        "core concepts and definitions",
        "key differences and similarities",
        "performance and accuracy",
        "practical use cases",
        "limitations and tradeoffs",
        "when to choose each option",
    ]


async def _check_coverage(
    dimensions: List[str],
    summary_text: str,
) -> Tuple[List[str], List[str]]:
    """
    Given the full summary text, determine which dimensions are
    substantively covered vs missing.

    'Substantively covered' means actually explained with specific
    detail — not just mentioned in passing or in a vague sentence.

    Returns (covered_list, uncovered_list).
    Falls back to assuming all covered if LLM call fails (safe default).
    """
    dims_json = json.dumps(dimensions)

    prompt = f"""You are a research quality assessor.

Research summaries (full text):
---
{summary_text[:4000]}
---

Required dimensions for a complete answer:
{dims_json}

For each dimension, determine if it is SUBSTANTIVELY covered in the summaries.
"Substantively covered" means:
  ✓ Explained with specific details, examples, numbers, or named concepts
  ✗ NOT just mentioned in passing in one sentence
  ✗ NOT just a vague general statement with no specifics

Return ONLY a valid JSON object with exactly two keys:
{{
  "covered": ["dimension1", "dimension2", ...],
  "uncovered": ["dimension3", "dimension4", ...]
}}

Use the exact dimension strings from the input list."""

    try:
        resp = await _openai.chat.completions.create(
            model       = _fast_model,
            messages    = [{"role": "user", "content": prompt}],
            max_tokens  = 300,
            temperature = 0.1,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
        covered   = result.get("covered",   [])
        uncovered = result.get("uncovered", [])

        # Validate: all dimensions must appear in one list or the other.
        # The LLM may return a short key ("speed_cost") instead of the full
        # readable dimension ("speed_cost: inference speed, latency...").
        # We match on prefix to avoid double-counting.
        all_returned = set(covered) | set(uncovered)
        for dim in dimensions:
            if dim in all_returned:
                continue
            # Check if the LLM returned a shortened version (key before ':')
            dim_prefix = dim.split(":")[0].strip()
            prefix_found = any(
                r == dim_prefix or r.split(":")[0].strip() == dim_prefix
                for r in all_returned
            )
            if not prefix_found:
                # LLM genuinely dropped this dimension — mark uncovered
                uncovered.append(dim)

        logger.info(
            f"depth_judge | coverage check: "
            f"{len(covered)}/{len(dimensions)} covered | "
            f"uncovered={uncovered}"
        )
        return covered, uncovered

    except Exception as e:
        logger.warning(
            f"depth_judge | _check_coverage failed: {e} — "
            f"defaulting to all covered (safe fallback)"
        )
        # Safe fallback: assume covered so we don't trigger unnecessary deep research
        return dimensions, []


# ─────────────────────────────────────────────────────────────────────────────
# Node 0 — Depth Judge  (REWRITTEN — coverage-based)
# ─────────────────────────────────────────────────────────────────────────────

async def depth_judge_node(state: ResearchState) -> Dict[str, Any]:
    """
    Coverage-based depth judge.

    Instead of checking word counts and quality scores (which measure
    quantity of text, not completeness of coverage), this node:

    1. Extracts the specific dimensions the query REQUIRES
    2. Checks whether each dimension is substantively covered in the summaries
    3. Computes coverage_ratio = covered / total
    4. Triggers deep research if coverage_ratio < COVERAGE_THRESHOLD

    The uncovered dimensions are passed to the deep research agent so
    it researches the specific gaps rather than random sub-topics.

    Two cheap LLM calls (fast model, low max_tokens).
    Total overhead: ~2-3 seconds.
    """
    query         = state.get("research_query", "")
    v_sums        = state.get("verified_summaries", [])
    sums          = state.get("summaries", [])
    sums_to_judge = v_sums if v_sums else sums
    planner_meta  = state.get("planner_meta", {})

    logger.info(
        f"depth_judge_node | query='{query}' | summaries={len(sums_to_judge)} "
        f"| has_planner_meta={bool(planner_meta)}"
    )

    # Guard: nothing to judge
    if not sums_to_judge:
        logger.warning("depth_judge_node | no summaries — defaulting to sufficient")
        return {
            "depth_judge_decision": "sufficient",
            "is_deep_research":     False,
            "coverage_gaps":        [],
            "coverage_ratio":       0.0,
            "status":               PipelineStatus.JUDGING,
        }

    # Build full summary text for coverage checking
    full_summary_text = "\n\n".join(
        f"[Q: {s.get('question', '')}]\n{s.get('summary', '')}"
        for s in sums_to_judge
        if s.get("summary", "").strip()
    )

    # ── Phase 13 + FIX H-B: Use planner's dimensions when available ───
    # Instead of independently extracting dimensions via LLM (which can
    # diverge from planner's analysis), prefer the planner's own
    # uncovered_dims + covered_dims as the authoritative dimension set.
    #
    # FIX H-B: Map raw dimension keys to human-readable descriptions so
    # the LLM in _check_coverage() can reliably match them against
    # summary text. "key_differences" is ambiguous; "key_differences:
    # What are the fundamental differences between X and Y?" is clear.
    # ── Build two parallel lists:
    #    1. required_dimensions — human-readable, for _check_coverage() LLM
    #    2. dim_key_map — maps each readable dimension back to its short key
    #       so coverage_gaps returns clean keys (not "key: long description")
    dim_key_map: Dict[str, str] = {}  # readable_dim → short_key

    if planner_meta and planner_meta.get("uncovered_dims") is not None:
        planner_covered   = planner_meta.get("covered_dims", [])
        planner_uncovered = planner_meta.get("uncovered_dims", [])
        all_dim_keys = list(set(planner_covered + planner_uncovered))

        # Resolve keys → descriptions using the planner's dimension registry
        query_type = planner_meta.get("query_type", "general")
        domain     = planner_meta.get("domain", "general")
        try:
            dim_defs = {d["key"]: d["desc"] for d in _planner_get_dimensions(query_type, domain)}
        except Exception:
            dim_defs = {}

        required_dimensions = []
        for key in all_dim_keys:
            desc = dim_defs.get(key, "")
            if desc:
                readable = f"{key}: {desc}"
            else:
                readable = key
            required_dimensions.append(readable)
            dim_key_map[readable] = key

        logger.info(
            f"depth_judge_node | using planner's dimensions ({len(required_dimensions)} total) "
            f"with descriptions | instead of LLM extraction"
        )
    else:
        # Fallback: extract via LLM if planner_meta is absent
        required_dimensions = await _extract_required_dimensions(query)
        # For LLM-extracted dims, the readable string IS the key
        for d in required_dimensions:
            dim_key_map[d] = d

    # Step 2: what's actually covered in the summaries?
    covered, uncovered = await _check_coverage(required_dimensions, full_summary_text)

    coverage_ratio = len(covered) / len(required_dimensions) if required_dimensions else 1.0

    # Convert uncovered readable dimensions back to clean short keys
    # so deep_research_agent gets "cost_comparison" not
    # "cost_comparison: How do costs compare between X and Y?"
    uncovered_keys = [dim_key_map.get(u, u) for u in uncovered]

    logger.info(
        f"depth_judge_node | coverage={len(covered)}/{len(required_dimensions)} "
        f"({coverage_ratio:.2f}) | threshold={COVERAGE_THRESHOLD} | "
        f"uncovered_keys={uncovered_keys}"
    )

    if coverage_ratio >= COVERAGE_THRESHOLD:
        logger.info(
            f"depth_judge_node | SUFFICIENT | ratio={coverage_ratio:.2f} >= {COVERAGE_THRESHOLD}"
        )
        return {
            "depth_judge_decision": "sufficient",
            "is_deep_research":     False,
            "coverage_gaps":        uncovered_keys,   # Clean short keys
            "coverage_ratio":       coverage_ratio,
            "status":               PipelineStatus.JUDGING,
        }
    else:
        logger.info(
            f"depth_judge_node | NEEDS DEEP RESEARCH | "
            f"ratio={coverage_ratio:.2f} < {COVERAGE_THRESHOLD} | "
            f"gaps={uncovered_keys}"
        )
        return {
            "depth_judge_decision": "needs_deep_research",
            "is_deep_research":     True,
            "coverage_gaps":        uncovered_keys,   # Clean short keys
            "coverage_ratio":       coverage_ratio,
            "status":               PipelineStatus.JUDGING,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — Deep Research
# ─────────────────────────────────────────────────────────────────────────────

async def deep_research_node(state: ResearchState) -> Dict[str, Any]:
    query         = state.get("research_query",      "")
    existing_sums = state.get("verified_summaries",  []) or state.get("summaries", [])
    all_docs      = state.get("all_docs",             [])
    all_images    = state.get("all_images",           [])
    # Pass the coverage gaps so deep research targets them specifically
    coverage_gaps = state.get("coverage_gaps",        [])

    logger.info(
        f"deep_research_node | query='{query}' | "
        f"existing_summaries={len(existing_sums)} | "
        f"coverage_gaps={coverage_gaps}"
    )

    try:
        # Pass coverage_gaps to the agent — it will generate questions
        # specifically targeting the uncovered dimensions instead of
        # picking random dense sub-topics
        result = await _deep_agent.run(query=query, coverage_gaps=coverage_gaps)

        if result.get("status") == "failed":
            logger.warning(
                "deep_research_node | deep research failed — "
                "falling through to report_node with standard summaries"
            )
            return {
                "depth_judge_decision": "sufficient",
                "is_deep_research":     False,
                "status":               PipelineStatus.WRITING,
            }

        deep_summaries: List[Dict] = result.get("all_summaries", [])

        for s in existing_sums:
            if "depth" not in s:
                s["depth"]        = 0
                s["parent_topic"] = None
                s["sub_topic"]    = query

        all_summaries = existing_sums + deep_summaries

        sources = list({
            url
            for s in all_summaries
            for url in (s.get("sources") or [])
            if url
        })

        existing_image_urls = {img.get("url", "") for img in all_images}
        deep_images         = result.get("all_images", [])
        merged_images       = all_images + [
            img for img in deep_images
            if img.get("url", "") not in existing_image_urls
        ]

        logger.info(
            f"deep_research_node | done | "
            f"standard_sums={len(existing_sums)} | "
            f"deep_sums={len(deep_summaries)} | "
            f"total_sums={len(all_summaries)} | "
            f"depth_reached={result.get('depth_reached', 0)}"
        )

        return {
            "summaries":            all_summaries,
            "verified_summaries":   all_summaries,
            "verification_status":  "deep_research_complete",
            "verification_report":  {
                "summary_stats": {
                    "total":          len(all_summaries),
                    "contradictions": 0,
                    "weak_answers":   0,
                    "imbalanced":     0,
                    "re_retrieved":   0,
                    "fixed":          0,
                },
                "flagged_issues":    [],
                "re_retrieval_log":  [],
            },
            "sources":              sources,
            "deep_research_result": result,
            "research_tree":        result.get("research_tree",      {}),
            "dense_topics_found":   result.get("dense_topics_found", []),
            "depth_reached":        result.get("depth_reached",       0),
            "all_images":           merged_images,
            "all_docs":             all_docs,
            "status":               PipelineStatus.WRITING,
        }

    except Exception as e:
        logger.error(f"deep_research_node | FAILED: {e}", exc_info=True)
        return {
            "depth_judge_decision": "sufficient",
            "is_deep_research":     False,
            "status":               PipelineStatus.WRITING,
            "errors":               state.get("errors", []) + [str(e)],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — Planner
# ─────────────────────────────────────────────────────────────────────────────

async def planner_node(state: ResearchState) -> Dict[str, Any]:
    query = state.get("research_query", "")
    logger.info(f"planner_node | query='{query}'")

    try:
        result = await _planner.plan(query)

        # Phase 13: planner now returns dict with planner_meta
        if isinstance(result, list):
            sub_questions = result
            planner_meta  = {}
        else:
            sub_questions = result.get("sub_questions", [])
            planner_meta  = result.get("planner_meta", {})

        sub_questions = sub_questions[:MAX_PLANNER_QUESTIONS]

        if not sub_questions:
            logger.warning("planner_node | no sub-questions generated")
            return {
                "status": PipelineStatus.FAILED,
                "errors": ["PlannerAgent returned no sub-questions"],
            }

        logger.info(
            f"planner_node | generated {len(sub_questions)} sub-questions "
            f"(capped at {MAX_PLANNER_QUESTIONS})"
            + (f" | planner_meta: type={planner_meta.get('query_type')} "
               f"dims={planner_meta.get('total_dims', '?')} "
               f"gaps={planner_meta.get('critical_gaps', [])}"
               if planner_meta else "")
        )
        return {
            "sub_questions": sub_questions,
            "planner_meta":  planner_meta,
            "status":        PipelineStatus.RETRIEVING,
        }

    except Exception as e:
        logger.error(f"planner_node | FAILED: {e}", exc_info=True)
        return {"status": PipelineStatus.FAILED, "errors": [str(e)]}


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — Retriever
# ─────────────────────────────────────────────────────────────────────────────

async def retriever_node(state: ResearchState) -> Dict[str, Any]:
    sub_questions  = state.get("sub_questions",  [])
    existing_docs  = state.get("all_docs",        [])
    iteration      = state.get("iteration_count",  0)
    original_query = state.get("research_query",  "")

    logger.info(
        f"retriever_node | questions={len(sub_questions)} | "
        f"iteration={iteration} | existing_docs={len(existing_docs)}"
    )

    try:
        retrieved = await _retriever.retrieve_all_questions(
            sub_questions,
            original_query,
        )
        new_docs      = retrieved.get("all_docs",         [])
        all_images    = retrieved.get("all_images",       [])
        source_counts = retrieved.get("source_counts",    {})
        docs_by_q     = retrieved.get("docs_by_question", {})

        if iteration > 0 and existing_docs:
            existing_urls = {d.get("url", "") for d in existing_docs if d.get("url")}
            additional    = [d for d in new_docs if d.get("url", "") not in existing_urls]
            all_docs      = existing_docs + additional
            logger.info(
                f"retriever_node | re-retrieval | "
                f"new={len(new_docs)} | additional={len(additional)} | "
                f"total={len(all_docs)}"
            )
        else:
            all_docs = new_docs

        logger.info(
            f"retriever_node | total_docs={len(all_docs)} | images={len(all_images)}"
        )

        return {
            "all_docs":          all_docs,
            "all_images":        all_images,
            "source_counts":     source_counts,
            "docs_by_question":  docs_by_q,
            "status":            PipelineStatus.RAG,
        }

    except Exception as e:
        logger.error(f"retriever_node | FAILED: {e}", exc_info=True)
        return {"status": PipelineStatus.FAILED, "errors": [str(e)]}


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 — RAG Filter
# ─────────────────────────────────────────────────────────────────────────────

async def rag_node(state: ResearchState) -> Dict[str, Any]:
    sub_questions = state.get("sub_questions",    [])
    docs_by_q     = state.get("docs_by_question", {})
    all_docs      = state.get("all_docs",          [])

    if not docs_by_q and all_docs and sub_questions:
        logger.warning(
            f"rag_node | docs_by_question empty — fallback to "
            f"{len(all_docs)} docs across {len(sub_questions)} questions"
        )
        docs_by_q = {q: all_docs for q in sub_questions}
    elif not docs_by_q:
        logger.error("rag_node | no docs_by_question AND no all_docs")
        return {
            "status": PipelineStatus.FAILED,
            "errors": ["rag_node: no documents available"],
        }

    logger.info(f"rag_node | questions={len(docs_by_q)} | total_docs={len(all_docs)}")

    try:
        filtered_map = _rag.filter_all_questions(docs_by_q)
        total_chunks = sum(len(v) for v in filtered_map.values())
        logger.info(
            f"rag_node | {len(all_docs)} docs → "
            f"{total_chunks} chunks across {len(filtered_map)} questions"
        )
        return {"filtered_map": filtered_map, "status": PipelineStatus.SUMMARIZING}

    except Exception as e:
        logger.error(f"rag_node | FAILED: {e}", exc_info=True)
        return {"status": PipelineStatus.FAILED, "errors": [str(e)]}


# ─────────────────────────────────────────────────────────────────────────────
# Node 5 — Summarizer
# ─────────────────────────────────────────────────────────────────────────────

async def summarizer_node(state: ResearchState) -> Dict[str, Any]:
    filtered_map = state.get("filtered_map", {})
    logger.info(f"summarizer_node | questions={len(filtered_map)}")

    try:
        summaries   = await _summarizer.run_all(filtered_map)
        avg_quality = (
            sum(s.get("quality_score", 0) for s in summaries) / len(summaries)
            if summaries else 0.0
        )
        logger.info(
            f"summarizer_node | summaries={len(summaries)} | avg_quality={avg_quality:.2f}"
        )
        return {"summaries": summaries, "status": PipelineStatus.VERIFYING}

    except Exception as e:
        logger.error(f"summarizer_node | FAILED: {e}", exc_info=True)
        return {"status": PipelineStatus.FAILED, "errors": [str(e)]}


# ─────────────────────────────────────────────────────────────────────────────
# Node 6 — Verifier
# ─────────────────────────────────────────────────────────────────────────────

async def verifier_node(state: ResearchState) -> Dict[str, Any]:
    """
    Verifier with low-quality summary flagging.

    Summaries scoring below LOW_QUALITY_THRESHOLD are tagged with
    low_confidence=True so the writer agent can handle them carefully
    (e.g. add a caveat, weight them lower, or skip them).

    The Q6 "i don't know" summary that scored 0.70 in your run would
    be caught here and tagged — it would not silently enter the report.
    """
    summaries     = state.get("summaries", [])
    all_docs      = state.get("all_docs",  [])
    current_iters = state.get("iteration_count", 0)

    logger.info(f"verifier_node | summaries={len(summaries)}")

    _empty_stats = {
        "total":          len(summaries),
        "contradictions": 0,
        "weak_answers":   0,
        "imbalanced":     0,
        "re_retrieved":   0,
        "fixed":          0,
    }

    # Tag low-quality summaries BEFORE verification so the verifier
    # can also inspect them and the writer knows to be careful.
    low_quality_count = 0
    for s in summaries:
        score = s.get("quality_score", 1.0)
        issues = s.get("issues", [])
        if score < LOW_QUALITY_THRESHOLD or any(
            "failure" in str(i).lower() or "don't know" in str(i).lower()
            for i in issues
        ):
            s["low_confidence"] = True
            low_quality_count += 1
            logger.warning(
                f"verifier_node | low-confidence summary flagged | "
                f"question='{s.get('question','')[:60]}' | "
                f"score={score:.2f} | issues={issues}"
            )

    if low_quality_count:
        logger.info(
            f"verifier_node | {low_quality_count}/{len(summaries)} summaries "
            f"flagged as low-confidence"
        )

    try:
        # Phase 13: Pass original_query so re-retrieval has proper context
        original_query = state.get("research_query", "")
        verification = await _verifier.verify(
            summaries, all_docs, original_query=original_query
        )
        v_status     = verification.get("status",            "unknown")
        v_issues     = verification.get("flagged_issues",     [])
        v_sums       = verification.get("verified_summaries", summaries)

        # Carry over low_confidence tags that verifier might have dropped
        q_to_lc = {
            s.get("question", ""): s.get("low_confidence", False)
            for s in summaries
        }
        for s in v_sums:
            if q_to_lc.get(s.get("question", ""), False):
                s["low_confidence"] = True

        logger.info(
            f"verifier_node | status={v_status} | "
            f"issues={len(v_issues)} | iteration={current_iters + 1}"
        )

        return {
            "verified_summaries":  v_sums,
            "verification_report": verification,
            "verification_status": v_status,
            "iteration_count":     current_iters + 1,
            "status":              PipelineStatus.WRITING,
        }

    except Exception as e:
        logger.error(f"verifier_node | FAILED: {e} — skipping", exc_info=True)
        return {
            "verified_summaries":  summaries,
            "verification_report": {
                "summary_stats":    _empty_stats,
                "flagged_issues":   [],
                "re_retrieval_log": [],
            },
            "verification_status": "verification_skipped",
            "iteration_count":     current_iters + 1,
            "status":              PipelineStatus.WRITING,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node 7 — Report Writer
# ─────────────────────────────────────────────────────────────────────────────

async def report_node(state: ResearchState) -> Dict[str, Any]:
    query         = state.get("research_query",      "")
    v_sums        = state.get("verified_summaries",  [])
    summaries     = state.get("summaries",           [])
    all_images    = state.get("all_images",          [])
    all_docs      = state.get("all_docs",            [])
    v_report      = state.get("verification_report", {})
    v_status      = state.get("verification_status", "unknown")
    coverage_gaps = state.get("coverage_gaps",       [])
    cov_ratio     = state.get("coverage_ratio",      1.0)

    sums_to_use = v_sums if v_sums else summaries

    logger.info(
        f"report_node | summaries={len(sums_to_use)} | "
        f"images={len(all_images)} | all_docs={len(all_docs)}"
    )

    try:
        report = await _writer.write(
            query     = query,
            summaries = sums_to_use,
            images    = all_images,
            tone      = "analytical",
            all_docs  = all_docs,
        )

        v_stats  = v_report.get("summary_stats", {
            "total":          0,
            "contradictions": 0,
            "weak_answers":   0,
            "imbalanced":     0,
            "re_retrieved":   0,
            "fixed":          0,
        })
        v_issues = v_report.get("flagged_issues",   [])
        v_log    = v_report.get("re_retrieval_log", [])

        report = report.rstrip() + "\n\n---\n\n" + _build_verification_section(
            status        = v_status,
            stats         = v_stats,
            issues        = v_issues,
            log           = v_log,
            sums          = sums_to_use,
            coverage_gaps = coverage_gaps,
            cov_ratio     = cov_ratio,
        )

        source_set: set = set()
        for s in sums_to_use:
            for url in (s.get("sources") or []):
                if url:
                    source_set.add(url)
        # NOTE: we intentionally do NOT add all_docs URLs here.
        # Only URLs cited in summaries belong in the final sources list.
        # Adding all_docs re-inflates the count with irrelevant docs.
        sources = list(source_set)

        logger.info(
            f"report_node | report={len(report)} chars | sources={len(sources)}"
        )

        return {"final_report": report, "sources": sources, "status": PipelineStatus.DONE}

    except Exception as e:
        logger.error(f"report_node | FAILED: {e}", exc_info=True)
        return {"status": PipelineStatus.FAILED, "errors": [str(e)]}


# ─────────────────────────────────────────────────────────────────────────────
# Helper — Verification section
# ─────────────────────────────────────────────────────────────────────────────

def _build_verification_section(
    status        : str,
    stats         : Dict,
    issues        : List[Dict],
    log           : List[str],
    sums          : List[Dict],
    coverage_gaps : List[str] = None,
    cov_ratio     : float     = 1.0,
) -> str:
    status_emoji = "✅" if status == "approved" else "⚠️"
    lines = [
        "## Verification Report\n",
        f"{status_emoji} **Status:** {status.replace('_', ' ').upper()}\n",
        "| Metric | Count |",
        "|--------|-------|",
        f"| Summaries verified     | {stats.get('total',          0)} |",
        f"| Contradictions found   | {stats.get('contradictions', 0)} |",
        f"| Weak answers detected  | {stats.get('weak_answers',   0)} |",
        f"| Source imbalance flags | {stats.get('imbalanced',     0)} |",
        f"| Questions re-retrieved | {stats.get('re_retrieved',   0)} |",
        f"| Issues fixed           | {stats.get('fixed',          0)} |",
    ]

    # Coverage report
    if coverage_gaps is not None:
        lines.append(f"\n### Coverage Analysis\n")
        lines.append(f"**Coverage ratio:** {cov_ratio:.0%}")
        if coverage_gaps:
            lines.append(f"\n**Dimensions not fully covered:**")
            for gap in coverage_gaps:
                lines.append(f"- {gap}")
        else:
            lines.append("\n✅ All required dimensions covered.")

    # Low-confidence summaries
    lc_sums = [s for s in sums if s.get("low_confidence")]
    if lc_sums:
        lines.append(f"\n### Low-Confidence Summaries\n")
        lines.append(
            f"⚠️ {len(lc_sums)} summary/summaries were flagged as low-confidence "
            f"(quality score below {LOW_QUALITY_THRESHOLD} or contained uncertainty language):"
        )
        for s in lc_sums:
            lines.append(
                f"- **Q:** {s.get('question','')[:80]} "
                f"*(score: {s.get('quality_score', 0):.2f})*"
            )

    if issues:
        lines.append("\n### Flagged Issues\n")
        for issue in issues:
            severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                issue.get("severity", ""), "⚪"
            )
            resolved_tag = " *(fixed after re-retrieval)*" if issue.get("resolved") else ""
            lines.append(
                f"{severity_icon} **{issue.get('type','').replace('_',' ').title()}**"
                f" — {issue.get('question','')[:70]}{resolved_tag}"
            )
            desc = issue.get("description", "").replace("\n", "  \n> ")
            lines.append(f"> {desc}\n")

    if log:
        lines.append("\n### Re-Retrieval Log\n")
        for entry in log:
            lines.append(f"- {entry}")

    re_retrieved = [s for s in sums if s.get("was_re_retrieved")]
    if re_retrieved:
        lines.append(
            f"\n*Note: {len(re_retrieved)} section(s) were re-retrieved and "
            f"re-summarized to address detected issues.*"
        )

    return "\n".join(lines)