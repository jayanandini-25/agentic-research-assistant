"""
API Routes — Phase 12
=====================================================================
File location:  api/routes.py

KEY CHANGES vs Phase 11:
  FIX 1 — deep_depth and deep_breadth are no longer read from settings
           or inflated by explicit_deep flag. Phase 12 DeepResearchAgent
           is hardcoded to 3 root + 1 subtopic × 2 = 5 questions max.
           These parameters are removed from initial_state entirely.

  FIX 2 — _build_message updated to not reference deep_depth/deep_breadth.

  EVAL  — run_auto_eval() is called automatically at the end of every
           successful start_research() run. Generates a .docx evaluation
           report to eval_reports/ and prints metrics to the console.
           If eval_metrics.py is missing, the endpoint still works fine.
"""

import os
import re
import uuid
import tempfile
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from core.logger import setup_logger
from app.graph.pipeline import research_graph
from app.graph.state    import PipelineStatus
from config.settings    import get_settings

from app.export.pdf_exporter  import PDFExporter
from app.export.docx_exporter import DOCXExporter

try:
    from core.models import (
        ResearchRequest, ResearchResponse, ResearchStatus, ImageResult
    )
except ImportError:
    from core.models import ResearchRequest, ResearchResponse, ResearchStatus, ImageResult

settings = get_settings()
router   = APIRouter()
logger   = setup_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Eval hook — imported once at startup, gracefully absent if file is missing
# ─────────────────────────────────────────────────────────────────────────────

try:
    from eval_metrics import run_auto_eval as _run_auto_eval
    _EVAL_ENABLED = True
    logger.info("eval_metrics.py found — evaluation will run after every pipeline call.")
except ImportError:
    _EVAL_ENABLED = False
    logger.info("eval_metrics.py not found — skipping post-pipeline evaluation.")


def _maybe_eval(session_data: dict) -> None:
    """
    Called at the end of start_research() with the full response dict.
    Runs eval asynchronously in a thread so the HTTP response is not delayed.
    Failures are caught and logged — they never affect the API response.
    """
    if not _EVAL_ENABLED:
        return
    import threading
    def _run():
        try:
            _run_auto_eval(session_data)
        except Exception as exc:
            logger.warning(f"[eval] Post-pipeline evaluation error (non-fatal): {exc}")
    threading.Thread(target=_run, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# Export request model
# ─────────────────────────────────────────────────────────────────────────────

class ExportRequest(BaseModel):
    report : str
    title  : Optional[str] = "Research Report"


# ─────────────────────────────────────────────────────────────────────────────
# Helper — Build rich cited_sources for Sources tab
# ─────────────────────────────────────────────────────────────────────────────

def _build_cited_sources(
    all_docs    : List[Dict],
    sources_raw : List[str],
) -> List[Dict[str, Any]]:
    url_to_doc: Dict[str, Dict] = {}
    for doc in all_docs:
        url = doc.get("url", "")
        if url and url not in url_to_doc:
            url_to_doc[url] = doc

    def extract_domain(url: str) -> str:
        try:
            m = re.search(r"https?://(?:www\.)?([^/]+)", url)
            return m.group(1) if m else ""
        except Exception:
            return ""

    cited: List[Dict[str, Any]] = []
    seen:  set                   = set()

    for url in sources_raw:
        if not url or url in seen:
            continue
        seen.add(url)
        doc   = url_to_doc.get(url, {})
        title = (doc.get("title") or url).strip()
        cited.append({
            "num"   : len(cited) + 1,
            "title" : title[:120],
            "url"   : url,
            "domain": extract_domain(url),
            "source": doc.get("source", ""),
        })

    for doc in all_docs:
        url = doc.get("url", "")
        if not url or url in seen:
            continue
        seen.add(url)
        title = (doc.get("title") or url).strip()
        cited.append({
            "num"   : len(cited) + 1,
            "title" : title[:120],
            "url"   : url,
            "domain": extract_domain(url),
            "source": doc.get("source", ""),
        })

    return cited


# ─────────────────────────────────────────────────────────────────────────────
# Helper — Flatten RAG chunks
# ─────────────────────────────────────────────────────────────────────────────

def _flatten_rag_chunks(filtered_map: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for question, docs in (filtered_map or {}).items():
        for doc in (docs or []):
            chunk              = dict(doc)
            chunk["_question"] = question
            if "rag_score" not in chunk:
                chunk["rag_score"] = chunk.get("score", chunk.get("similarity", 0.0))
            chunks.append(chunk)
    chunks.sort(key=lambda x: x.get("rag_score", 0), reverse=True)
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Helper — Build eval session dict from final pipeline state
# ─────────────────────────────────────────────────────────────────────────────

def _build_eval_session(
    session_id     : str,
    query          : str,
    sub_questions  : List[str],
    all_docs       : List[Dict],
    source_counts  : Dict,
    filtered_map   : Dict,
    summaries      : List[Dict],
    v_sums         : List[Dict],
    report         : str,
    sources_raw    : List[str],
    timing         : Dict,
    token_counts   : Dict,
) -> dict:
    """
    Assembles the session data dict that eval_metrics.py understands.
    Maps your pipeline's field names to what eval_metrics expects.
    """
    # Build rag_log from filtered_map
    per_question = []
    total_before = 0
    total_after  = 0
    for q in sub_questions:
        docs = filtered_map.get(q, [])
        scores = [float(d.get("rag_score", d.get("score", d.get("similarity", 0)))) for d in docs]
        per_question.append({
            "question": q,
            "scores":   scores,
        })
        total_after += len(docs)
    # rough total_before estimate (each source doc goes through RAG)
    total_before = len(all_docs) if all_docs else total_after

    # Build summaries list in expected format
    sums_for_eval = v_sums if v_sums else summaries
    eval_summaries = []
    for s in sums_for_eval:
        eval_summaries.append({
            "question": s.get("question", ""),
            "answer":   s.get("answer",   s.get("summary", "")),
            "sources":  s.get("sources",  []),
            "quality":  float(s.get("quality", s.get("quality_score", 1.0))),
            "attempts": int(s.get("attempts", s.get("retries", 0)) + 1),
        })

    return {
        "session_id":    session_id,
        "query":         query,
        "sub_questions": sub_questions,
        "planner_log": {
            "dims_covered":  len(sub_questions),
            "total_dims":    max(len(sub_questions), 10),
            "critical_gaps": [],
        },
        "retrieval_log": {
            "source_counts": source_counts,
        },
        "rag_log": {
            "total_chunks_before": total_before,
            "total_chunks_after":  total_after,
            "per_question":        per_question,
        },
        "summaries": eval_summaries,
        "report":    report,
        "sources":   sources_raw,
        "timing":    timing,
        "token_counts": token_counts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helper — Build status message
# ─────────────────────────────────────────────────────────────────────────────

def _build_message(
    is_deep        : bool,
    judge_decision : str,
    sub_questions  : List[str],
    all_docs       : List[Dict],
    all_images     : List[Dict],
    source_counts  : Dict,
    filtered_map   : Dict,
    summaries      : List[Dict],
    v_sums         : List[Dict],
    v_status       : str,
    v_stats        : Dict,
    v_issues       : List,
    report         : str,
    deep_result    : Dict,
    depth_reached  : int,
    dense_topics   : List[str],
) -> str:
    total_docs     = len(all_docs)
    total_filtered = sum(len(d) for d in (filtered_map or {}).values())
    source_names   = ", ".join(sorted(source_counts.keys())) if source_counts else "N/A"
    sums_for_stats = v_sums if v_sums else summaries

    lines: List[str] = []

    if is_deep:
        lines = [
            f"🔬 Deep Research triggered (decision: {judge_decision})",
            f"Depth reached   : {depth_reached}/1 | subtopics=1 max",
            f"Total questions : {deep_result.get('total_questions', len(sums_for_stats))} (3 root + up to 2 sub-topic)",
            f"Total documents : {deep_result.get('total_docs', total_docs)}",
            f"Total summaries : {len(sums_for_stats)}",
            "",
        ]
        if dense_topics:
            lines.append("Dense sub-topics discovered:")
            for t in dense_topics:
                lines.append(f"  • {t}")
            lines.append("")
        lines.append(f"Report  — {len(report)} characters")
        lines.append(f"Images  — {len(all_images)} images")
    else:
        lines = [
            f"Depth judge decision : {judge_decision.upper()}",
            f"Planning complete    — {len(sub_questions)} sub-questions generated.",
            f"Retrieval complete   — {total_docs} documents from: {source_names}",
            f"RAG filtering        — {total_docs} → {total_filtered} relevant chunks",
            f"Summarization        — {len(sums_for_stats)} focused summaries generated",
            f"Verification         — {v_status.upper()} | "
            f"issues={len(v_issues)} | "
            f"re_retrieved={v_stats.get('re_retrieved', 0)} | "
            f"fixed={v_stats.get('fixed', 0)}",
            f"Report written       — {len(report)} characters",
            f"Images retrieved     — {len(all_images)} images",
            "",
            "Sub-questions researched:",
        ]
        for i, q in enumerate(sub_questions, 1):
            chunk_count = len((filtered_map or {}).get(q, []))
            v_item      = next(
                (x for x in sums_for_stats if x.get("question") == q), None
            )
            re_tag      = " [RE-RETRIEVED]" if v_item and v_item.get("was_re_retrieved") else ""
            issues_tag  = (
                f" [{len(v_item.get('issues', []))} issue(s)]"
                if v_item and v_item.get("issues") else ""
            )
            lines.append(f"  {i}. {q}  [{chunk_count} chunks{re_tag}{issues_tag}]")

    if all_images:
        lines.append("")
        lines.append("Top images found:")
        for img in all_images[:3]:
            lines.append(
                f"  [{img.get('source','')}] {img.get('alt','')[:60]} "
                f"| score={img.get('score',0):.2f} | {img.get('url','')[:70]}"
            )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main Research Endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/research")
async def start_research(request: ResearchRequest):
    session_id = str(uuid.uuid4())

    logger.info(
        f"Session started | id={session_id} | query='{request.query}'"
    )

    try:
        initial_state = {
            "research_query"       : request.query,
            "session_id"           : session_id,
            "sub_questions"        : [],
            "docs_by_question"     : {},
            "all_docs"             : [],
            "all_images"           : [],
            "source_counts"        : {},
            "filtered_map"         : {},
            "summaries"            : [],
            "verified_summaries"   : [],
            "verification_report"  : {},
            "verification_status"  : "pending",
            "depth_judge_decision" : "sufficient",
            "final_report"         : "",
            "sources"              : [],
            "status"               : PipelineStatus.PLANNING,
            "errors"               : [],
            "iteration_count"      : 0,
            "is_deep_research"     : False,
            "deep_research_result" : {},
            "research_tree"        : {},
            "dense_topics_found"   : [],
            "depth_reached"        : 0,
        }

        logger.info(f"Session {session_id} | invoking pipeline")
        final_state = await research_graph.ainvoke(initial_state)
        logger.info(
            f"Session {session_id} | pipeline done | "
            f"status={final_state.get('status')}"
        )

        if final_state.get("status") == PipelineStatus.FAILED:
            errors = final_state.get("errors", ["Unknown pipeline failure"])
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline failed: {'; '.join(errors)}"
            )

        # ── Unpack final state ─────────────────────────────────────────────
        sub_questions  = final_state.get("sub_questions",        [])
        all_docs       = final_state.get("all_docs",             [])
        all_images     = final_state.get("all_images",           [])
        source_counts  = final_state.get("source_counts",        {})
        filtered_map   = final_state.get("filtered_map",         {})
        summaries      = final_state.get("summaries",            [])
        v_sums         = final_state.get("verified_summaries",   [])
        v_report       = final_state.get("verification_report",  {})
        v_status       = final_state.get("verification_status",  "unknown")
        report         = final_state.get("final_report",         "")
        sources_raw    = final_state.get("sources",              [])
        docs_by_q      = final_state.get("docs_by_question",     {})
        judge_decision = final_state.get("depth_judge_decision", "sufficient")
        is_deep        = final_state.get("is_deep_research",     False)

        deep_result        = final_state.get("deep_research_result", {})
        research_tree      = final_state.get("research_tree",        {})
        dense_topics_found = final_state.get("dense_topics_found",   [])
        depth_reached      = final_state.get("depth_reached",        0)

        if is_deep and deep_result:
            if not research_tree:
                research_tree = deep_result.get("research_tree", {})
            if not dense_topics_found:
                dense_topics_found = deep_result.get("dense_topics_found", [])
            if not depth_reached:
                depth_reached = deep_result.get("depth_reached", 0)
            if not all_images:
                all_images = deep_result.get("all_images", [])

        v_stats  = v_report.get("summary_stats", {
            "total": 0, "contradictions": 0, "weak_answers": 0,
            "imbalanced": 0, "re_retrieved": 0, "fixed": 0,
        })
        v_issues = v_report.get("flagged_issues", [])

        summaries_for_response = v_sums if v_sums else summaries

        # ── Build tab data ─────────────────────────────────────────────────
        cited_sources = _build_cited_sources(all_docs, sources_raw)
        rag_chunks    = _flatten_rag_chunks(filtered_map)

        logger.info(
            f"Session {session_id} | "
            f"judge={judge_decision} | is_deep={is_deep} | "
            f"summaries={len(summaries_for_response)} | "
            f"sources={len(cited_sources)} | "
            f"rag_chunks={len(rag_chunks)} | "
            f"images={len(all_images)}"
        )

        message = _build_message(
            is_deep        = is_deep,
            judge_decision = judge_decision,
            sub_questions  = sub_questions,
            all_docs       = all_docs,
            all_images     = all_images,
            source_counts  = source_counts,
            filtered_map   = filtered_map,
            summaries      = summaries,
            v_sums         = v_sums,
            v_status       = v_status,
            v_stats        = v_stats,
            v_issues       = v_issues,
            report         = report,
            deep_result    = deep_result,
            depth_reached  = depth_reached,
            dense_topics   = dense_topics_found,
        )

        image_results = []
        for img in all_images:
            if not isinstance(img, dict):
                continue
            image_results.append(ImageResult(
                source     = img.get("source",     ""),
                url        = img.get("url",        ""),
                alt        = img.get("alt",        ""),
                page_url   = img.get("page_url",   ""),
                width      = img.get("width",       0),
                height     = img.get("height",      0),
                score      = float(img.get("score", 0.0)),
                caption    = img.get("caption",    ""),
                confidence = img.get("confidence", "medium"),
                scored_by  = img.get("scored_by",  "lexical"),
                domain     = img.get("domain",     ""),
            ))

        response_data = {
            "session_id"           : session_id,
            "query"                : request.query,
            "status"               : ResearchStatus.done,
            "message"              : message,
            "report"               : report,
            "sources"              : sources_raw,
            "images"               : image_results,
            "retrieved_docs"       : docs_by_q,
            "summaries"            : summaries_for_response,
            "cited_sources"        : cited_sources,
            "rag_chunks"           : rag_chunks,
            "verification_report"  : v_report,
            "verification_status"  : v_status,
            "deep_research_result" : deep_result    if is_deep else {},
            "research_tree"        : research_tree  if is_deep else {},
            "dense_topics_found"   : dense_topics_found if is_deep else [],
            "depth_reached"        : depth_reached  if is_deep else 0,
            # Phase 13: Pipeline telemetry for frontend dashboard
            "source_counts"        : source_counts,
            "planner_meta"         : final_state.get("planner_meta", {}),
        }

        # ── Auto-eval hook ─────────────────────────────────────────────────
        # Runs in a background thread so the HTTP response is returned
        # immediately. Eval report saved to eval_reports/ automatically.
        _maybe_eval(_build_eval_session(
            session_id    = session_id,
            query         = request.query,
            sub_questions = sub_questions,
            all_docs      = all_docs,
            source_counts = source_counts,
            filtered_map  = filtered_map,
            summaries     = summaries,
            v_sums        = v_sums,
            report        = report,
            sources_raw   = sources_raw,
            timing        = final_state.get("timing", {}),
            token_counts  = final_state.get("token_counts", {}),
        ))

        try:
            return ResearchResponse(**response_data)
        except TypeError:
            from fastapi.responses import JSONResponse
            response_data["images"] = [
                {
                    "source": img.source, "url": img.url, "alt": img.alt,
                    "page_url": img.page_url, "width": img.width,
                    "height": img.height, "score": img.score,
                    "caption": img.caption, "confidence": img.confidence,
                    "scored_by": img.scored_by, "domain": img.domain,
                }
                for img in image_results
            ]
            response_data["status"] = "done"
            return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session {session_id} failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# PDF Export
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/export/pdf")
async def export_pdf(request: ExportRequest):
    if not request.report or not request.report.strip():
        raise HTTPException(status_code=400, detail="report field is required")

    tmp_dir  = tempfile.mkdtemp()
    out_path = os.path.join(tmp_dir, "report.pdf")
    logger.info(f"PDF export | title='{(request.title or '')[:50]}'")

    try:
        PDFExporter().export(
            markdown_report=request.report,
            output_path=out_path,
            title=request.title or "Research Report",
        )
        return FileResponse(
            path=out_path, media_type="application/pdf",
            filename="research_report.pdf",
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"export_pdf failed: {e}")
        raise HTTPException(status_code=500, detail=f"PDF export failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# DOCX Export
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/export/docx")
async def export_docx(request: ExportRequest):
    if not request.report or not request.report.strip():
        raise HTTPException(status_code=400, detail="report field is required")

    tmp_dir  = tempfile.mkdtemp()
    out_path = os.path.join(tmp_dir, "report.docx")
    logger.info(f"DOCX export | title='{(request.title or '')[:50]}'")

    try:
        DOCXExporter().export(
            markdown_report=request.report,
            output_path=out_path,
            title=request.title or "Research Report",
        )
        return FileResponse(
            path=out_path,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename="research_report.docx",
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"export_docx failed: {e}")
        raise HTTPException(status_code=500, detail=f"DOCX export failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/health")
async def health_check():
    return {
        "status" : "ok",
        "service": "agentic-research-assistant",
        "phase"  : "12",
    }