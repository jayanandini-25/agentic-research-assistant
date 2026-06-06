"""
LangGraph Pipeline Builder — Phase 12 (Coverage-Based Depth Judge)
=====================================================================
File location:  app/graph/pipeline.py

Changes vs Phase 11:
  - depth_judge_node registered as a new node between verifier and report/deep.
  - after_start always routes to planner_node (no more entry-point deep branch).
  - after_verifier routes to depth_judge_node (not report_node directly).
  - after_judge routes to either deep_research_node or report_node.
  - after_deep_research routes to report_node (not END) so the writer
    can produce a unified report over merged summaries.
  - Re-retrieval loop removed from edges (handled inside verifier_node).

Graph structure:

    START
      │
      ▼
    planner_node
      │
    retriever_node
      │
    rag_node
      │
    summarizer_node
      │
    verifier_node
      │
    depth_judge_node          ← LLM reads summaries, decides depth
      │
      ├── "sufficient"        ──► report_node
      │
      └── "needs_deep_research" ► deep_research_node
                                        │
                                        ▼
                                   report_node  (writes unified report)
                                        │
                                       END
"""

from langgraph.graph import StateGraph, END

from app.graph.state import ResearchState
from app.graph.nodes import (
    planner_node,
    retriever_node,
    rag_node,
    summarizer_node,
    verifier_node,
    depth_judge_node,
    deep_research_node,
    report_node,
)
from app.graph.edges import (
    after_start,
    after_planner,
    after_retriever,
    after_rag,
    after_summarizer,
    after_verifier,
    after_judge,
    after_deep_research,
    after_report,
)


def build_research_graph():
    """
    Builds and compiles the full research pipeline as a LangGraph StateGraph.
    Returns a compiled graph ready for .ainvoke().
    """
    graph = StateGraph(ResearchState)

    # ── Register nodes ─────────────────────────────────────────────────────────
    graph.add_node("planner_node",      planner_node)       # Phase 2
    graph.add_node("retriever_node",    retriever_node)     # Phase 3
    graph.add_node("rag_node",          rag_node)           # Phase 6
    graph.add_node("summarizer_node",   summarizer_node)    # Phase 8
    graph.add_node("verifier_node",     verifier_node)      # Phase 7
    graph.add_node("depth_judge_node",  depth_judge_node)   # NEW: auto deep research
    graph.add_node("deep_research_node", deep_research_node) # Phase 10
    graph.add_node("report_node",       report_node)        # Phase 11

    # ── Entry point — always planner ───────────────────────────────────────────
    graph.set_conditional_entry_point(
        after_start,
        {"planner_node": "planner_node"},
    )

    # ── Standard pipeline ──────────────────────────────────────────────────────
    graph.add_conditional_edges(
        "planner_node",
        after_planner,
        {"retriever_node": "retriever_node", "end": END},
    )

    graph.add_conditional_edges(
        "retriever_node",
        after_retriever,
        {"rag_node": "rag_node", "end": END},
    )

    graph.add_conditional_edges(
        "rag_node",
        after_rag,
        {"summarizer_node": "summarizer_node", "end": END},
    )

    graph.add_conditional_edges(
        "summarizer_node",
        after_summarizer,
        {"verifier_node": "verifier_node", "end": END},
    )

    graph.add_conditional_edges(
        "verifier_node",
        after_verifier,
        {
            "retriever_node"  : "retriever_node",    # re-retrieval loop
            "depth_judge_node": "depth_judge_node",  # judge after settling
            "end"             : END,
        },
    )

    # ── Depth judge ────────────────────────────────────────────────────────────
    graph.add_conditional_edges(
        "depth_judge_node",
        after_judge,
        {
            "report_node"        : "report_node",
            "deep_research_node" : "deep_research_node",
            "end"                : END,
        },
    )

    # ── Deep research → always report (not END) ────────────────────────────────
    graph.add_conditional_edges(
        "deep_research_node",
        after_deep_research,
        {"report_node": "report_node"},
    )

    # ── Report → END ───────────────────────────────────────────────────────────
    graph.add_conditional_edges(
        "report_node",
        after_report,
        {"end": END},
    )

    return graph.compile()


# Module-level singleton
research_graph = build_research_graph()