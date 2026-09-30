"""
Hybrid RAG Pipeline — the top-level orchestrator.

Flow:
  1. Router  → decide SQL / Vector / Both
  2. Retrieve → query SQL tool and/or vector store
  3. Self-RAG → check if context is sufficient
  4. Generate → produce final answer with citations
"""

from dataclasses import dataclass, field
from typing import Optional

from langchain_community.vectorstores import FAISS

from core.router import route_query, Route
from core.sql_tool import answer_from_sql, SQLResult
from core.vector_store import retrieve, VectorResult
from core.self_rag import check_relevance, RelevanceCheck
from core.generator import generate_answer, GeneratedAnswer
from core.db_observer import SchemaContext


@dataclass
class PipelineResult:
    question: str
    route: Route
    route_reason: str
    relevance: RelevanceCheck
    answer: GeneratedAnswer
    sql_result: Optional[SQLResult] = None
    vector_results: list[VectorResult] = field(default_factory=list)


def run_pipeline(
    question: str,
    schema: Optional[SchemaContext],
    faiss_index: Optional[FAISS],
    db_path: Optional[str],
) -> PipelineResult:
    """
    Execute the full Hybrid RAG pipeline for a user question.

    Args:
        question:    The user's natural language question.
        schema:      SchemaContext from db_observer (None if no DB loaded).
        faiss_index: Loaded FAISS index (None if no documents loaded).
        db_path:     Path to the SQLite file (None if no DB loaded).

    Returns:
        PipelineResult with all intermediate artefacts for UI display.
    """

    # ── Step 1: Route ────────────────────────────────────────────────────────
    # Override route if only one source is available
    has_db  = schema is not None and db_path is not None
    has_vec = faiss_index is not None

    raw_route, reason = route_query(question)

    if raw_route == Route.SQL    and not has_db:  raw_route = Route.VECTOR
    if raw_route == Route.VECTOR and not has_vec: raw_route = Route.SQL
    if raw_route == Route.BOTH   and not has_db:  raw_route = Route.VECTOR
    if raw_route == Route.BOTH   and not has_vec: raw_route = Route.SQL

    # ── Step 2: Retrieve ─────────────────────────────────────────────────────
    sql_result:     Optional[SQLResult]     = None
    vector_results: list[VectorResult]      = []

    if raw_route in (Route.SQL, Route.BOTH) and has_db:
        sql_result = answer_from_sql(question, schema, db_path)

    if raw_route in (Route.VECTOR, Route.BOTH) and has_vec:
        vector_results = retrieve(question, faiss_index)

    # ── Step 3: Self-RAG check ───────────────────────────────────────────────
    context_preview = ""
    if vector_results:
        context_preview += "\n".join(vr.content for vr in vector_results[:3])
    if sql_result and sql_result.success:
        context_preview += "\n" + sql_result.as_text()

    relevance = check_relevance(question, context_preview)

    # ── Step 4: Generate ─────────────────────────────────────────────────────
    answer = generate_answer(
        question=question,
        vector_results=vector_results or None,
        sql_result=sql_result,
    )

    return PipelineResult(
        question=question,
        route=raw_route,
        route_reason=reason,
        relevance=relevance,
        answer=answer,
        sql_result=sql_result,
        vector_results=vector_results,
    )
