"""
Answer generator.

Takes the assembled context (from SQL + Vector) and generates a final
response with inline citations.

Citations format:
  [PDF: filename.pdf, p.4]   — for vector chunks
  [SQL: SELECT ...]           — for database results
"""

from dataclasses import dataclass

from openai import OpenAI

from core.config import OPENAI_API_KEY, CHAT_MODEL
from core.vector_store import VectorResult
from core.sql_tool import SQLResult


@dataclass
class GeneratedAnswer:
    answer: str
    citations: list[str]   # human-readable citation strings
    context_used: str      # raw context injected into the prompt (for debugging)


_SYSTEM_PROMPT = """
You are a helpful assistant answering questions using provided context.

Rules:
- Answer ONLY from the provided context. Do not add outside knowledge.
- If the context is insufficient, clearly say so.
- When referencing information, use inline citation tags like [1], [2], etc.
  matching the numbered sources provided.
- Be concise and accurate.
""".strip()


def generate_answer(
    question: str,
    vector_results: list[VectorResult] | None = None,
    sql_result: SQLResult | None = None,
) -> GeneratedAnswer:
    """Assemble context, build citations, and generate the final answer."""
    context_parts: list[str] = []
    citations: list[str] = []
    idx = 1

    # Add vector context chunks
    if vector_results:
        for vr in vector_results:
            page_str = f", p.{vr.page + 1}" if vr.page is not None else ""
            citation = f"[{idx}] PDF: {vr.source}{page_str}"
            context_parts.append(f"[{idx}] (from {vr.source}{page_str}):\n{vr.content}")
            citations.append(citation)
            idx += 1

    # Add SQL result
    if sql_result and sql_result.success and sql_result.rows:
        citation = f"[{idx}] SQL: `{sql_result.sql}`"
        context_parts.append(
            f"[{idx}] (from database query):\n{sql_result.as_text()}"
        )
        citations.append(citation)
        idx += 1

    if not context_parts:
        return GeneratedAnswer(
            answer="I could not find relevant information to answer your question.",
            citations=[],
            context_used="",
        )

    full_context = "\n\n".join(context_parts)

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context:\n{full_context}\n\n"
                    f"Question: {question}"
                ),
            },
        ],
        max_tokens=600,
        temperature=0.2,
    )

    return GeneratedAnswer(
        answer=response.choices[0].message.content.strip(),
        citations=citations,
        context_used=full_context,
    )
