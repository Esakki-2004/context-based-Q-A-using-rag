"""
Agentic Router.

Classifies a user query into one of three routes:
  - "sql"    → structured data question (numbers, counts, filters, dates)
  - "vector" → unstructured text search (definitions, descriptions, policies)
  - "both"   → hybrid (needs facts from DB + context from documents)

Uses a lightweight LLM call with a strict JSON response schema.
"""

import json
from enum import Enum

from openai import OpenAI

from core.config import OPENAI_API_KEY, CHAT_MODEL


class Route(str, Enum):
    SQL    = "sql"
    VECTOR = "vector"
    BOTH   = "both"


_SYSTEM_PROMPT = """
You are a query router for a hybrid RAG system that has two information sources:
1. SQL database — contains structured/tabular data (numbers, records, statistics, dates, IDs).
2. Vector DB — contains unstructured text (PDF documents, policy files, markdown notes).

Given a user question, respond with ONLY a JSON object like:
{"route": "sql" | "vector" | "both", "reason": "<one sentence>"}

Rules:
- "sql"    → question asks for specific numbers, aggregations, records, or comparisons from data.
- "vector" → question asks about explanations, policies, definitions, or content from documents.
- "both"   → question clearly needs BOTH structured data AND document context to answer fully.
""".strip()


def route_query(question: str) -> tuple[Route, str]:
    """
    Returns (Route, reason_string).
    Falls back to 'both' if the LLM response cannot be parsed.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ],
        max_tokens=100,
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    try:
        data = json.loads(raw)
        route = Route(data.get("route", "both"))
        reason = data.get("reason", "")
        return route, reason
    except (json.JSONDecodeError, ValueError):
        return Route.BOTH, "Could not parse route; defaulting to hybrid."
