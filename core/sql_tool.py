"""
Text-to-SQL tool.

Given a natural language question and a SchemaContext, generates a SELECT
query, executes it safely (read-only connection), and returns the result
alongside the SQL used — for transparent citations.
"""

import sqlite3
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from core.config import OPENAI_API_KEY, CHAT_MODEL
from core.db_observer import SchemaContext


@dataclass
class SQLResult:
    sql: str
    rows: list[dict[str, Any]]
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None

    def as_text(self) -> str:
        if not self.success:
            return f"SQL error: {self.error}"
        if not self.rows:
            return "Query returned no results."
        # Simple tabular text representation
        headers = list(self.rows[0].keys())
        lines = [" | ".join(headers)]
        lines.append("-" * len(lines[0]))
        for row in self.rows:
            lines.append(" | ".join(str(v) for v in row.values()))
        return "\n".join(lines)


_DISALLOWED = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|ATTACH|DETACH)\b",
    re.IGNORECASE,
)


def _is_safe(sql: str) -> bool:
    return not bool(_DISALLOWED.search(sql))


def generate_sql(question: str, schema: SchemaContext) -> str:
    """Ask the LLM to produce a SELECT query for the given question."""
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert SQLite query writer. "
                    "Given a database schema and a user question, write a single "
                    "valid SQLite SELECT statement that answers the question. "
                    "Return ONLY the SQL — no explanation, no markdown fences."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Schema:\n{schema.as_prompt_context()}\n\n"
                    f"Question: {question}"
                ),
            },
        ],
        max_tokens=300,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def execute_sql(sql: str, db_path: str) -> SQLResult:
    """Execute a SQL query in read-only mode and return results."""
    if not _is_safe(sql):
        return SQLResult(sql=sql, rows=[], error="Query blocked: only SELECT is permitted.")

    uri = f"file:{db_path}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = [dict(r) for r in cursor.fetchall()]
        return SQLResult(sql=sql, rows=rows)
    except sqlite3.Error as e:
        return SQLResult(sql=sql, rows=[], error=str(e))


def answer_from_sql(question: str, schema: SchemaContext, db_path: str) -> SQLResult:
    """Full pipeline: generate SQL → validate → execute."""
    sql = generate_sql(question, schema)
    return execute_sql(sql, db_path)
