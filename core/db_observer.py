"""
Database Observer — "self-discovery" module.

Connects to a SQLite file in READ-ONLY mode, introspects every table
via PRAGMA, and produces:
  1. A structured SchemaContext (used by the SQL tool).
  2. A plain-English natural language summary (shown in the UI on upload).
"""

import sqlite3
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI

from core.config import OPENAI_API_KEY, CHAT_MODEL


@dataclass
class ColumnInfo:
    name: str
    type: str
    not_null: bool
    primary_key: bool


@dataclass
class TableSchema:
    name: str
    columns: list[ColumnInfo] = field(default_factory=list)
    foreign_keys: list[dict] = field(default_factory=list)
    sample_rows: list[dict] = field(default_factory=list)


@dataclass
class SchemaContext:
    db_path: str
    tables: list[TableSchema] = field(default_factory=list)
    nl_summary: str = ""

    def as_prompt_context(self) -> str:
        """Compact DDL-style string suitable for injecting into an LLM prompt."""
        lines = []
        for t in self.tables:
            cols = ", ".join(
                f"{c.name} {c.type}{'  PK' if c.primary_key else ''}"
                for c in t.columns
            )
            lines.append(f"TABLE {t.name} ({cols})")
            if t.foreign_keys:
                for fk in t.foreign_keys:
                    lines.append(
                        f"  FK {fk['from']} -> {fk['table']}.{fk['to']}"
                    )
            if t.sample_rows:
                lines.append(f"  SAMPLE: {t.sample_rows[0]}")
        return "\n".join(lines)


def observe_database(db_path: str) -> SchemaContext:
    """
    Open the database READ-ONLY, collect full schema metadata + sample rows.
    Never requests write or delete permissions.
    """
    uri = f"file:{db_path}?mode=ro"
    ctx = SchemaContext(db_path=db_path)

    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # List all user tables
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        table_names = [r["name"] for r in cursor.fetchall()]

        for tname in table_names:
            ts = TableSchema(name=tname)

            # Column metadata
            cursor.execute(f"PRAGMA table_info('{tname}')")
            for row in cursor.fetchall():
                ts.columns.append(
                    ColumnInfo(
                        name=row["name"],
                        type=row["type"],
                        not_null=bool(row["notnull"]),
                        primary_key=bool(row["pk"]),
                    )
                )

            # Foreign key relationships
            cursor.execute(f"PRAGMA foreign_key_list('{tname}')")
            for row in cursor.fetchall():
                ts.foreign_keys.append(
                    {"from": row["from"], "table": row["table"], "to": row["to"]}
                )

            # 3 sample rows for context
            try:
                cursor.execute(f"SELECT * FROM '{tname}' LIMIT 3")
                ts.sample_rows = [dict(r) for r in cursor.fetchall()]
            except Exception:
                pass

            ctx.tables.append(ts)

    ctx.nl_summary = _generate_nl_summary(ctx)
    return ctx


def _generate_nl_summary(ctx: SchemaContext) -> str:
    """Ask the LLM to describe the database in plain English."""
    if not ctx.tables:
        return "The database appears to be empty."

    table_names = ", ".join(t.name for t in ctx.tables)
    fallback_summary = f"Database contains {len(ctx.tables)} tables: {table_names}."

    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-..."):
        return f"{fallback_summary} (Add a valid OPENAI_API_KEY to .env for AI summary)"

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        schema_text = ctx.as_prompt_context()

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful data analyst. Given a database schema, "
                        "describe what the database is about in 2-3 sentences. "
                        "Mention what kind of data it stores and any key relationships."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Describe this database:\n\n{schema_text}",
                },
            ],
            max_tokens=200,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"{fallback_summary} (AI summary unavailable: {e})"

