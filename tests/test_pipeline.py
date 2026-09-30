"""
Basic tests — run with: pytest tests/
These are unit tests that mock OpenAI so they run without an API key.
"""

import pytest
from unittest.mock import patch, MagicMock

from core.router import route_query, Route
from core.sql_tool import _is_safe, execute_sql, SQLResult
from core.self_rag import check_relevance


# ── Router tests ──────────────────────────────────────────────────────────────

def _mock_route(route_str: str):
    """Return a mock OpenAI response with the given route."""
    mock = MagicMock()
    mock.choices[0].message.content = f'{{"route": "{route_str}", "reason": "test"}}'
    return mock


@patch("core.router.OpenAI")
def test_router_returns_sql(mock_openai):
    mock_openai.return_value.chat.completions.create.return_value = _mock_route("sql")
    route, reason = route_query("What is the total salary in engineering?")
    assert route == Route.SQL
    assert reason == "test"


@patch("core.router.OpenAI")
def test_router_returns_vector(mock_openai):
    mock_openai.return_value.chat.completions.create.return_value = _mock_route("vector")
    route, _ = route_query("What is the remote work policy?")
    assert route == Route.VECTOR


@patch("core.router.OpenAI")
def test_router_fallback_on_bad_json(mock_openai):
    mock = MagicMock()
    mock.choices[0].message.content = "NOT JSON"
    mock_openai.return_value.chat.completions.create.return_value = mock
    route, _ = route_query("some question")
    assert route == Route.BOTH


# ── SQL safety tests ──────────────────────────────────────────────────────────

def test_safe_select():
    assert _is_safe("SELECT * FROM employees WHERE salary > 100000") is True


def test_blocks_delete():
    assert _is_safe("DELETE FROM employees") is False


def test_blocks_drop():
    assert _is_safe("DROP TABLE employees") is False


def test_blocks_insert():
    assert _is_safe("INSERT INTO employees VALUES (1,'x',1,'eng',100,'2020-01-01')") is False


def test_execute_sql_with_real_db(tmp_path):
    import sqlite3
    db = tmp_path / "test.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE t (id INTEGER, val TEXT)")
    conn.execute("INSERT INTO t VALUES (1, 'hello')")
    conn.commit()
    conn.close()

    result = execute_sql("SELECT * FROM t", str(db))
    assert result.success
    assert result.rows == [{"id": 1, "val": "hello"}]


# ── Self-RAG tests ────────────────────────────────────────────────────────────

def _mock_relevance(is_suff: bool, conf: float):
    mock = MagicMock()
    mock.choices[0].message.content = (
        f'{{"is_sufficient": {str(is_suff).lower()}, '
        f'"confidence": {conf}, "reasoning": "test reason"}}'
    )
    return mock


@patch("core.self_rag.OpenAI")
def test_relevance_high_confidence(mock_openai):
    mock_openai.return_value.chat.completions.create.return_value = _mock_relevance(True, 0.9)
    result = check_relevance("What is the budget?", "The budget is $2.5M")
    assert result.is_sufficient is True
    assert result.confidence == 0.9


@patch("core.self_rag.OpenAI")
def test_relevance_low_confidence(mock_openai):
    mock_openai.return_value.chat.completions.create.return_value = _mock_relevance(False, 0.2)
    result = check_relevance("What is the budget?", "The cat sat on the mat.")
    assert result.is_sufficient is False
    assert result.confidence == 0.2
