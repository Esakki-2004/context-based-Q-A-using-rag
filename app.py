"""
Hybrid RAG — Streamlit Dashboard
"""

import os
import tempfile
import streamlit as st
from pathlib import Path

from core.config import SQLITE_DB_PATH
from core.db_observer import observe_database, SchemaContext
from core.vector_store import ingest_files, save_index, load_index
from core.pipeline import run_pipeline, PipelineResult
from core.router import Route

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid RAG",
    page_icon="🔍",
    layout="wide",
)

# ── Session state defaults ────────────────────────────────────────────────────
if "schema"      not in st.session_state: st.session_state.schema      = None
if "db_path"     not in st.session_state: st.session_state.db_path     = None
if "faiss_index" not in st.session_state: st.session_state.faiss_index = load_index()
if "chat_history" not in st.session_state: st.session_state.chat_history = []


# ── Helpers ───────────────────────────────────────────────────────────────────
ROUTE_BADGE = {
    Route.SQL:    ("🗄️ SQL",    "#1a6b3a", "#d4edda"),
    Route.VECTOR: ("📄 Vector", "#0c4270", "#d0e8ff"),
    Route.BOTH:   ("⚡ Hybrid", "#5a2d82", "#ede0ff"),
}


def badge(route: Route) -> str:
    label, color, bg = ROUTE_BADGE[route]
    return (
        f'<span style="background:{bg};color:{color};'
        f'padding:2px 10px;border-radius:12px;font-size:12px;'
        f'font-weight:600">{label}</span>'
    )


def confidence_bar(score: float) -> str:
    pct  = int(score * 100)
    color = "#1a6b3a" if score >= 0.7 else "#9a6700" if score >= 0.4 else "#b91c1c"
    return (
        f'<div style="background:#e5e7eb;border-radius:6px;height:8px;width:180px;display:inline-block">'
        f'<div style="background:{color};width:{pct}%;height:100%;border-radius:6px"></div></div>'
        f' <span style="font-size:12px;color:{color};font-weight:600">{pct}%</span>'
    )


# ── Sidebar: Data Sources ─────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Hybrid RAG")
    st.caption("Context-based QA with SQL + Vector retrieval")

    from core.config import OPENAI_API_KEY
    if not OPENAI_API_KEY or OPENAI_API_KEY == "sk-..." or OPENAI_API_KEY.startswith("sk-..."):
        st.warning("⚠️ **OpenAI API Key Missing**\nPlease edit `.env` and add your valid `OPENAI_API_KEY`.")

    st.divider()

    # ── DB Upload ──
    st.subheader("🗄️ Structured Database")
    db_file = st.file_uploader("Upload a SQLite (.db) file", type=["db"])

    if db_file:
        with st.spinner("Observing database schema…"):
            tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
            tmp.write(db_file.read())
            tmp.flush()
            try:
                schema = observe_database(tmp.name)
                st.session_state.schema  = schema
                st.session_state.db_path = tmp.name
                st.success(f"✅ {len(schema.tables)} tables found")
                with st.expander("📊 Database summary", expanded=True):
                    st.write(schema.nl_summary)
                    for t in schema.tables:
                        cols = ", ".join(f"{c.name} ({c.type})" for c in t.columns)
                        st.markdown(f"**{t.name}** — {cols}")
            except Exception as e:
                st.error(f"Error reading database: {e}")

    # Use seed DB if none uploaded
    if st.session_state.schema is None:
        seed_path = "data/database.db"
        if os.path.exists(seed_path):
            if st.button("Load sample database"):
                with st.spinner("Loading…"):
                    schema = observe_database(seed_path)
                    st.session_state.schema  = schema
                    st.session_state.db_path = seed_path
                    st.rerun()
        else:
            st.info("Run `python data/seed_db.py` to create a sample DB")

    st.divider()

    # ── Document Upload ──
    st.subheader("📄 Unstructured Documents")
    doc_files = st.file_uploader(
        "Upload PDFs or Markdown files",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
    )

    if doc_files and st.button("🔄 Ingest documents"):
        with st.spinner("Chunking and embedding…"):
            tmp_paths = []
            try:
                for f in doc_files:
                    suffix = Path(f.name).suffix
                    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                    tmp.write(f.read())
                    tmp.flush()
                    tmp_paths.append(tmp.name)

                index = ingest_files(tmp_paths)
                save_index(index)
                st.session_state.faiss_index = index
                st.success(f"✅ {len(doc_files)} file(s) ingested")
            except Exception as e:
                st.error(f"Ingestion error: {e}")

    if st.session_state.faiss_index is None:
        sample_md = "data/sample_policy.md"
        if os.path.exists(sample_md):
            if st.button("Load sample document"):
                with st.spinner("Loading…"):
                    index = ingest_files([sample_md])
                    save_index(index)
                    st.session_state.faiss_index = index
                    st.rerun()

    st.divider()
    # ── Status indicators ──
    st.markdown("**System status**")
    st.markdown(
        f"{'✅' if st.session_state.schema      else '❌'} Database\n\n"
        f"{'✅' if st.session_state.faiss_index  else '❌'} Vector store"
    )
    if st.button("🗑️ Clear chat"):
        st.session_state.chat_history = []
        st.rerun()


# ── Main: Chat interface ──────────────────────────────────────────────────────
st.header("Ask a question")

# Render chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg["content"])
        else:
            result: PipelineResult = msg["result"]

            # Route badge + confidence
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(badge(result.route), unsafe_allow_html=True)
            with col2:
                st.markdown(
                    f"Confidence: {confidence_bar(result.relevance.confidence)}",
                    unsafe_allow_html=True,
                )

            # Low confidence warning
            if not result.relevance.is_sufficient:
                st.warning(
                    f"⚠️ Low confidence — {result.relevance.reasoning}"
                )

            # Answer
            st.markdown(result.answer.answer)

            # Citations
            if result.answer.citations:
                with st.expander("📎 Citations"):
                    for c in result.answer.citations:
                        st.code(c, language=None)

            # SQL details
            if result.sql_result and result.sql_result.success:
                with st.expander("🗄️ SQL query used"):
                    st.code(result.sql_result.sql, language="sql")
                    if result.sql_result.rows:
                        import pandas as pd
                        st.dataframe(pd.DataFrame(result.sql_result.rows))

            # Debug: router reasoning
            with st.expander("🔍 Router decision"):
                st.caption(f"Route: **{result.route.value}** — {result.route_reason}")
                st.caption(f"Self-RAG: {result.relevance.reasoning}")


# ── Input ────────────────────────────────────────────────────────────────────
if question := st.chat_input("Ask anything about your data…"):

    if st.session_state.schema is None and st.session_state.faiss_index is None:
        st.error("Please load at least one data source from the sidebar first.")
        st.stop()

    # User message
    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Routing → Retrieving → Checking → Generating…"):
            try:
                result = run_pipeline(
                    question=question,
                    schema=st.session_state.schema,
                    faiss_index=st.session_state.faiss_index,
                    db_path=st.session_state.db_path,
                )
                st.session_state.chat_history.append(
                    {"role": "assistant", "result": result, "content": result.answer.answer}
                )
                st.rerun()

            except Exception as e:
                st.error(f"Pipeline error: {e}")
                raise
