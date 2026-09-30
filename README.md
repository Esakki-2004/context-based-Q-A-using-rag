# Hybrid RAG — Context-Based QA System

A production-quality internship project implementing **Hybrid Retrieval-Augmented Generation**
with SQL + Vector DB, agentic routing, Self-RAG, and interactive citations.

---

## Architecture

```
User Question
     │
     ▼
Agentic Router  ──────────────────────────────────────────┐
  (SQL / Vector / Both)                                    │
     │                                                     │
     ├──► Text-to-SQL ──► SQLite (read-only) ──► SQLResult │
     │                                                      ├──► Self-RAG Check
     └──► FAISS Vector DB ──► Embedding Model ──► Chunks  │
                                                            │
                                                     ──► Generator ──► Answer + Citations
```

---

## Quick Start

### 1. Clone and install

```bash
git clone <repo>
cd hybrid_rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Seed the sample database

```bash
python data/seed_db.py
```

### 4. Run the app

```bash
streamlit run app.py
```

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Features

| Feature | Description |
|---|---|
| **Schema self-discovery** | Upload any `.db` file — the AI tells you what it contains |
| **Agentic router** | LLM-powered classification: SQL / Vector / Hybrid |
| **Text-to-SQL** | Natural language → validated SELECT query → results |
| **Self-RAG** | Relevance scoring before generation — flags low-confidence answers |
| **Interactive citations** | Every answer shows PDF page refs or the exact SQL used |
| **Read-only DB** | SQLite connection enforced with `?mode=ro` — no writes possible |

---

## Project Structure

```
hybrid_rag/
├── app.py                  # Streamlit dashboard
├── requirements.txt
├── .env.example
├── core/
│   ├── config.py           # Central config from .env
│   ├── db_observer.py      # Schema self-discovery
│   ├── vector_store.py     # FAISS ingestion + retrieval
│   ├── sql_tool.py         # Text-to-SQL pipeline
│   ├── router.py           # Agentic query router
│   ├── self_rag.py         # Context relevance check
│   ├── generator.py        # Answer generation + citations
│   └── pipeline.py         # Orchestration layer
├── data/
│   ├── seed_db.py          # Sample SQLite database
│   └── sample_policy.md    # Sample unstructured document
└── tests/
    └── test_pipeline.py    # Unit tests (mocked OpenAI)
```

---

## Sample Questions to Try

**SQL-only:**
- "Which department has the highest budget?"
- "How many employees are in Data Science?"
- "List all active projects and their budgets."

**Vector-only:**
- "What is the remote work policy?"
- "What does the code review standard say about PR size?"
- "When is a project charter required?"

**Hybrid (both sources):**
- "Which department had the highest budget, and what does the policy say about budget overruns?"
- "Who heads Engineering, and what is their hiring policy for interns?"

---

## Advanced Extensions

- **GraphRAG**: Add `networkx` + a graph construction step for multi-hop reasoning
- **Re-ranking**: Add a cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) after FAISS retrieval
- **Streaming**: Replace `client.chat.completions.create` with streaming + `st.write_stream`
- **PostgreSQL**: Swap SQLite for PostgreSQL by changing the connection string in `sql_tool.py`
- **Authentication**: Add `streamlit-authenticator` for multi-user deployments

---

## Safety Notes

- The SQLite connection uses `?mode=ro` (read-only URI) — the LLM **cannot** write, update, or delete data regardless of what SQL it generates.
- A regex blocklist (`_is_safe`) provides a second layer of protection.
- Never expose the app publicly without adding authentication.
