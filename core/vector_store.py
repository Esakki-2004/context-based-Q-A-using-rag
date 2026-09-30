"""
Vector store — FAISS-backed retrieval for unstructured text (PDFs, Markdown).

Handles:
  - Ingesting files into chunked embeddings
  - Persisting / loading the FAISS index
  - Returning retrieved chunks WITH citation metadata (source file + page)
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from core.config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RETRIEVAL,
)


@dataclass
class VectorResult:
    content: str
    source: str          # filename
    page: Optional[int]  # page number if PDF, else None
    score: float         # cosine similarity (0–1, higher = more relevant)


def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )


def ingest_files(file_paths: list[str]) -> FAISS:
    """Load, chunk, embed, and return a FAISS index from the given files."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    all_docs = []

    for fp in file_paths:
        ext = Path(fp).suffix.lower()
        if ext == ".pdf":
            loader = PyPDFLoader(fp)
        elif ext in (".md", ".txt"):
            loader = TextLoader(fp, encoding="utf-8")
        else:
            continue  # skip unsupported types

        docs = loader.load_and_split(text_splitter=splitter)
        # Normalise metadata so every doc has 'source' and 'page'
        for doc in docs:
            doc.metadata.setdefault("source", Path(fp).name)
            doc.metadata.setdefault("page", None)
        all_docs.extend(docs)

    if not all_docs:
        raise ValueError("No supported documents found in the provided paths.")

    embeddings = _get_embeddings()
    index = FAISS.from_documents(all_docs, embeddings)
    return index


def save_index(index: FAISS) -> None:
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    index.save_local(FAISS_INDEX_PATH)


def load_index() -> Optional[FAISS]:
    if not os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")):
        return None
    return FAISS.load_local(
        FAISS_INDEX_PATH,
        _get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def retrieve(query: str, index: FAISS) -> list[VectorResult]:
    """Return top-K chunks with citation metadata."""
    results = index.similarity_search_with_relevance_scores(query, k=TOP_K_RETRIEVAL)
    return [
        VectorResult(
            content=doc.page_content,
            source=doc.metadata.get("source", "unknown"),
            page=doc.metadata.get("page"),
            score=score,
        )
        for doc, score in results
    ]
