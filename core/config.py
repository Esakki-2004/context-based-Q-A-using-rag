from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL      = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL           = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
FAISS_INDEX_PATH     = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
SQLITE_DB_PATH       = os.getenv("SQLITE_DB_PATH", "data/database.db")

CHUNK_SIZE           = 512
CHUNK_OVERLAP        = 64
TOP_K_RETRIEVAL      = 5
