from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# --- Chunking ---
CHUNK_SIZE = 600      # caracteres por fragmento
CHUNK_OVERLAP = 100   # solapamiento entre fragmentos contiguos

# --- Embeddings (modelo local, no requiere API) ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 384 dims, ~90 MB, multilingüe

# --- ChromaDB ---
COLLECTION_NAME = "agente_rag_docs"

# --- Recuperación ---
TOP_K = 5   # fragmentos devueltos por consulta por defecto
