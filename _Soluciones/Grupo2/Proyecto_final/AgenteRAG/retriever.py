"""
Miembro 1 → interfaz para Miembro 2
Carga el vectorstore persistido y expone dos funciones:
  - search(query, k)      → lista de dicts {content, source, page, score}
  - get_langchain_retriever(k) → LangChain Retriever para LCEL / LangGraph

Prerequisito: haber ejecutado `python ingest.py` al menos una vez.
"""
from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import VECTORSTORE_DIR, EMBEDDING_MODEL, COLLECTION_NAME, TOP_K

# Singleton: la base vectorial se carga una sola vez por sesión
_vectorstore: Chroma | None = None


def _get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        if not VECTORSTORE_DIR.exists():
            raise RuntimeError(
                "Vectorstore no encontrado. Ejecuta primero: python ingest.py"
            )
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(VECTORSTORE_DIR),
        )
    return _vectorstore


def search(query: str, k: int = TOP_K) -> list[dict]:
    """
    Recupera los k fragmentos más relevantes para una consulta.

    Args:
        query: Pregunta o texto de búsqueda.
        k:     Número de fragmentos a devolver (default: TOP_K de config.py).

    Returns:
        Lista de dicts con:
          - content : texto del fragmento
          - source  : nombre del archivo PDF de origen
          - page    : número de página (base 0)
          - score   : similitud coseno normalizada en [0, 1]
    """
    vs = _get_vectorstore()
    results = vs.similarity_search_with_relevance_scores(query, k=k)

    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "desconocido"),
            "page": doc.metadata.get("page", 0) + 1,  # base 1 para lectura humana
            "score": round(score, 4),
        }
        for doc, score in results
    ]


def get_langchain_retriever(k: int = TOP_K):
    """
    Retorna un LangChain BaseRetriever listo para usar con LCEL o LangGraph.

    Ejemplo de uso en el agente (Miembro 2):
        from retriever import get_langchain_retriever
        retriever = get_langchain_retriever(k=5)
        docs = retriever.invoke("¿Cuáles son los incidentes registrados?")
    """
    return _get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


if __name__ == "__main__":
    print("=== Test de recuperación ===\n")
    test_queries = [
        "¿Qué incidentes se reportaron?",
        "¿Qué empresa fue perjudicada?",
        "¿Quiénes estuvieron involucrados?",
    ]
    for q in test_queries:
        print(f"Consulta: {q}")
        resultados = search(q, k=3)
        for i, r in enumerate(resultados, 1):
            print(f"  [{i}] {r['source']} p.{r['page']} | score={r['score']}")
            print(f"       {r['content'][:150].strip()}...")
        print()
