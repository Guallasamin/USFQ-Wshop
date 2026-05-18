"""Meta-herramientas que resuelven preguntas estructurales sin invocar al LLM."""
from __future__ import annotations

from src.retriever.store import IndexStore


def list_files(store: IndexStore) -> list[str]:
    """Devuelve la lista única de archivos indexados."""
    return sorted({c["file_name"] for c in store.chunks})


def chronology(store: IndexStore) -> list[dict]:
    """Orden cronológico de los documentos por fecha detectada."""
    seen: dict[str, dict] = {}
    for c in store.chunks:
        fname = c["file_name"]
        if fname not in seen:
            seen[fname] = {
                "file_name": fname,
                "doc_type": c["doc_type"],
                "detected_date": c["detected_date"],
            }
    docs = list(seen.values())
    docs.sort(key=lambda d: (d["detected_date"] is None, d["detected_date"] or ""))
    return docs
