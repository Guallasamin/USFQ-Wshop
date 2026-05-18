"""Búsqueda INFORMADA (semántica) por similitud coseno sobre embeddings densos.

Es "informada" porque cada vector codifica el significado del texto aprendido
por el modelo de embeddings (bge-m3), no solo presencia de palabras.
"""
from __future__ import annotations

import numpy as np

from src.utils.ollama_client import embed_one


class VectorRetriever:
    def __init__(self, embeddings: np.ndarray):
        # se asume que `embeddings` ya está L2-normalizado al construir el índice
        self.matrix = embeddings

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        q = np.array(embed_one(query), dtype=np.float32)
        n = np.linalg.norm(q)
        if n == 0:
            return []
        q = q / n
        scores = self.matrix @ q  # coseno (ambos L2-normalizados)
        idx = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in idx]
