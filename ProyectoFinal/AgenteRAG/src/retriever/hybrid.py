"""Fusión informada + no informada con Reciprocal Rank Fusion (RRF).

RRF(d) = sum_r  1 / (k + rank_r(d))

Ventaja sobre suma de scores normalizados: no requiere calibrar escalas
distintas (BM25 vs coseno). Es la técnica estándar en sistemas híbridos
(Elasticsearch hybrid search, Vespa, Weaviate).
"""
from __future__ import annotations

from .bm25 import BM25Retriever
from .vector import VectorRetriever


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Retriever,
        vector: VectorRetriever,
        k_rrf: int = 60,
    ):
        self.bm25 = bm25
        self.vector = vector
        self.k_rrf = k_rrf

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidates: int = 15,
    ) -> list[tuple[int, float, dict]]:
        bm25_hits = self.bm25.search(query, top_k=candidates)
        vec_hits = self.vector.search(query, top_k=candidates)

        ranks: dict[int, dict] = {}
        for rank, (idx, score) in enumerate(bm25_hits):
            ranks.setdefault(idx, {"bm25_rank": None, "vec_rank": None,
                                   "bm25_score": 0.0, "vec_score": 0.0})
            ranks[idx]["bm25_rank"] = rank
            ranks[idx]["bm25_score"] = score
        for rank, (idx, score) in enumerate(vec_hits):
            ranks.setdefault(idx, {"bm25_rank": None, "vec_rank": None,
                                   "bm25_score": 0.0, "vec_score": 0.0})
            ranks[idx]["vec_rank"] = rank
            ranks[idx]["vec_score"] = score

        fused = []
        for idx, info in ranks.items():
            rrf = 0.0
            if info["bm25_rank"] is not None:
                rrf += 1.0 / (self.k_rrf + info["bm25_rank"] + 1)
            if info["vec_rank"] is not None:
                rrf += 1.0 / (self.k_rrf + info["vec_rank"] + 1)
            fused.append((idx, rrf, info))

        fused.sort(key=lambda x: -x[1])
        return fused[:top_k]
