"""Búsqueda NO informada (léxica) con BM25.

No usa señal semántica: solo frecuencia de términos. Sirve como baseline y
como rama del retriever híbrido para capturar entidades exactas (nombres,
URLs, montos) que los embeddings pueden suavizar.
"""
from __future__ import annotations

import re
import unicodedata

from rank_bm25 import BM25Okapi

STOPWORDS_ES = {
    "a", "al", "ante", "bajo", "con", "contra", "de", "del", "desde", "en",
    "entre", "hacia", "hasta", "para", "por", "según", "sin", "sobre", "tras",
    "y", "o", "u", "e", "ni", "que", "qué", "como", "cómo", "cuando", "cuándo",
    "donde", "dónde", "es", "son", "ser", "fue", "fueron", "la", "el", "los",
    "las", "un", "una", "unos", "unas", "se", "su", "sus", "le", "les", "lo",
    "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas", "mi", "tu",
}


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def tokenize_es(text: str) -> list[str]:
    text = _strip_accents(text.lower())
    tokens = re.findall(r"[a-z0-9]+", text)
    return [t for t in tokens if t not in STOPWORDS_ES and len(t) > 1]


class BM25Retriever:
    def __init__(self, texts: list[str]):
        self.tokenized = [tokenize_es(t) for t in texts]
        self.bm25 = BM25Okapi(self.tokenized)

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        q_tokens = tokenize_es(query)
        if not q_tokens:
            return []
        scores = self.bm25.get_scores(q_tokens)
        idx = scores.argsort()[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]
