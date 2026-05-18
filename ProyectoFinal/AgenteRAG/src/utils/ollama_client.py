"""Cliente unificado para Ollama: chat (LLM) y embeddings."""
from __future__ import annotations

import os
from typing import Iterable

import ollama

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "qwen2.5:7b-instruct")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "bge-m3")

_client = ollama.Client(host=OLLAMA_HOST)


def chat(system: str, user: str, temperature: float = 0.1) -> str:
    """Una sola llamada de chat al LLM local."""
    response = _client.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": temperature},
    )
    return response["message"]["content"].strip()


def embed_batch(texts: Iterable[str]) -> list[list[float]]:
    """Embeddings para una lista de textos."""
    response = _client.embed(model=EMBED_MODEL, input=list(texts))
    return response["embeddings"]


def embed_one(text: str) -> list[float]:
    return embed_batch([text])[0]


def health_check() -> dict:
    """Verifica que los modelos requeridos estén disponibles."""
    available = {m["model"] for m in _client.list().get("models", [])}
    return {
        "llm_ok": any(LLM_MODEL.split(":")[0] in m for m in available),
        "embed_ok": any(EMBED_MODEL.split(":")[0] in m for m in available),
        "available": sorted(available),
        "required_llm": LLM_MODEL,
        "required_embed": EMBED_MODEL,
    }
