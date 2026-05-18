"""Carga del índice persistido (chunks + matriz de embeddings)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class IndexStore:
    chunks: list[dict]
    embeddings: np.ndarray  # ya L2-normalizados

    @property
    def n(self) -> int:
        return len(self.chunks)

    def texts(self) -> list[str]:
        return [c["text"] for c in self.chunks]


def load_index(index_dir: Path) -> IndexStore:
    metadata_path = index_dir / "metadata.json"
    emb_path = index_dir / "embeddings.npy"
    if not metadata_path.exists() or not emb_path.exists():
        raise FileNotFoundError(
            f"No se encontró el índice en {index_dir}. "
            "Ejecuta primero: python -m src.ingest.build_index"
        )
    chunks = json.loads(metadata_path.read_text(encoding="utf-8"))
    embeddings = np.load(emb_path)
    return IndexStore(chunks=chunks, embeddings=embeddings)
