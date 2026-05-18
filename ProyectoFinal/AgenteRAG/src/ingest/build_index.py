"""Pipeline de ingesta: PDF -> chunks -> embeddings -> índice persistido.

Salidas:
  data/processed/chunks.jsonl     (texto + metadatos)
  data/index/embeddings.npy       (matriz NxD float32)
  data/index/metadata.json        (mapeo de fila -> chunk_id)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import track

from src.ingest.chunker import Chunk, chunk_document
from src.ingest.parser import parse_directory
from src.utils.ollama_client import embed_batch

console = Console()
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
INDEX_DIR = ROOT / "data" / "index"


def run() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    console.rule("[bold cyan]1. Parseo de PDFs")
    parsed_docs = list(parse_directory(RAW_DIR))
    console.print(f"  Documentos parseados: [bold]{len(parsed_docs)}[/bold]")
    for d in parsed_docs:
        console.print(
            f"   - {d.file_name}  type=[yellow]{d.doc_type}[/yellow]  "
            f"date=[green]{d.detected_date}[/green]  pages={d.n_pages}"
        )

    console.rule("[bold cyan]2. Chunking")
    all_chunks: list[Chunk] = []
    for d in parsed_docs:
        ck = chunk_document(d.to_dict())
        console.print(f"   - {d.file_name}: {len(ck)} chunks")
        all_chunks.extend(ck)
    console.print(f"  Total chunks: [bold]{len(all_chunks)}[/bold]")

    chunks_path = PROCESSED_DIR / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    console.print(f"  Persistido en: [dim]{chunks_path}[/dim]")

    console.rule("[bold cyan]3. Embeddings (bge-m3 via Ollama)")
    texts = [c.text for c in all_chunks]
    vectors: list[list[float]] = []
    batch = 8
    for i in track(range(0, len(texts), batch), description="Embedding..."):
        vectors.extend(embed_batch(texts[i : i + batch]))
    matrix = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    np.save(INDEX_DIR / "embeddings.npy", matrix)
    metadata = [c.to_dict() for c in all_chunks]
    (INDEX_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    console.print(
        f"  Matriz: [bold]{matrix.shape}[/bold] dtype={matrix.dtype} "
        f"-> [dim]{INDEX_DIR/'embeddings.npy'}[/dim]"
    )
    console.print(f"  Metadatos -> [dim]{INDEX_DIR/'metadata.json'}[/dim]")
    console.rule("[bold green]Ingesta completada")


if __name__ == "__main__":
    run()
