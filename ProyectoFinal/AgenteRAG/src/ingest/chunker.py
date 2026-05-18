"""Chunking semántico por párrafos con ventana deslizante de tokens (aprox. palabras)."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, asdict


@dataclass
class Chunk:
    chunk_id: str
    file_name: str
    doc_type: str
    detected_date: str | None
    page: int
    chunk_index: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _word_chunks(text: str, target_words: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= target_words:
        return [text]
    chunks = []
    step = target_words - overlap
    for start in range(0, len(words), step):
        piece = words[start : start + target_words]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if start + target_words >= len(words):
            break
    return chunks


def chunk_document(
    parsed_doc: dict,
    target_words: int = 180,
    overlap: int = 40,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0
    for page in parsed_doc["pages"]:
        page_num = page["page"]
        paragraphs = _split_paragraphs(page["text"])
        joined = "\n\n".join(paragraphs) if paragraphs else page["text"]
        for piece in _word_chunks(joined, target_words, overlap):
            piece = piece.strip()
            if len(piece) < 20:
                continue
            digest = hashlib.md5(
                f"{parsed_doc['file_name']}|{page_num}|{idx}|{piece[:50]}".encode()
            ).hexdigest()[:12]
            chunks.append(
                Chunk(
                    chunk_id=digest,
                    file_name=parsed_doc["file_name"],
                    doc_type=parsed_doc["doc_type"],
                    detected_date=parsed_doc["detected_date"],
                    page=page_num,
                    chunk_index=idx,
                    text=piece,
                )
            )
            idx += 1
    return chunks
