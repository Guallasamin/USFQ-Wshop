"""Agente RAG implementado como máquina de estados explícita.

Flujo (cada nodo es una función pura):
    classify_intent  →  ┬─ list_files     ─┐
                        ├─ chronology     ─┤
                        └─ rag_pipeline ──┘─→  response

`rag_pipeline` ejecuta: retrieve_hybrid → build_prompt → generate → optional judge
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.agent.prompts import (
    SYSTEM_INTENT,
    SYSTEM_JUDGE,
    SYSTEM_QA,
    USER_QA_TEMPLATE,
)
from src.agent.tools import chronology, list_files
from src.retriever.bm25 import BM25Retriever
from src.retriever.hybrid import HybridRetriever
from src.retriever.store import IndexStore
from src.retriever.vector import VectorRetriever
from src.utils.ollama_client import chat

VALID_INTENTS = {"LIST_FILES", "CHRONOLOGY", "CONTENT_QA"}


@dataclass
class AgentResult:
    answer: str
    intent: str
    contexts: list[dict] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    faithful: bool | None = None
    judge_reason: str | None = None


class RAGAgent:
    def __init__(self, store: IndexStore, top_k: int = 4, judge: bool = False):
        self.store = store
        self.top_k = top_k
        self.judge_enabled = judge
        bm25 = BM25Retriever(store.texts())
        vector = VectorRetriever(store.embeddings)
        self.retriever = HybridRetriever(bm25, vector)

    # ---------- nodos ----------

    def classify_intent(self, question: str) -> str:
        raw = chat(SYSTEM_INTENT, question, temperature=0.0).upper()
        for tag in VALID_INTENTS:
            if tag in raw:
                return tag
        return "CONTENT_QA"

    def retrieve(self, question: str) -> list[tuple[int, float, dict]]:
        return self.retriever.search(question, top_k=self.top_k, candidates=12)

    def build_context(self, hits: list[tuple[int, float, dict]]) -> tuple[str, list[dict]]:
        blocks = []
        contexts = []
        for rank, (idx, score, info) in enumerate(hits, start=1):
            chunk = self.store.chunks[idx]
            blocks.append(
                f"[FRAGMENTO {rank}] (archivo: {chunk['file_name']}, "
                f"pág {chunk['page']}, fecha: {chunk['detected_date']})\n"
                f"{chunk['text']}"
            )
            contexts.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "file_name": chunk["file_name"],
                    "page": chunk["page"],
                    "rrf": score,
                    "bm25_rank": info["bm25_rank"],
                    "vec_rank": info["vec_rank"],
                    "text": chunk["text"],
                }
            )
        return "\n\n".join(blocks), contexts

    def generate(self, question: str, context: str) -> str:
        prompt = USER_QA_TEMPLATE.format(context=context, question=question)
        return chat(SYSTEM_QA, prompt, temperature=0.1)

    def judge(self, context: str, answer: str) -> tuple[bool, str]:
        import json
        prompt = f"CONTEXTO:\n{context}\n\nRESPUESTA:\n{answer}"
        raw = chat(SYSTEM_JUDGE, prompt, temperature=0.0)
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            data = json.loads(raw[start:end])
            return bool(data.get("faithful", False)), str(data.get("reason", ""))
        except Exception:
            return False, f"unparseable: {raw[:80]}"

    # ---------- orquestación ----------

    def ask(self, question: str) -> AgentResult:
        intent = self.classify_intent(question)

        if intent == "LIST_FILES":
            files = list_files(self.store)
            answer = "Archivos disponibles:\n- " + "\n- ".join(files)
            return AgentResult(answer=answer, intent=intent, citations=files)

        if intent == "CHRONOLOGY":
            docs = chronology(self.store)
            lines = [
                f"{d['detected_date'] or 'sin fecha'}  →  {d['file_name']} ({d['doc_type']})"
                for d in docs
            ]
            return AgentResult(
                answer="Orden cronológico de los documentos:\n" + "\n".join(lines),
                intent=intent,
                citations=[d["file_name"] for d in docs],
            )

        # CONTENT_QA
        hits = self.retrieve(question)
        context, contexts = self.build_context(hits)
        answer = self.generate(question, context)
        citations = sorted({c["file_name"] for c in contexts})
        faithful = None
        reason = None
        if self.judge_enabled:
            faithful, reason = self.judge(context, answer)
        return AgentResult(
            answer=answer,
            intent=intent,
            contexts=contexts,
            citations=citations,
            faithful=faithful,
            judge_reason=reason,
        )
