"""Evaluación del agente RAG con métricas explícitas.

Métricas implementadas:
1. Retrieval:
   - Hit@K            : ¿el archivo correcto está entre los K recuperados?
   - File-Precision@K : proporción de fuentes recuperadas que son correctas
2. Generación:
   - KeywordRecall    : fracción de palabras clave esperadas presentes en la respuesta
3. Fidelidad (faithfulness):
   - LLM-as-judge     : el propio LLM verifica que la respuesta esté en el contexto
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.agent.graph import RAGAgent
from src.retriever.store import load_index

console = Console()
ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = ROOT / "data" / "index"
TESTSET = Path(__file__).parent / "testset.json"


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def keyword_recall(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    a = _norm(answer)
    hits = sum(1 for k in keywords if _norm(k) in a)
    return hits / len(keywords)


def hit_at_k(retrieved_files: list[str], expected_files: list[str]) -> int:
    return int(any(f in retrieved_files for f in expected_files))


def file_precision(retrieved_files: list[str], expected_files: list[str]) -> float:
    if not retrieved_files:
        return 0.0
    correct = sum(1 for f in retrieved_files if f in expected_files)
    return correct / len(retrieved_files)


def run() -> None:
    testset = json.loads(TESTSET.read_text(encoding="utf-8"))
    store = load_index(INDEX_DIR)
    agent = RAGAgent(store, top_k=4, judge=True)

    table = Table(title="Evaluación RAG", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("pregunta", width=40)
    table.add_column("Hit@K", justify="center")
    table.add_column("File-Prec", justify="center")
    table.add_column("KW-Rec", justify="center")
    table.add_column("Faithful", justify="center")

    agg = {"hit": 0, "prec": 0.0, "kw": 0.0, "faithful": 0, "n": 0}
    detailed = []

    for i, item in enumerate(testset, start=1):
        q = item["question"]
        expected_kw = item["expected_keywords"]
        expected_files = item["expected_files"]

        res = agent.ask(q)
        retrieved_files = (
            list({c["file_name"] for c in res.contexts}) if res.contexts else res.citations
        )

        h = hit_at_k(retrieved_files, expected_files)
        p = file_precision(retrieved_files, expected_files)
        kw = keyword_recall(res.answer, expected_kw)
        faithful = bool(res.faithful) if res.faithful is not None else True

        agg["hit"] += h
        agg["prec"] += p
        agg["kw"] += kw
        agg["faithful"] += int(faithful)
        agg["n"] += 1

        table.add_row(
            str(i),
            q[:38] + ("…" if len(q) > 38 else ""),
            "✅" if h else "❌",
            f"{p:.2f}",
            f"{kw:.2f}",
            "✅" if faithful else "❌",
        )
        detailed.append({
            "question": q,
            "answer": res.answer,
            "intent": res.intent,
            "retrieved_files": retrieved_files,
            "expected_files": expected_files,
            "hit_at_k": h,
            "file_precision": p,
            "keyword_recall": kw,
            "faithful": faithful,
            "judge_reason": res.judge_reason,
        })

    console.print(table)

    n = agg["n"]
    console.rule("[bold]Promedios")
    summary = Table()
    summary.add_column("métrica")
    summary.add_column("valor", justify="right")
    summary.add_row("Hit@K",            f"{agg['hit']/n:.3f}")
    summary.add_row("File-Precision@K", f"{agg['prec']/n:.3f}")
    summary.add_row("Keyword-Recall",   f"{agg['kw']/n:.3f}")
    summary.add_row("Faithfulness",     f"{agg['faithful']/n:.3f}")
    console.print(summary)

    out = ROOT / "data" / "eval_results.json"
    out.write_text(json.dumps({
        "summary": {
            "hit_at_k": agg["hit"] / n,
            "file_precision": agg["prec"] / n,
            "keyword_recall": agg["kw"] / n,
            "faithfulness": agg["faithful"] / n,
        },
        "details": detailed,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"\nDetalles guardados en [dim]{out}[/dim]")
