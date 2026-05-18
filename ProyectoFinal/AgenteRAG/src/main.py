"""CLI principal del Agente RAG.

Comandos:
    python -m src.main health          # verifica modelos Ollama
    python -m src.main ingest          # parsea PDFs y construye índice
    python -m src.main ask "..."       # responde una pregunta
    python -m src.main demo            # corre las 6 preguntas obligatorias
    python -m src.main eval            # evalúa contra testset.json
    python -m src.main repl            # modo interactivo
"""
from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "index"
console = Console()


DEMO_QUESTIONS = [
    "¿Cuáles son los archivos que tienes disponibles para proporcionar información?",
    "¿Cuál es el orden cronológico de los documentos?",
    "¿Cómo se llama la empresa y cuál es el perjuicio ocasionado?",
    "¿Qué tipo de problema fue y cuál fue la causa del incidente?",
    "¿Quiénes son los involucrados? ¿Se conoce el nombre de quien reporta el incidente?",
    "¿Qué archivo describe las sanciones ocasionadas?",
]


def _agent(judge: bool = False):
    from src.agent.graph import RAGAgent
    from src.retriever.store import load_index

    store = load_index(INDEX_DIR)
    return RAGAgent(store, judge=judge)


def cmd_health() -> None:
    from src.utils.ollama_client import health_check

    info = health_check()
    table = Table(title="Ollama health check")
    table.add_column("requisito")
    table.add_column("estado")
    table.add_row("LLM " + info["required_llm"], "✅" if info["llm_ok"] else "❌")
    table.add_row("Embed " + info["required_embed"], "✅" if info["embed_ok"] else "❌")
    console.print(table)
    console.print(f"Modelos disponibles: {info['available']}")


def cmd_ingest() -> None:
    from src.ingest.build_index import run

    run()


def cmd_ask(question: str) -> None:
    agent = _agent(judge=True)
    res = agent.ask(question)
    console.print(Panel.fit(question, title="Pregunta", border_style="cyan"))
    console.print(Panel(res.answer, title=f"Respuesta · intent={res.intent}",
                        border_style="green"))
    if res.citations:
        console.print(f"[dim]Fuentes:[/dim] {', '.join(res.citations)}")
    if res.faithful is not None:
        color = "green" if res.faithful else "red"
        console.print(f"[{color}]faithful={res.faithful}[/{color}] · {res.judge_reason}")


def cmd_demo() -> None:
    agent = _agent(judge=False)
    for q in DEMO_QUESTIONS:
        res = agent.ask(q)
        console.print(Panel.fit(q, title="Pregunta", border_style="cyan"))
        console.print(Panel(res.answer, title=f"intent={res.intent}",
                            border_style="green"))
        if res.citations:
            console.print(f"[dim]Fuentes:[/dim] {', '.join(res.citations)}\n")


def cmd_eval() -> None:
    from src.eval.evaluate import run as run_eval
    run_eval()


def cmd_repl() -> None:
    agent = _agent(judge=False)
    console.print("[bold]Agente RAG[/bold]  (Ctrl+C para salir)")
    while True:
        try:
            q = console.input("\n[bold cyan]?> [/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            console.print("\nadiós")
            return
        if not q.strip():
            continue
        res = agent.ask(q)
        console.print(Panel(res.answer, title=f"intent={res.intent}",
                            border_style="green"))
        if res.citations:
            console.print(f"[dim]Fuentes:[/dim] {', '.join(res.citations)}")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        console.print(__doc__)
        return 1
    cmd = argv[1]
    if cmd == "health":
        cmd_health()
    elif cmd == "ingest":
        cmd_ingest()
    elif cmd == "ask":
        if len(argv) < 3:
            console.print("Uso: ask \"<pregunta>\"")
            return 1
        cmd_ask(" ".join(argv[2:]))
    elif cmd == "demo":
        cmd_demo()
    elif cmd == "eval":
        cmd_eval()
    elif cmd == "repl":
        cmd_repl()
    else:
        console.print(f"Comando desconocido: {cmd}")
        console.print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
