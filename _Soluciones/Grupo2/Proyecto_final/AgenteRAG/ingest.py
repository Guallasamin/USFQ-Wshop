"""
Miembro 1 - Ingesta de documentos
Pipeline: PDF → chunks → embeddings → ChromaDB local

Uso:
    python ingest.py            # ingesta normal
    python ingest.py --force    # borra y re-indexa aunque ya exista el vectorstore
"""
import argparse
import shutil
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import (
    DATA_DIR, VECTORSTORE_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_MODEL, COLLECTION_NAME,
)


def load_pdfs() -> list:
    """Carga todos los PDFs de data/ como páginas individuales con metadatos."""
    pdf_paths = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No se encontraron PDFs en {DATA_DIR}")

    all_docs = []
    for path in pdf_paths:
        print(f"  Leyendo: {path.name}")
        loader = PyPDFLoader(str(path))
        pages = loader.load()
        # Normalizar metadato 'source' al nombre del archivo solamente
        for page in pages:
            page.metadata["source"] = path.name
        all_docs.extend(pages)

    print(f"  -> {len(all_docs)} paginas cargadas desde {len(pdf_paths)} archivos\n")
    return all_docs


def chunk_documents(docs: list) -> list:
    """
    Divide los documentos en fragmentos usando separadores jerárquicos.
    Prioriza cortes en párrafos, luego líneas, luego oraciones.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)

    print(f"  Parámetros: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    print(f"  -> {len(chunks)} fragmentos generados\n")
    return chunks


def build_vectorstore(chunks: list, force: bool = False) -> Chroma:
    """
    Genera embeddings locales y persiste la base vectorial en disco.
    Si force=True, elimina cualquier vectorstore previo antes de indexar.
    """
    if VECTORSTORE_DIR.exists():
        if force:
            print(f"  Eliminando vectorstore previo en {VECTORSTORE_DIR}")
            shutil.rmtree(VECTORSTORE_DIR)
        else:
            print(
                f"  Vectorstore ya existe en {VECTORSTORE_DIR}.\n"
                "  Usa --force para re-indexar desde cero."
            )
            return _load_existing(chunks)

    print(f"  Cargando modelo de embeddings: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  Indexando {len(chunks)} fragmentos (puede tardar 1-2 min)...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTORSTORE_DIR),
    )
    total = vectorstore._collection.count()
    print(f"  -> {total} vectores almacenados en {VECTORSTORE_DIR}\n")
    return vectorstore


def _load_existing(chunks: list) -> Chroma:
    """Carga el vectorstore existente sin re-indexar."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )
    total = vs._collection.count()
    print(f"  -> Vectorstore cargado: {total} vectores existentes\n")
    return vs


def print_summary(chunks: list) -> None:
    """Imprime estadísticas de la ingesta para el informe."""
    from collections import Counter
    sources = Counter(c.metadata.get("source", "?") for c in chunks)
    print("=== Resumen de ingesta ===")
    print(f"{'Archivo':<35} {'Chunks':>7}")
    print("-" * 44)
    for src, count in sorted(sources.items()):
        print(f"{src:<35} {count:>7}")
    print("-" * 44)
    print(f"{'TOTAL':<35} {sum(sources.values()):>7}")
    print()


def main(force: bool = False) -> None:
    print("\n=== Miembro 1: Ingesta de documentos ===\n")

    print("[1/3] Cargando PDFs...")
    docs = load_pdfs()

    print("[2/3] Fragmentando textos...")
    chunks = chunk_documents(docs)

    print("[3/3] Construyendo base vectorial...")
    build_vectorstore(chunks, force=force)

    print_summary(chunks)
    print("Ingesta completada. El vectorstore está listo para el Miembro 2.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingesta de documentos para AgenteRAG")
    parser.add_argument("--force", action="store_true", help="Re-indexa aunque ya exista el vectorstore")
    args = parser.parse_args()
    main(force=args.force)
