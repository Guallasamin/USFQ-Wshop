"""PDF parser con extracción de metadatos (fecha, tipo de documento)."""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

from pypdf import PdfReader

MESES_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9,
    "octubre": 10, "noviembre": 11, "diciembre": 12,
}

DATE_PATTERNS = [
    re.compile(r"(\d{1,2})\s+de\s+([a-záéíóú]+)\s+de\s+(\d{4})", re.IGNORECASE),
    re.compile(r"(\d{4})-(\d{2})-(\d{2})"),
    re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})"),
]

DOC_TYPE_BY_NAME = {
    "AccionDePersonal": "accion_personal",
    "DiarioDeGerente": "diario_gerente",
    "EmailPersonal": "email",
    "PlanMejoras": "plan_mejoras",
    "ReporteIncidentes": "reporte_incidente",
}


@dataclass
class ParsedDocument:
    file_name: str
    doc_type: str
    detected_date: str | None
    n_pages: int
    pages: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize_date(match: re.Match) -> str | None:
    """Convierte un match de fecha en ISO `YYYY-MM-DD`."""
    groups = match.groups()
    try:
        if len(groups) == 3 and groups[1].lower() in MESES_ES:
            d, mes, y = int(groups[0]), MESES_ES[groups[1].lower()], int(groups[2])
        elif len(groups) == 3 and groups[0].isdigit() and len(groups[0]) == 4:
            y, mes, d = int(groups[0]), int(groups[1]), int(groups[2])
        elif len(groups) == 3:
            d, mes, y = int(groups[0]), int(groups[1]), int(groups[2])
        else:
            return None
        return f"{y:04d}-{mes:02d}-{d:02d}"
    except (ValueError, KeyError):
        return None


def extract_date(text: str) -> str | None:
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            normalized = _normalize_date(match)
            if normalized:
                return normalized
    return None


def _doc_type_from_name(stem: str) -> str:
    for key, val in DOC_TYPE_BY_NAME.items():
        if key.lower() in stem.lower():
            return val
    return "desconocido"


def parse_pdf(path: Path) -> ParsedDocument:
    reader = PdfReader(str(path))
    pages = []
    full_text = ""
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = re.sub(r"[ \t]+", " ", text).strip()
        pages.append({"page": i + 1, "text": text})
        full_text += "\n" + text

    return ParsedDocument(
        file_name=path.name,
        doc_type=_doc_type_from_name(path.stem),
        detected_date=extract_date(full_text),
        n_pages=len(pages),
        pages=pages,
    )


def parse_directory(raw_dir: Path) -> Iterator[ParsedDocument]:
    for pdf_path in sorted(raw_dir.glob("*.pdf")):
        yield parse_pdf(pdf_path)
