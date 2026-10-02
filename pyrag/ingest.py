"""Ingestao: PDF, Markdown, TXT, HTML, CSV/JSON, paginas web -> Documentos chunkados."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import get_settings

log = logging.getLogger(__name__)

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".py", ".csv", ".json", ".log", ".yaml", ".yml"}


def _load_single(path: Path) -> list[Document]:
    suf = path.suffix.lower()
    try:
        if suf == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader

            return list(PyPDFLoader(str(path)).load())
        if suf in (".html", ".htm"):
            from langchain_community.document_loaders import BSHTMLLoader

            return list(BSHTMLLoader(str(path)).load())
        if suf == ".json":
            return list(Document(
                page_content=json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2)
            ))
        if suf == ".csv":
            import csv

            with path.open(encoding="utf-8", newline="") as f:
                rows = list(csv.reader(f))
            return [Document(page_content="\n".join(" | ".join(r) for r in rows))]
        if suf in TEXT_SUFFIXES:
            return [Document(page_content=path.read_text(encoding="utf-8", errors="ignore"))]
    except Exception as exc:
        log.warning("Falha ao ler %s: %s", path, exc)
        return []
    return []


def load_documents(
    paths: Iterable[str | Path],
    collection: str = "default",
) -> list[Document]:
    """Carrega arquivos locais e URLs, ja anotando metadados (path/title/collection)."""
    docs: list[Document] = []

    for raw in paths:
        p = str(raw)
        if p.startswith(("http://", "https://")):
            try:
                from langchain_community.document_loaders import WebBaseLoader

                loaded = list(WebBaseLoader(p).load())
            except Exception as exc:
                log.warning("Falha ao baixar %s: %s", p, exc)
                continue
            for d in loaded:
                d.metadata.setdefault("path", p)
            docs.extend(loaded)
            continue

        path = Path(p).expanduser().resolve()
        if not path.exists():
            log.warning("Caminho inexistente ignorado: %s", path)
            continue
        files = [path] if path.is_file() else [f for f in sorted(path.rglob("*")) if f.suffix.lower() in TEXT_SUFFIXES | {".pdf", ".html", ".htm"}]
        for f in files:
            for d in _load_single(f):
                d.metadata.update({
                    "path": str(f),
                    "title": d.metadata.get("title") or f.stem,
                    "source": str(f),
                    "collection": collection,
                })
                docs.append(d)

    log.info("Carregados %d documentos brutos de %d caminho(s)", len(docs), len(list(paths)))
    return docs


def split_documents(
    docs: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    s = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or s.chunk_size,
        chunk_overlap=chunk_overlap or s.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    out = splitter.split_documents(docs)
    for i, d in enumerate(out):
        d.metadata["chunk"] = i
        d.metadata.setdefault("title", Path(d.metadata.get("path", "doc")).stem)
    log.info("Chunking: %d docs -> %d chunks", len(docs), len(out))
    return out


def ingest(
    paths: Iterable[str | Path],
    collection: str = "default",
) -> list[Document]:
    """Pipeline completo: load -> split. A persistencia fica em vectorstore.upsert."""
    return split_documents(load_documents(paths, collection=collection), )