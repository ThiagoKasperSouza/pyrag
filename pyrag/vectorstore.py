"""Vectorstore hibrido: Chroma (denso) + BM25 (lexical) com reranking.

O indice BM25 e persistido em JSON para nao depender de build nativo (rank_bm25).
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document

from .config import get_settings
from .llm import cached_embeddings

log = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


class BM25Index:
    """BM25Okapi minimo e persistente (dados em JSON)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs: list[list[str]] = []
        self.doc_ids: list[str] = []
        self.df: Counter[str] = Counter()
        self.avgdl: float = 0.0

    def build(self, doc_ids: Iterable[str], texts: Iterable[str]) -> "BM25Index":
        self.docs = [_tokens(t) for t in texts]
        self.doc_ids = list(doc_ids)
        self.df = Counter()
        for toks in self.docs:
            self.df.update(set(toks))
        self.avgdl = (sum(len(d) for d in self.docs) / len(self.docs)) if self.docs else 0.0
        return self

    def scores(self, query: str) -> list[float]:
        if not self.docs:
            return []
        n = len(self.docs)
        q = _tokens(query)
        out = []
        for toks in self.docs:
            tf = Counter(toks)
            dl = len(toks) or 1
            s = 0.0
            for term in q:
                f = tf.get(term, 0)
                if not f:
                    continue
                idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
            out.append(s)
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "k1": self.k1, "b": self.b, "doc_ids": self.doc_ids,
            "docs": self.docs, "df": dict(self.df), "avgdl": self.avgdl,
        }, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        idx = cls(k1=data.get("k1", 1.5), b=data.get("b", 0.75))
        idx.doc_ids = data["doc_ids"]
        idx.docs = data["docs"]
        idx.df = Counter(data["df"])
        idx.avgdl = data.get("avgdl", 0.0)
        return idx


def _chroma():
    from langchain_chroma import Chroma

    s = get_settings()
    return Chroma(
        collection_name="pyrag",
        embedding_function=cached_embeddings(),
        persist_directory=str(s.chroma_dir),
    )


class HybridVectorStore:
    """ Fachada com Chroma (denso) + BM25 (lexical) e busca hibrida com rerank."""

    def __init__(self) -> None:
        self._store = None
        self._bm25: BM25Index | None = None

    # ------------------------------------------------------------------ infra
    @property
    def bm25_path(self) -> Path:
        return get_settings().cache_dir / "bm25.json"

    @property
    def store(self):
        if self._store is None:
            self._store = _chroma()
        return self._store

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index.load(self.bm25_path)
        return self._bm25

    # ----------------------------------------------------------------- escrita
    def upsert(self, chunks: list[Document], rebuild_bm25: bool = True) -> int:
        if not chunks:
            return 0
        for i, c in enumerate(chunks):
            c.metadata.setdefault("uid", f"{c.metadata.get('path', 'doc')}#{i}")
        ids = [c.metadata["uid"] for c in chunks]
        self.store.add_documents(chunks, ids=ids)
        self._bm25 = None
        if rebuild_bm25:
            self.rebuild_bm25()
        return len(chunks)

    def rebuild_bm25(self) -> int:
        data = self.store.get(include=["documents", "metadatas"])
        ids = data.get("ids", [])
        docs = data.get("documents", []) or []
        metas = data.get("metadatas", []) or [{}] * len(docs)
        self.bm25.build(ids, [f"{d}\n{json.dumps(m, ensure_ascii=False)}" for d, m in zip(docs, metas)])
        self.bm25.save(self.bm25_path)
        log.info("BM25 reindexado com %d documentos", len(ids))
        return len(ids)

    def delete_collection(self) -> None:
        self.store.delete_collection()
        self._store = None
        self.bm25_path.unlink(missing_ok=True)
        self._bm25 = None

    def count(self) -> int:
        return self.store._collection.count()  # noqa: SLF001

    # ------------------------------------------------------------------ leitura
    def similar(self, query: str, k: int | None = None) -> list[dict]:
        k = k or get_settings().top_k_rerank
        out = []
        for doc, dist in self.store.similarity_search_with_score(query, k=k):
            out.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "dense_score": 1.0 / (1.0 + float(dist)),
                "lexical_score": 0.0,
                "score": 1.0 / (1.0 + float(dist)),
            })
        return out

    def search(self, query: str, k: int | None = None, alpha: float = 0.6) -> list[dict]:
        """Busca hibrida: alpha peso do denso, (1-alpha) do lexical; depois rerank."""
        s = get_settings()
        k = k or s.top_k
        pool = max(s.top_k_rerank, k)
        dense = self.similar(query, k=pool)
        by_id = {d["metadata"].get("uid", str(i)): d for i, d in enumerate(dense)}

        lex_norm = _norm(self.bm25.scores(query))
        lexical_by_id = dict(zip(self.bm25.doc_ids, lex_norm))

        merged: list[dict] = []
        for uid, d in by_id.items():
            lex = lexical_by_id.get(uid, 0.0)
            d["lexical_score"] = lex
            d["score"] = alpha * d["dense_score"] + (1 - alpha) * lex
            merged.append(d)

        # docs com apenas match lexical ainda entram no pool
        for uid, lex in zip(self.bm25.doc_ids, lex_norm):
            if uid not in by_id and lex > 0.05:
                merged.append(self._materialize(uid, lex, alpha))

        merged = [m for m in merged if m["score"] > s.similarity_threshold]
        merged.sort(key=lambda x: x["score"], reverse=True)
        return self._rerank(query, merged)[:k]

    def _materialize(self, uid: str, lex: float, alpha: float) -> dict:
        data = self.store.get(ids=[uid], include=["documents", "metadatas"])
        docs = data.get("documents") or [""]
        metas = data.get("metadatas") or [{}]
        return {"text": docs[0], "metadata": metas[0], "dense_score": 0.0,
                "lexical_score": lex, "score": (1 - alpha) * lex}

    def _rerank(self, query: str, docs: list[dict]) -> list[dict]:
        """Reranking leve: score de busca + cobertura de termos + sinais de qualidade."""
        if get_settings().rerank == "none" or not docs:
            return docs
        q_terms = set(_tokens(query))
        for d in docs:
            body = d["text"]
            toks = set(_tokens(body))
            coverage = len(q_terms & toks) / (len(q_terms) or 1)
            numbers = 1.0 if re.search(r"\d", body) else 0.0
            title_bonus = 0.15 if any(t in str(d["metadata"].get("title", "")).lower() for t in q_terms) else 0.0
            size_penalty = 0.0 if 200 <= len(body) <= 4000 else 0.08
            d["rerank_score"] = (0.6 * d["score"] + 0.25 * coverage + 0.05 * numbers
                                 + title_bonus - size_penalty)
        docs.sort(key=lambda x: x.get("rerank_score", x["score"]), reverse=True)
        for d in docs:
            d["score"] = d.get("rerank_score", d["score"])
        return docs


_store_singleton: HybridVectorStore | None = None


def get_store() -> HybridVectorStore:
    global _store_singleton
    if _store_singleton is None:
        _store_singleton = HybridVectorStore()
    return _store_singleton
    from langchain_chroma import Chroma

    s = get_settings()
    return Chroma(
        collection_name="pyrag",
        embedding_function=cached_embeddings(),
        persist_directory=str(s.chroma_dir),
    )


def _norm(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [1.0 if hi > 0 else 0.0 for _ in scores]
    return [(x - lo) / (hi - lo) for x in scores]