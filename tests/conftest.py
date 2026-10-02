"""Fixtures compartilhadas: garante config offline e deterministica."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Config offline antes de importar qualquer modulo do pyrag
os.environ.setdefault("PYRAG_DATA_DIR", str(ROOT / "data_test"))
os.environ.setdefault("PYRAG_CHROMA_DIR", str(ROOT / "data_test" / "chroma"))
os.environ.setdefault("PYRAG_CACHE_DIR", str(ROOT / "data_test" / "cache"))
os.environ.setdefault("PYRAG_SQL_DB_PATH", str(ROOT / "data_test" / "test.db"))
os.environ.setdefault("PYRAG_A2A_PEERS_FILE", str(ROOT / "data_test" / "peers.json"))
os.environ.setdefault("PYRAG_LLM_PROVIDER", "none")
os.environ.setdefault("PYRAG_ENABLE_WEB_SEARCH", "false")
os.environ.setdefault("PYRAG_CHUNK_SIZE", "400")
os.environ.setdefault("PYRAG_CHUNK_OVERLAP", "50")

import pytest  # noqa: E402

from pyrag.config import get_settings  # noqa: E402


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture(scope="session")
def sample_store():
    """Store com 3 documentos curtos e embeddings deterministicos (hash)."""
    from langchain_core.documents import Document

    from pyrag.llm import HashingEmbeddings
    from pyrag.vectorstore import HybridVectorStore

    store = HybridVectorStore()
    store._store = _FakeChroma(HashingEmbeddings())
    store._bm25 = None

    docs = [
        Document(page_content="O sistema pyrag usa recuperacao hibrida com Chroma e BM25. "
                              "O score final combina busca densa e busca lexical.",
                 metadata={"path": "/docs/arquitetura.md", "title": "arquitetura", "chunk": 0}),
        Document(page_content="A comunicacao A2A usa JSON-RPC 2.0 e Agent Card publicado em "
                              "/.well-known/agent.json para descoberta entre agentes.",
                 metadata={"path": "/docs/a2a.md", "title": "a2a", "chunk": 0}),
        Document(page_content="O agente ReAct usa ferramentas como calculadora, busca web, "
                              "SQL somente leitura e delegacao A2A.",
                 metadata={"path": "/docs/agente.md", "title": "agente", "chunk": 0}),
    ]
    store.upsert(docs)
    return store


class _FakeCollection:
    def __init__(self, docs, embeddings):
        self._ids: list[str] = []
        self._docs: list[str] = []
        self._metas: list[dict] = []
        self._embeddings = embeddings

    def add(self, ids, documents, metadatas, embeddings=None):
        self._ids.extend(ids)
        self._docs.extend(documents)
        self._metas.extend(metadatas)

    def count(self):
        return len(self._ids)

    def get(self, ids=None, include=None):
        idxs = range(len(self._ids)) if ids is None else [self._ids.index(i) for i in ids if i in self._ids]
        return {"ids": [self._ids[i] for i in idxs],
                "documents": [self._docs[i] for i in idxs],
                "metadatas": [self._metas[i] for i in idxs]}


class _FakeChroma:
    """Chroma minimo em memoria, para testes sem dependencia de persistencia."""

    def __init__(self, embeddings):
        self._collection = _FakeCollection([], embeddings)
        self.embedding_function = embeddings

    def add_documents(self, docs, ids=None):
        texts = [d.page_content for d in docs]
        vectors = self.embedding_function.embed_documents(texts)
        ids = ids or [str(i) for i in range(len(docs))]
        self._collection.add(ids, texts, [d.metadata for d in docs], vectors)

    def get(self, include=None, ids=None):
        return self._collection.get(ids=ids)

    def similarity_search_with_score(self, query, k=4):
        qv = self.embedding_function.embed_query(query)
        docs = self._collection.get()
        scored = []
        for i, (uid, text) in enumerate(zip(docs["ids"], docs["documents"])):
            vec = self.embedding_function.embed_query(text)
            scored.append((1 - _cos(qv, vec), uid, text, docs["metadatas"][i]))
        scored.sort()
        return [(_FakeDoc(text, meta), dist) for dist, _uid, text, meta in scored[:k]]

    def delete_collection(self):
        self._collection = _FakeCollection([], self.embedding_function)


class _FakeDoc:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b))