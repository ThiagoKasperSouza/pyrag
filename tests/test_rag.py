"""Testes de retrieval, BM25, rerank e pipeline RAG (sem LLM)."""

from __future__ import annotations

import math

from langchain_core.documents import Document

from pyrag.rag import RAGPipeline
from pyrag.vectorstore import BM25Index, _tokens, _norm


def test_tokens_normalizes():
    assert _tokens("O pyrag usa A2A, JSON-RPC 2.0!") == ["o", "pyrag", "usa", "a2a", "json", "rpc", "2", "0"]


def test_norm_scores():
    assert _norm([]) == []
    assert _norm([1.0, 1.0]) == [1.0, 1.0]
    out = _norm([1.0, 3.0])
    assert math.isclose(out[0], 0.0) and math.isclose(out[1], 1.0)


def test_bm25_ranks_relevant_doc_first():
    idx = BM25Index().build(
        ["a", "b"],
        ["o gato dorme no sofa", "a reuniao de预算 ocorreu em marco"],
    )
    scores = idx.scores("reuniao de marco")
    assert scores[1] > scores[0]


def test_bm25_persistence_roundtrip(tmp_path):
    idx = BM25Index().build(["a"], ["texto de teste com BM25"])
    p = tmp_path / "bm25.json"
    idx.save(p)
    loaded = BM25Index.load(p)
    assert loaded.doc_ids == ["a"]
    assert loaded.scores("teste")[0] > 0


def test_search_returns_chunks_with_sources(sample_store):
    chunks = sample_store.search("recuperacao hibrida BM25", k=2)
    assert 1 <= len(chunks) <= 2
    top = chunks[0]
    assert "path" in top["metadata"]
    assert top["score"] > 0


def test_rerank_prefers_term_overlap(sample_store):
    res = sample_store.search("Agent Card JSON-RPC descoberta agentes", k=1)
    assert "agent.json" in res[0]["text"]


def test_rag_pipeline_retrieve_and_ask(sample_store):
    pipeline = RAGPipeline(store=sample_store)
    chunks = pipeline.retrieve("BM25", k=2)
    assert chunks and chunks[0].source.path
    assert chunks[0].source.preview

    answer = pipeline.ask("como funciona o BM25?", k=2)
    assert answer.answer  # fallback extrativo retorna texto
    assert answer.elapsed_ms >= 0
    assert answer.used_fallback is True
    assert len(answer.citations) == len(answer.chunks)


def test_rag_context_includes_numbered_sources(sample_store):
    pipeline = RAGPipeline(store=sample_store)
    chunks = pipeline.retrieve("A2A", k=2)
    ctx = pipeline.build_context(chunks)
    assert "[1]" in ctx and "arquivo=" in ctx


def test_ask_without_index_returns_empty_message(tmp_path, monkeypatch):
    import pyrag.vectorstore as vs
    from pyrag.config import reset_settings_cache

    monkeypatch.setenv("PYRAG_CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("PYRAG_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("PYRAG_DATA_DIR", str(tmp_path / "data"))
    reset_settings_cache()
    vs._store_singleton = None          # garante store limpo para este teste
    try:
        pipeline = RAGPipeline()
        answer = pipeline.ask("qualquer pergunta", k=2)
        assert "Nao encontrei" in answer.answer
        assert answer.chunks == []
    finally:
        vs._store_singleton = None
        reset_settings_cache()


def test_index_documents_and_count(tmp_path, monkeypatch):
    from pyrag.config import reset_settings_cache
    from tests.conftest import _FakeChroma
    from pyrag.llm import HashingEmbeddings
    from pyrag.vectorstore import HybridVectorStore

    monkeypatch.setenv("PYRAG_CACHE_DIR", str(tmp_path / "cache"))
    reset_settings_cache()
    try:
        store = HybridVectorStore()
        store._store = _FakeChroma(HashingEmbeddings())
        n = store.upsert([Document(page_content="doc de teste sobre LangChain",
                                   metadata={"path": "/a.md", "title": "a"})])
        assert n == 1
        assert store.count() == 1
        assert store.rebuild_bm25() == 1
    finally:
        reset_settings_cache()