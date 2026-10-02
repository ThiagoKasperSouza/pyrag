#!/usr/bin/env python3
"""Valida o pipeline com LLM real (Ollama) ou fallback, depending do que estiver no ar."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pyrag.llm import effective_provider  # noqa: E402
from pyrag.rag import RAGPipeline  # noqa: E402

provider = effective_provider()
print(f"provider efetivo = {provider}\n")

pipeline = RAGPipeline()
print("status:", pipeline.status())

print("\n=== Recuperacao hibrida (top 3) ===")
chunks = pipeline.retrieve("como a busca hibrida reordena os resultados?", k=3)
for i, c in enumerate(chunks, 1):
    print(f"[{i}] score={c.score:.3f} (denso={c.dense_score:.3f} "
          f"lexical={c.lexical_score:.3f}) {Path(c.source.path).name} chunk={c.source.chunk}")
    print(f"    {c.text[:150]}...".replace("\n", " "))

print("\n=== Resposta com LLM ===")
answer = pipeline.ask("como a busca hibrida reordena os resultados?")
print(f"modelo={answer.model} fallback={answer.used_fallback} {answer.elapsed_ms}ms")
print(answer.answer[:900])
print(f"\ncitacoes ({len(answer.citations)}):")
for c in answer.citations[:4]:
    print(f"  [{c.index}] {c.source.path} (chunk {c.source.chunk}, score {c.source.score})")

print("\n=== Memoria: pergunta anforica ===")
a2 = pipeline.chat("e o BM25?", session="llm-check")
print(a2.answer[:400])

print("\n=== Ferramenta rag_search ===")
from pyrag.tools import rag_search  # noqa: E402

print(rag_search.invoke({"query": "Agent Card", "k": 1})[:250])

print("\nOK" if answer.answer and not answer.used_fallback else
      "\nOK (fallback extrativo - sem LLM disponivel)")