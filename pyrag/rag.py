"""Pipeline RAG: reescrita de query -> busca hibrida -> prompt com citacoes -> resposta."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .config import get_settings
from .llm import effective_provider, get_llm_or_fallback
from .memory import append_turn, clear_session, load_session
from .models import Citation, RAGAnswer, RetrievedChunk, Source
from .vectorstore import get_store

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """Voce e um assistente de RAG preciso. Responda APENAS com base no CONTEXTO fornecido.
Regras:
- Se a resposta nao estiver no contexto, diga exatamente: "Nao encontrei essa informacao nos documentos."
- Cite a fonte usando [n] imediatamente apos a afirmacao, com n = numero do chunk.
- Seja conciso e organize a resposta em topicos quando houver mais de um item.
- Nunca invente dados, numeros ou nomes."""

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("system", "[CONTEXTO]\n{context}\n[FIM DO CONTEXTO]"),
    MessagesPlaceholder(variable_name="history", optional=True),
    ("human", "Pergunta: {question}\n\nResponda usando o contexto acima, com citacoes [n]."),
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Reescreva a pergunta do usuario como uma busca autocontida e densa em PT-BR. "
               "Responda SOMENTE com a consulta reescrita, sem aspas."),
    ("human", "Historico:\n{history}\n\nPergunta: {question}"),
])


class RAGPipeline:
    """Encapsula o fluxo RAG completo."""

    def __init__(self, store=None) -> None:
        self.store = store or get_store()

    # ------------------------------------------------------------------ index
    def index_documents(self, docs: Iterable[Document], rebuild: bool = True) -> int:
        return self.store.upsert(list(docs), rebuild_bm25=rebuild)

    def index_paths(self, paths: Iterable[str], collection: str = "default") -> int:
        from .ingest import ingest

        return self.index_documents(ingest(paths, collection=collection))

    def status(self) -> dict:
        s = get_settings()
        return {
            "chunks": self.store.count(),
            "chroma_dir": str(s.chroma_dir),
            "bm25_docs": len(self.store.bm25.doc_ids),
            "provider": effective_provider(),
        }

    # --------------------------------------------------------------- retrieval
    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        hits = self.store.search(query, k=k or get_settings().top_k)
        chunks: list[RetrievedChunk] = []
        for h in hits:
            meta = h["metadata"]
            path = str(meta.get("path") or meta.get("source") or "desconhecido")
            chunks.append(RetrievedChunk(
                text=h["text"],
                metadata=meta,
                score=float(h["score"]),
                dense_score=h.get("dense_score"),
                lexical_score=h.get("lexical_score"),
                source=Source(
                    path=path,
                    title=str(meta.get("title") or Path(path).stem or path),
                    chunk=int(meta.get("chunk", 0) or 0),
                    score=round(float(h["score"]), 4),
                    preview=h["text"][:280],
                ),
            ))
        return chunks

    def rewrite_query(self, question: str, history: list[BaseMessage] | None = None) -> str:
        """Reescreve a pergunta com o historico (para perguntas anforicas como 'e ele?')."""
        if not history:
            return question
        try:
            llm = get_llm_or_fallback()
            hist_txt = "\n".join(
                f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
                for m in history[-get_settings().history_turns * 2:]
            )
            out = REWRITE_PROMPT.invoke({"history": hist_txt, "question": question})
            rewritten = str(getattr(out, "content", out)).strip()
            return rewritten or question
        except Exception as exc:  # pragma: no cover
            log.warning("Falha na reescrita de query: %s", exc)
            return question

    @staticmethod
    def build_context(chunks: list[RetrievedChunk]) -> str:
        blocks = []
        for i, c in enumerate(chunks, start=1):
            src = c.source
            header = f"[{i}] arquivo={src.path}"
            if src.chunk:
                header += f" chunk={src.chunk}"
            header += f" score={src.score}"
            blocks.append(f"{header}\n{c.text.strip()}")
        return "\n\n---\n\n".join(blocks)

    # -------------------------------------------------------------------- ask
    def ask(
        self,
        question: str,
        history: list[BaseMessage] | None = None,
        k: int | None = None,
        retrieve_only: bool = False,
    ) -> RAGAnswer:
        start = time.time()
        provider = effective_provider()
        chunks = self.retrieve(self.rewrite_query(question, history), k=k)
        model_name = _model_label(provider)

        if retrieve_only:
            return RAGAnswer(query=question, answer="", chunks=chunks, provider=provider,
                             model=model_name, elapsed_ms=_ms(start))

        if not chunks:
            return RAGAnswer(
                query=question,
                answer="Nao encontrei essa informacao nos documentos.",
                chunks=[], provider=provider, model=model_name,
                elapsed_ms=_ms(start), used_fallback=provider == "none",
            )

        llm = get_llm_or_fallback()
        prompt = ANSWER_PROMPT.invoke({
            "context": self.build_context(chunks),
            "question": question,
            "history": history[-get_settings().history_turns * 2:] if history else [],
        })
        response = llm.invoke(prompt)
        text = getattr(response, "content", response)
        text = text.strip() if isinstance(text, str) else str(text)

        return RAGAnswer(
            query=question,
            answer=text,
            citations=[Citation(index=i, source=c.source) for i, c in enumerate(chunks, start=1)],
            chunks=chunks,
            provider=provider,
            model=model_name,
            elapsed_ms=_ms(start),
            used_fallback=provider == "none",
        )

    def chat(self, question: str, session: str = "default", k: int | None = None) -> RAGAnswer:
        """ask() + persistencia de historico na sessao."""
        history = load_session(session)
        result = self.ask(question, history=history, k=k)
        append_turn(session, question, result.answer)
        return result

    def clear_session(self, session: str = "default") -> None:
        clear_session(session)


def _ms(start: float) -> int:
    return int((time.time() - start) * 1000)


def _model_label(provider: str) -> str:
    s = get_settings()
    if provider == "ollama":
        return s.chat_model
    if provider == "groq":
        return s.groq_chat_model
    if provider == "gemini":
        return s.gemini_chat_model
    if provider == "openrouter":
        return s.openrouter_chat_model
    return "extractive-fallback"