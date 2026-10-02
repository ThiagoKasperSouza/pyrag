"""Fabrica de LLMs gratuitos e embeddings, com deteccao automatica de provider."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from .config import Settings, get_settings

log = logging.getLogger(__name__)


def _ollama_available(base_url: str, timeout: float = 2.0) -> bool:
    try:
        import httpx

        httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        return True
    except Exception:
        return False


def resolve_provider(settings: Settings | None = None) -> str:
    """Resolve 'auto' para um provider concreto usando disponibilidade real."""
    s = settings or get_settings()
    p = s.llm_provider
    if p != "auto":
        return p
    if _ollama_available(s.ollama_base_url):
        return "ollama"
    if s.groq_api_key:
        return "groq"
    if s.gemini_api_key:
        return "gemini"
    if s.openrouter_api_key:
        return "openrouter"
    return "none"


def get_chat_llm(reasoner: bool = False, temperature: float | None = None, **kwargs: Any):
    """Retorna um chat model. Levanta erro claro se nenhum provider estiver disponivel."""
    s = get_settings()
    provider = resolve_provider(s)
    temp = s.temperature if temperature is None else temperature
    base: dict[str, Any] = {"temperature": temp, "max_tokens": s.max_tokens, **kwargs}

    if provider == "ollama":
        if not _ollama_available(s.ollama_base_url):
            raise RuntimeError(
                f"Ollama configurado em {s.ollama_base_url} mas nao responde. "
                "Inicie com `ollama serve` ou troque PYRAG_LLM_PROVIDER."
            )
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=s.reasoning_model if reasoner else s.chat_model,
            base_url=s.ollama_base_url,
            num_predict=s.max_tokens,
            temperature=temp,
        )

    if provider == "groq":
        if not s.groq_api_key:
            raise RuntimeError("GROQ_API_KEY nao configurada (use .env ou PYRAG_LLM_PROVIDER=ollama).")
        from langchain_groq import ChatGroq

        return ChatGroq(model=s.groq_reasoning_model if reasoner else s.groq_chat_model,
                        api_key=s.groq_api_key, **base)

    if provider == "gemini":
        if not s.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY nao configurada.")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=s.gemini_reasoning_model if reasoner else s.gemini_chat_model,
                                      google_api_key=s.gemini_api_key, temperature=temp,
                                      max_output_tokens=s.max_tokens)

    if provider == "openrouter":
        if not s.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY nao configurada.")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=s.openrouter_reasoning_model if reasoner else s.openrouter_chat_model,
                          api_key=s.openrouter_api_key, base_url="https://openrouter.co/api/v1", **base)

    raise RuntimeError(
        "Nenhum LLM disponivel. Suba o Ollama (`ollama serve`) ou configure uma chave "
        "gratuita (GROQ_API_KEY / GEMINI_API_KEY / OPENROUTER_API_KEY) no .env."
    )


class ExtractiveFallbackLLM(BaseChatModel):
    """LLM sem API: extrai frases do contexto. Permite rodar o sistema offline."""

    max_tokens: int = 700

    @property
    def _llm_type(self) -> str:
        return "extractive-fallback"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        prompt = "\n".join(
            m.content if isinstance(m.content, str) else str(m.content) for m in messages
        )
        answer = _extractive_answer(prompt, self.max_tokens)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=answer))])


def _extractive_answer(prompt: str, max_chars: int) -> str:
    ctx = prompt
    idx = prompt.find("Contexto")
    if idx != -1:
        ctx = prompt[idx:]
    sentences = [s.strip() for s in ctx.replace("\n", " ").split(". ") if len(s.strip()) > 40]
    picked = sentences[:6] or [ctx.strip()[:max_chars]]
    out = "[RESPOSTA EXTRATIVA - nenhum LLM configurado]\n\n"
    return (out + "\n".join(f"- {s.rstrip('.')}." for s in picked))[:max_chars]


def effective_provider() -> str:
    """Provider que realmente sera usado: 'none' quando o configurado esta fora do ar."""
    p = resolve_provider()
    if p == "ollama" and not _ollama_available(get_settings().ollama_base_url):
        return "none"
    return p


def get_llm_or_fallback(reasoner: bool = False, **kwargs: Any) -> BaseChatModel:
    """Tenta um LLM real; se indisponivel, usa o fallback extrativo."""
    try:
        return get_chat_llm(reasoner=reasoner, **kwargs)
    except Exception as exc:  # pragma: no cover - depende do ambiente
        log.warning("LLM indisponivel (%s); usando fallback extrativo.", exc)
        return ExtractiveFallbackLLM(max_tokens=get_settings().max_tokens)


def get_embeddings() -> Embeddings:
    """Embeddings locais gratuitos (Ollama -> fastembed -> hash fallback)."""
    s = get_settings()

    if _ollama_available(s.ollama_base_url):
        try:
            from langchain_ollama import OllamaEmbeddings

            return OllamaEmbeddings(model=s.embed_model, base_url=s.ollama_base_url)
        except Exception as exc:  # pragma: no cover
            log.warning("OllamaEmbeddings falhou (%s); tentando fastembed.", exc)

    try:
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

        return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    except Exception as exc:  # pragma: no cover
        log.warning("fastembed indisponivel (%s); usando embeddings por hash.", exc)
        return HashingEmbeddings()


class HashingEmbeddings(Embeddings):
    """Embeddings deterministicos por hashing (sem download; so para smoke tests)."""

    dim: int = 384

    def _vec(self, text: str) -> list[float]:
        import hashlib
        import math

        v = [0.0] * self.dim
        for token in (text or "").lower().split():
            h = hashlib.sha256(token.encode()).digest()
            v[int.from_bytes(h[:4], "big") % self.dim] += 1.0 if h[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@lru_cache(maxsize=4)
def _cached_embeddings() -> Embeddings:
    return get_embeddings()


def cached_embeddings() -> Embeddings:
    return _cached_embeddings()
    raise RuntimeError(
        "Nenhum LLM disponivel. Suba o Ollama (`ollama serve`) ou configure uma chave "
        "gratuita (GROQ_API_KEY / GEMINI_API_KEY / OPENROUTER_API_KEY) no .env."
    )