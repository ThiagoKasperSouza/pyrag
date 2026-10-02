"""Configuracao central (12-factor via pydantic-settings + .env)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["ollama", "groq", "gemini", "openrouter", "auto", "none"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PYRAG_",
        env_file=(".env",),
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # ---------- LLM ----------
    llm_provider: Provider = "ollama"
    chat_model: str = "llama3.1:8b"
    reasoning_model: str = "qwen2.5:7b"
    embed_model: str = "embeddinggemma:300m"
    temperature: float = 0.1
    max_tokens: int = 1200
    history_turns: int = 6
    agent_max_iterations: int = 8

    ollama_base_url: str = "http://localhost:11434"
    groq_api_key: str = ""
    groq_chat_model: str = "llama-3.3-70b-versatile"
    groq_reasoning_model: str = "deepseek-r1-distill-llama-70b"
    gemini_api_key: str = ""
    gemini_chat_model: str = "gemini-2.0-flash"
    gemini_reasoning_model: str = "gemini-2.0-flash-thinking-exp-1219"
    openrouter_api_key: str = ""
    openrouter_chat_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    openrouter_reasoning_model: str = "deepseek/deepseek-r1-70b:free"

    # ---------- RAG ----------
    chunk_size: int = 1000
    chunk_overlap: int = 180
    top_k: int = 6
    top_k_rerank: int = 20
    rerank: str = "heuristic"
    similarity_threshold: float = 0.0

    # ---------- Storage ----------
    data_dir: Path = Path("./data")
    chroma_dir: Path = Path("./data/chroma")
    cache_dir: Path = Path("./data/cache")

    # ---------- Tools ----------
    enable_web_search: bool = True
    enable_sql: bool = True
    enable_calc: bool = True
    sql_db_path: Path = Path("./data/pyrag.db")
    sql_allow_write: bool = False

    # ---------- A2A ----------
    a2a_enabled: bool = True
    a2a_bind_host: str = "0.0.0.0"
    a2a_bind_port: int = 9100
    a2a_public_url: str = "http://localhost:9100"
    agent_name: str = "pyrag-orchestrator"
    agent_description: str = "Agente RAG com ferramentas que delega tarefas via A2A."
    agent_version: str = "0.1.0"
    a2a_peers_file: Path = Path("./data/peers.json")
    a2a_api_key: str = ""
    a2a_timeout: int = 180

    # ---------- Server ----------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    @field_validator("data_dir", "chroma_dir", "cache_dir", "sql_db_path", "a2a_peers_file", mode="before")
    @classmethod
    def _expand(cls, v):
        return Path(str(v)).expanduser()

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.chroma_dir, self.cache_dir, self.sql_db_path.parent):
            p.mkdir(parents=True, exist_ok=True)

    @property
    def has_cloud_llm(self) -> bool:
        return bool(
            (self.llm_provider in ("groq", "auto") and self.groq_api_key)
            or (self.llm_provider in ("gemini", "auto") and self.gemini_api_key)
            or (self.llm_provider in ("openrouter", "auto") and self.openrouter_api_key)
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s


def reset_settings_cache() -> None:
    get_settings.cache_clear()