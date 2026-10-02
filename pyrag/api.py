"""API HTTP: REST para RAG/agente + endpoints A2A na mesma aplicacao."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .a2a.peers import list_peers
from .a2a.server import A2AServer, build_a2a_router
from .config import get_settings
from .llm import effective_provider
from .memory import clear_session, list_sessions
from .rag import RAGPipeline

log = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session: str = "default"
    k: int | None = None


class SearchRequest(BaseModel):
    query: str
    k: int = 6


class AgentRequest(BaseModel):
    task: str
    session: str = "default"


class IngestRequest(BaseModel):
    paths: list[str]
    collection: str = "default"


class PeerRequest(BaseModel):
    name: str
    url: str


def create_app() -> FastAPI:
    s = get_settings()
    s.ensure_dirs()
    pipeline = RAGPipeline()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(level=getattr(logging, s.log_level.upper(), logging.INFO))
        log.info("pyrag API no ar | provider=%s | dados=%s", effective_provider(), s.data_dir)
        yield

    app = FastAPI(title="pyrag API", version=s.agent_version,
                  description="RAG com LLMs gratuitos, ferramentas agenticas e A2A",
                  lifespan=lifespan)

    # ------------------------------------------------------------- health/meta
    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "provider": effective_provider(), "index": pipeline.status()}

    @app.get("/tools")
    def tools() -> list[dict]:
        from .tools import get_tools

        return [{"name": t.name, "description": t.description, "args": t.args} for t in get_tools()]

    @app.get("/peers")
    def peers() -> list[dict]:
        return [p.model_dump() for p in list_peers()]

    @app.post("/peers")
    def add_peer_route(body: PeerRequest) -> dict:
        from .a2a.peers import add_peer

        return add_peer(body.name, body.url).model_dump()

    @app.get("/sessions")
    def sessions() -> list[str]:
        return list_sessions()

    @app.delete("/sessions/{session}")
    def drop_session(session: str) -> dict:
        return {"removed": clear_session(session)}

    # ----------------------------------------------------------------- indexing
    @app.post("/ingest")
    def ingest_route(body: IngestRequest) -> dict:
        try:
            n = pipeline.index_paths(body.paths, collection=body.collection)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"indexed_chunks": n, "total": pipeline.status()}

    # ---------------------------------------------------------------------- RAG
    @app.post("/search")
    def search_route(body: SearchRequest) -> dict:
        chunks = pipeline.retrieve(body.query, k=body.k)
        return {"query": body.query, "results": [
            {"rank": i, "score": c.score, "source": c.source.model_dump(),
             "text": c.text[:1200]} for i, c in enumerate(chunks, 1)]}

    @app.post("/ask")
    def ask_route(body: AskRequest) -> dict:
        answer = pipeline.chat(body.question, session=body.session)
        return answer.model_dump(mode="json")

    # -------------------------------------------------------------------- agente
    @app.post("/agent/run")
    def agent_route(body: AgentRequest) -> dict:
        from .agent import run_agent
        from .memory import load_session

        result = run_agent(body.task, history=load_session(body.session))
        clear_session(body.session)
        return result

    # --------------------------------------------------------------------- A2A
    a2a_server = A2AServer()
    if s.a2a_enabled:
        build_a2a_router(app, a2a_server)

    app.state.pipeline = pipeline
    app.state.a2a = a2a_server
    return app


app = create_app()


def run(host: str | None = None, port: int | None = None) -> None:
    import uvicorn

    s = get_settings()
    uvicorn.run(app, host=host or s.api_host, port=port or s.api_port, log_level=s.log_level.lower())