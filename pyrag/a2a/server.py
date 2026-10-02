"""Servidor A2A: Agent Card + JSON-RPC 2.0 (message/send, stream, tasks/get|cancel).

Independente de framework: `A2AServer` trata payloads JSON-RPC e pode ser montado
em FastAPI (build_a2a_router) ou em qualquer servidor ASGI.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..config import get_settings
from ..models import (A2AMessage, A2APart, A2ATask, Artifact, JSONRPCError,
                      JSONRPCResponse, TaskStatus, new_id)
from .card import build_agent_card

log = logging.getLogger(__name__)

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class A2AServer:
    """Lado servidor do protocolo A2A."""

    def __init__(self, handler: Callable[[str, dict[str, Any] | None], Any] | None = None,
                 name: str | None = None) -> None:
        self.handler = handler or default_handler
        self.tasks: dict[str, A2ATask] = {}
        self.name = name or get_settings().agent_name

    def agent_card(self, base_url: str | None = None) -> dict:
        return build_agent_card(base_url).model_dump(mode="json")

    def _store(self, task: A2ATask) -> A2ATask:
        self.tasks[task.id] = task
        return task

    # ------------------------------------------------------------ rpc dispatch
    def handle_rpc(self, payload: dict) -> dict:
        if not isinstance(payload, dict) or payload.get("jsonrpc") != "2.0":
            return JSONRPCResponse(id=(payload or {}).get("id"), error=JSONRPCError(
                code=INVALID_REQUEST, message="Requisicao JSON-RPC 2.0 invalida")).model_dump(mode="json")
        rpc_id = payload.get("id")
        method = payload.get("method")
        params = payload.get("params") or {}
        try:
            if method == "message/send":
                result = self.message_send(params)
            elif method == "message/stream":
                result = self.message_stream(params)
            elif method == "tasks/get":
                result = self.task_get(params)
            elif method == "tasks/cancel":
                result = self.task_cancel(params)
            elif method == "tasks/list":
                result = self.task_list()
            elif method in ("agent/getAuthenticatedExtendedCard", "agent/card"):
                result = self.agent_card()
            else:
                return JSONRPCResponse(id=rpc_id, error=JSONRPCError(
                    code=METHOD_NOT_FOUND, message=f"Metodo nao suportado: {method}")).model_dump(mode="json")
            return JSONRPCResponse(id=rpc_id, result=result).model_dump(mode="json")
        except KeyError as exc:
            return JSONRPCResponse(id=rpc_id, error=JSONRPCError(
                code=INVALID_PARAMS, message=str(exc))).model_dump(mode="json")
        except Exception as exc:  # pragma: no cover
            log.exception("Erro no metodo A2A %s", method)
            return JSONRPCResponse(id=rpc_id, error=JSONRPCError(
                code=INTERNAL_ERROR, message=str(exc))).model_dump(mode="json")

    # ---------------------------------------------------------------- mensagens
    def _task_from_params(self, params: dict) -> tuple[A2ATask, A2AMessage]:
        raw = params.get("message")
        if not isinstance(raw, dict) or not raw:
            raise KeyError("params.message e obrigatorio")
        message = A2AMessage(**raw)
        ctx = (params.get("configuration") or {}).get("contextId") or message.context_id
        task = A2ATask(context_id=ctx or new_id("ctx"))
        message.task_id = task.id
        message.context_id = task.context_id
        task.history.append(message)
        task.status = TaskStatus(state="submitted", message=A2AMessage.text_message(
            "Tarefa recebida; processando...", role="agent", task_id=task.id,
            context_id=task.context_id))
        return task, message

    def _run(self, task: A2ATask, message: A2AMessage) -> A2ATask:
        t0 = task.metadata.pop("_t0", time.time())
        task.status = TaskStatus(state="working", message=A2AMessage.text_message(
            "Processando...", role="agent", task_id=task.id, context_id=task.context_id))
        self._store(task)
        try:
            result = self.handler(message.text(), {"task": task, "history": task.history})
            text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
            task.status = TaskStatus(state="completed", message=A2AMessage.text_message(
                text, role="agent", task_id=task.id, context_id=task.context_id))
            task.artifacts.append(Artifact(name="resposta", description="Resposta do agente",
                                           parts=[A2APart(kind="text", text=text)]))
            task.metadata["elapsed_ms"] = int((time.time() - t0) * 1000)
        except Exception as exc:
            log.exception("Falha ao processar task %s", task.id)
            task.status = TaskStatus(state="failed", message=A2AMessage.text_message(
                f"Erro interno: {exc}", role="agent", task_id=task.id, context_id=task.context_id))
        return self._store(task)

    # ---------------------------------------------------------------- mensagens
    def message_send(self, params: dict) -> dict:
        task, message = self._task_from_params(params)
        task.metadata["_t0"] = time.time()
        return self._run(task, message).model_dump(mode="json")

    def message_stream(self, params: dict) -> dict:
        """Formato de task unica; o streaming real e via stream_events (SSE)."""
        return self.message_send(params)

    # -------------------------------------------------------------------- tasks
    def task_get(self, params: dict) -> dict:
        task_id = params.get("id") or params.get("taskId")
        task = self.tasks.get(str(task_id))
        if task is None:
            raise KeyError(f"Task nao encontrada: {task_id}")
        return task.model_dump(mode="json")

    def task_cancel(self, params: dict) -> dict:
        task_id = params.get("id") or params.get("taskId")
        task = self.tasks.get(str(task_id))
        if task is None:
            raise KeyError(f"Task nao encontrada: {task_id}")
        if task.status.state in ("completed", "failed", "canceled"):
            return task.model_dump(mode="json")
        task.status = TaskStatus(state="canceled")
        return self._store(task).model_dump(mode="json")

    def task_list(self) -> dict:
        return {"tasks": [t.model_dump(mode="json") for t in list(self.tasks.values())[-50:]]}

    # --------------------------------------------------------------- streaming SSE
    async def stream_events(self, payload: dict):
        """Gerador SSE: emite task submitted, working e completed."""
        def _sse(obj: Any) -> str:
            return f"data: {json.dumps(obj, ensure_ascii=False, default=str)}\n\n"

        rpc_id = payload.get("id")
        task, message = self._task_from_params(payload.get("params") or {})
        task.metadata["_t0"] = time.time()
        self._store(task)
        yield _sse(JSONRPCResponse(id=rpc_id, result=task.model_dump(mode="json")).model_dump(mode="json"))
        await asyncio.sleep(0)
        task.status = TaskStatus(state="working")
        self._store(task)
        yield _sse(JSONRPCResponse(id=rpc_id, result=task.model_dump(mode="json")).model_dump(mode="json"))
        yield _sse(JSONRPCResponse(id=rpc_id, result=self._run(task, message).model_dump(mode="json")).model_dump(mode="json"))
        yield "data: [DONE]\n\n"


def default_handler(text: str, ctx: dict[str, Any] | None = None) -> str:
    """Handler padrao: roteia a pergunta para RAG, agente ou outro agente A2A.

    - pedido explicito de ferramenta/agente -> agente ReAct
    - caso contrario -> pipeline RAG com memoria da task
    """
    low = text.lower()
    wants_agent = any(k in low for k in ("use as ferramentas", "use ferramentas", "tool", "agente",
                                         "delegue", "step-by-step", "passo a passo", "pesquise na web"))
    ctx = ctx or {}
    task: A2ATask | None = ctx.get("task")

    try:
        if wants_agent:
            from ..agent import run_agent

            result = run_agent(text)
            return result.get("output", "")

        from ..rag import RAGPipeline
        from ..memory import load_session

        session = f"a2a-{task.context_id}" if task else "a2a-default"
        answer = RAGPipeline().chat(text, session=session)
        return answer.answer
    except Exception as exc:  # pragma: no cover
        log.exception("Handler padrao falhou")
        return f"Erro ao processar: {exc}"


def build_a2a_router(app, server: A2AServer | None = None):
    """Monta os endpoints A2A em uma instancia FastAPI existente."""
    srv = server or A2AServer()
    router = APIRouter(tags=["a2a"])

    @router.get("/.well-known/agent.json")
    @router.get("/.well-known/agent-card.json")
    @router.get("/agent/card")
    async def agent_card(request: Request):
        base = str(request.base_url).rstrip("/")
        return srv.agent_card(base)

    @router.post("/a2a")
    async def rpc(request: Request):
        payload = await request.json()
        if isinstance(payload, list):
            return JSONResponse([srv.handle_rpc(p) for p in payload])
        if payload.get("method") == "message/stream":
            return StreamingResponse(srv.stream_events(payload), media_type="text/event-stream")
        return JSONResponse(srv.handle_rpc(payload))

    @router.get("/a2a/tasks")
    async def list_tasks():
        return srv.task_list()

    @router.get("/a2a/health")
    async def health():
        return {"status": "ok", "agent": srv.name, "tasks": len(srv.tasks)}

    app.include_router(router)
    return srv


def run_a2a_server(host: str | None = None, port: int | None = None) -> None:
    """Sobe um servidor dedicado so para A2A."""
    import uvicorn

    s = get_settings()
    app = FastAPI(title=f"pyrag A2A ({s.agent_name})", version=s.agent_version)

    @app.get("/")
    async def root():
        return {"agent": s.agent_name, "protocol": "a2a/0.3.0",
                "endpoints": ["/.well-known/agent.json", "/a2a (JSON-RPC 2.0)", "/a2a/health"]}

    build_a2a_router(app)
    uvicorn.run(app, host=host or s.a2a_bind_host, port=port or s.a2a_bind_port,
                log_level=s.log_level.lower())