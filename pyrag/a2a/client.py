"""Cliente A2A: fala JSON-RPC 2.0 com agentes remotos."""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator

from ..config import get_settings
from ..models import A2AMessage, AgentCard

log = logging.getLogger(__name__)


class A2AError(RuntimeError):
    """Erro retornado pelo agente remoto ou falha de transporte."""


class A2AClient:
    """Cliente minimo do protocolo A2A."""

    def __init__(self, base_url: str, timeout: int | None = None, api_key: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout or get_settings().a2a_timeout
        self.api_key = api_key if api_key is not None else get_settings().a2a_api_key
        self._id = 0

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    def _headers(self, streaming: bool = False) -> dict[str, str]:
        h = {"Content-Type": "application/json",
             "Accept": "text/event-stream" if streaming else "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _endpoint(self) -> str:
        return f"{self.base_url}/a2a"

    def _call(self, method: str, params: dict[str, Any]) -> Any:
        import httpx

        payload = {"jsonrpc": "2.0", "id": self._next_id(), "method": method, "params": params}
        try:
            r = httpx.post(self._endpoint(), json=payload, timeout=self.timeout, headers=self._headers())
        except Exception as exc:
            raise A2AError(f"Falha de transporte contacting {self.base_url}: {exc}") from exc
        if r.status_code >= 400:
            raise A2AError(f"HTTP {r.status_code} do agente {self.base_url}: {r.text[:300]}")
        data = r.json()
        if data.get("error"):
            err = data["error"]
            raise A2AError(f"{err.get('code')} {err.get('message')}: {err.get('data') or ''}")
        return data.get("result")

    # ------------------------------------------------------------------- card
    def get_agent_card(self) -> AgentCard:
        from .peers import discover_card

        return AgentCard(**discover_card(self.base_url))

    # --------------------------------------------------------------- mensagens
    def send_message(self, message: A2AMessage, context_id: str | None = None) -> Any:
        params: dict[str, Any] = {"message": message.model_dump(mode="json")}
        if context_id:
            params["configuration"] = {"contextId": context_id}
        return self._call("message/send", params)

    def send_text(self, text: str, role: str = "user", context_id: str | None = None) -> str:
        """Atalho: envia texto e devolve o texto da resposta do agente."""
        msg = A2AMessage.text_message(text, role=role, context_id=context_id)
        return extract_text(self.send_message(msg, context_id=context_id))

    def stream_message(self, message: A2AMessage, context_id: str | None = None) -> Iterator[str]:
        """Consome message/stream (SSE) e devolve os textos conforme chegam."""
        import httpx

        params: dict[str, Any] = {"message": message.model_dump(mode="json")}
        if context_id:
            params["configuration"] = {"contextId": context_id}
        payload = {"jsonrpc": "2.0", "id": self._next_id(), "method": "message/stream", "params": params}

        with httpx.stream("POST", self._endpoint(), json=payload, timeout=self.timeout,
                          headers=self._headers(streaming=True)) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if not chunk or chunk == "[DONE]":
                    continue
                try:
                    data = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                if data.get("error"):
                    raise A2AError(str(data["error"]))
                text = extract_text(data.get("result"))
                if text:
                    yield text

    # -------------------------------------------------------------------- tasks
    def get_task(self, task_id: str) -> Any:
        return self._call("tasks/get", {"id": task_id})

    def cancel_task(self, task_id: str) -> Any:
        return self._call("tasks/cancel", {"id": task_id})

    def list_tasks(self) -> Any:
        return self._call("tasks/list", {})

    # ------------------------------------------------------------------- health
    def ping(self) -> dict:
        """Checa se o agente remoto esta no ar e devolve nome e skills."""
        try:
            card = self.get_agent_card()
            return {"online": True, "name": card.name, "url": card.url,
                    "skills": [s.id for s in card.skills]}
        except Exception as exc:
            return {"online": False, "url": self.base_url, "error": str(exc)}


def extract_text(result: Any) -> str:
    """Extrai texto de um Task/Artifact/Message A2A, aceitando varios formatos."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        return "\n".join(filter(None, (extract_text(i) for i in result)))
    if not isinstance(result, dict):
        return str(result)

    parts: list[str] = []
    for art in result.get("artifacts") or []:
        parts.append(_parts_text(art.get("parts") if isinstance(art, dict) else []))
    if not parts:
        status = result.get("status") or {}
        msg = status.get("message") if isinstance(status, dict) else None
        if isinstance(msg, dict):
            parts.append(_parts_text(msg.get("parts") or []))
    if not parts:
        parts.append(_parts_text(result.get("parts") or []))
    return "\n".join(p for p in parts if p)


def _parts_text(parts: Any) -> str:
    if not isinstance(parts, list):
        return ""
    out: list[str] = []
    for p in parts:
        if isinstance(p, str):
            out.append(p)
        elif isinstance(p, dict):
            kind = p.get("kind", "text")
            if kind == "text" and p.get("text"):
                out.append(p["text"])
            elif kind == "data" and p.get("data"):
                out.append(json.dumps(p["data"], ensure_ascii=False))
            elif kind == "file" and p.get("file"):
                out.append(json.dumps(p["file"], ensure_ascii=False))
    return "\n".join(out)