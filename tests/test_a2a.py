"""Testes do protocolo A2A: server, card, peers e cliente (sem rede)."""

from __future__ import annotations

import json

from pyrag.a2a.card import build_agent_card, skill_definitions
from pyrag.a2a.client import _parts_text, extract_text
from pyrag.a2a.peers import add_peer, list_peers, remove_peer, resolve_peer
from pyrag.a2a.server import A2AServer
from pyrag.models import A2AMessage


def _msg(text: str) -> dict:
    return A2AMessage.text_message(text).model_dump(mode="json")


# ------------------------------------------------------------------ agent card
def test_agent_card_has_required_fields():
    card = build_agent_card("http://localhost:9100")
    assert card.protocol_version == "0.3.0"
    assert card.url == "http://localhost:9100"
    assert card.capabilities.streaming is True
    assert {s.id for s in card.skills} >= {"rag-qa", "agentic-task", "orchestration"}
    assert all(s.description and s.examples for s in skill_definitions())


# ------------------------------------------------------------------- dispatch
def test_message_send_completes_task():
    srv = A2AServer(handler=lambda text, ctx: f"eco: {text}")
    res = srv.handle_rpc({"jsonrpc": "2.0", "id": 1, "method": "message/send",
                          "params": {"message": _msg("ola")}})
    assert res["error"] is None
    task = res["result"]
    assert task["status"]["state"] == "completed"
    assert task["artifacts"][0]["parts"][0]["text"] == "eco: ola"
    assert task["id"] in srv.tasks


def test_task_get_and_cancel():
    srv = A2AServer(handler=lambda t, c: "ok")
    task = srv.handle_rpc({"jsonrpc": "2.0", "id": 1, "method": "message/send",
                           "params": {"message": _msg("x")}})["result"]
    got = srv.handle_rpc({"jsonrpc": "2.0", "id": 2, "method": "tasks/get",
                          "params": {"id": task["id"]}})["result"]
    assert got["id"] == task["id"]
    canceled = srv.handle_rpc({"jsonrpc": "2.0", "id": 3, "method": "tasks/cancel",
                               "params": {"id": task["id"]}})["result"]
    assert canceled["status"]["state"] in ("canceled", "completed")


def test_task_list():
    srv = A2AServer(handler=lambda t, c: "ok")
    srv.handle_rpc({"jsonrpc": "2.0", "id": 1, "method": "message/send",
                    "params": {"message": _msg("a")}})
    res = srv.handle_rpc({"jsonrpc": "2.0", "id": 2, "method": "tasks/list", "params": {}})
    assert len(res["result"]["tasks"]) == 1


def test_handler_failure_marks_task_failed():
    def boom(text, ctx):
        raise RuntimeError("falhou")

    res = A2AServer(handler=boom).handle_rpc({
        "jsonrpc": "2.0", "id": 1, "method": "message/send", "params": {"message": _msg("x")}})
    assert res["result"]["status"]["state"] == "failed"
    assert "falhou" in res["result"]["status"]["message"]["parts"][0]["text"]


def test_context_id_is_preserved():
    srv = A2AServer(handler=lambda t, c: "ok")
    res = srv.handle_rpc({"jsonrpc": "2.0", "id": 1, "method": "message/send", "params": {
        "message": _msg("x"), "configuration": {"contextId": "ctx-123"}}})
    assert res["result"]["contextId"] == "ctx-123"


# ----------------------------------------------------------------- jsonrpc 2.0
def test_invalid_jsonrpc_version():
    res = A2AServer().handle_rpc({"jsonrpc": "1.0", "id": 9, "method": "message/send"})
    assert res["error"]["code"] == -32600


def test_method_not_found():
    res = A2AServer().handle_rpc({"jsonrpc": "2.0", "id": 9, "method": "nao/existe"})
    assert res["error"]["code"] == -32601


def test_missing_params_is_invalid_params():
    res = A2AServer().handle_rpc({"jsonrpc": "2.0", "id": 1, "method": "message/send", "params": {}})
    assert res["error"]["code"] == -32602


def test_task_not_found():
    res = A2AServer().handle_rpc({"jsonrpc": "2.0", "id": 1, "method": "tasks/get",
                                  "params": {"id": "inexistente"}})
    assert "Task nao encontrada" in res["error"]["message"]


def test_rpc_response_is_json_serializable():
    res = A2AServer(handler=lambda t, c: "ok").handle_rpc({
        "jsonrpc": "2.0", "id": 1, "method": "message/send", "params": {"message": _msg("y")}})
    assert json.loads(json.dumps(res))["jsonrpc"] == "2.0"


# ------------------------------------------------------------------ extracao
def test_extract_text_from_artifacts_and_status():
    task = {"artifacts": [{"parts": [{"kind": "text", "text": "resultado"}]}]}
    assert extract_text(task) == "resultado"
    status = {"status": {"message": {"parts": [{"kind": "text", "text": "do status"}]}}}
    assert extract_text(status) == "do status"
    assert extract_text(None) == ""
    assert extract_text("plain") == "plain"


def test_parts_text_handles_data_parts():
    out = _parts_text([{"kind": "data", "data": {"a": 1}}, "texto", {"kind": "text", "text": "t"}])
    assert '"a": 1' in out and "texto" in out and "t" in out


# ---------------------------------------------------------------------- peers
def test_peer_skill_extraction():
    from pyrag.a2a.peers import _skill_ids

    card = {"skills": [{"id": "rag-qa", "name": "RAG"}, {"name": "so-nome"}]}
    assert _skill_ids(card) == ["rag-qa", "so-nome"]
    assert _skill_ids({}) == []


def test_peers_crud(tmp_path, monkeypatch):
    from pyrag.config import reset_settings_cache

    f = tmp_path / "peers.json"
    monkeypatch.setenv("PYRAG_A2A_PEERS_FILE", str(f))
    reset_settings_cache()
    try:
        assert list_peers() == []
        peer = add_peer("financeiro", "http://localhost:9200/", skills=["contas"])
        assert peer.url == "http://localhost:9200"
        assert resolve_peer("financeiro").url == "http://localhost:9200"
        assert resolve_peer("http://localhost:9200").name == "financeiro"
        assert resolve_peer("finan").name == "financeiro"
        assert resolve_peer("nao-existe") is None
        add_peer("vendas", "http://localhost:9100", skills=["crm"])
        assert len(list_peers()) == 2
        assert remove_peer("vendas") is True
        assert remove_peer("vendas") is False
        assert len(list_peers()) == 1
    finally:
        reset_settings_cache()