"""Testes de API, agente ReAct, memoria e ingestao (sem rede / sem LLM)."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from pyrag.agent import ReactAgent, build_tool_agent
from pyrag.api import create_app
from pyrag.ingest import ingest, load_documents, split_documents


# ------------------------------------------------------------------------ API
def test_health_and_endpoints():
    with TestClient(create_app()) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/tools").status_code == 200
        assert client.get("/sessions").status_code == 200
        assert client.get("/peers").status_code == 200


def test_agent_card_served_by_api():
    with TestClient(create_app()) as client:
        card = client.get("/.well-known/agent.json").json()
        assert card["protocolVersion"] == "0.3.0"
        assert card["skills"]
        assert client.get("/a2a/health").json()["status"] == "ok"


def test_a2a_rpc_over_http():
    app = create_app()
    app.state.a2a.handler = lambda text, ctx: f"resposta: {text}"
    with TestClient(app) as client:
        payload = {"jsonrpc": "2.0", "id": 1, "method": "message/send",
                   "params": {"message": {"role": "user", "parts": [{"kind": "text", "text": "oi"}]}}}
        res = client.post("/a2a", json=payload).json()
        assert res["result"]["status"]["state"] == "completed"
        assert "resposta: oi" in res["result"]["artifacts"][0]["parts"][0]["text"]


def test_a2a_stream_over_sse():
    app = create_app()
    app.state.a2a.handler = lambda text, ctx: "stream ok"
    with TestClient(app) as client:
        with client.stream("POST", "/a2a", json={
            "jsonrpc": "2.0", "id": 7, "method": "message/stream",
            "params": {"message": {"role": "user", "parts": [{"kind": "text", "text": "z"}]}}}) as r:
            body = "".join(r.iter_text())
    assert body.count("data:") >= 3
    assert body.strip().endswith("[DONE]")


def test_search_and_ask_routes():
    with TestClient(create_app()) as client:
        assert client.post("/search", json={"query": "a2a", "k": 3}).status_code == 200
        res = client.post("/ask", json={"question": "o que e a2a?", "session": "teste"}).json()
        assert "answer" in res and "citations" in res


def test_index_route(tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text("O sistema usa Chroma e BM25 juntos.", encoding="utf-8")
    with TestClient(create_app()) as client:
        res = client.post("/ingest", json={"paths": [str(doc)]}).json()
        assert res["indexed_chunks"] >= 1
        found = client.post("/search", json={"query": "BM25", "k": 2}).json()
        assert found["results"]


# --------------------------------------------------------------------- agente
def _scripted_llm(script: list[str]):
    from langchain_core.messages import AIMessage

    class FakeLLM:
        def __init__(self):
            self.i = 0

        def invoke(self, _messages):
            self.i += 1
            return AIMessage(content=script[min(self.i - 1, len(script) - 1)])

    return FakeLLM()


def test_react_agent_runs_tools_and_finishes():
    from pyrag.tools import calculator, finish, rag_search

    agent = ReactAgent(tools=[calculator, rag_search, finish], max_iterations=4)
    agent.llm = _scripted_llm([
        'PENSE: calcular\nACTION: {"tool": "calculator", "args": {"expression": "6*7"}}',
        'PENSE: pronto\nACTION: {"tool": "finish", "args": {"task_summary": "calc", "result": "42"}}',
    ])
    out = agent.run("calcule 6*7")
    assert "42" in out["output"]
    assert any(s.get("tool") == "calculator" for s in out["steps"])


# -------------------------------------------------------------------- memoria
def test_memory_roundtrip(tmp_path, monkeypatch):
    from pyrag.config import reset_settings_cache
    from pyrag.memory import append_turn, clear_session, list_sessions, load_session

    monkeypatch.setenv("PYRAG_DATA_DIR", str(tmp_path))
    reset_settings_cache()
    try:
        clear_session("s1")
        append_turn("s1", "pergunta 1", "resposta 1")
        append_turn("s1", "pergunta 2", "resposta 2")
        msgs = load_session("s1")
        assert len(msgs) == 4
        assert msgs[0].content == "pergunta 1"
        assert "s1" in list_sessions()
        assert clear_session("s1") is True
        assert load_session("s1") == []
    finally:
        reset_settings_cache()


def test_memory_trim_keeps_last_turns(tmp_path, monkeypatch):
    from pyrag.config import reset_settings_cache
    from pyrag.memory import append_turn, load_session, session_path

    monkeypatch.setenv("PYRAG_DATA_DIR", str(tmp_path))
    reset_settings_cache()
    try:
        for i in range(6):
            append_turn("trim", f"p{i}", f"r{i}", max_turns=2)
        lines = [l for l in session_path("trim").read_text(encoding="utf-8").splitlines() if l]
        assert len(lines) == 2                      # apenas 2 turnos no arquivo
        msgs = load_session("trim", 2)              # 2 turnos = 4 mensagens
        assert len(msgs) == 4
        assert msgs[0].content == "p4"
    finally:
        reset_settings_cache()


def test_rewrite_query_without_history_is_passthrough():
    from pyrag.rag import RAGPipeline

    assert RAGPipeline().rewrite_query("pergunta") == "pergunta"


# -------------------------------------------------------------------- ingestao
def test_load_and_split_markdown(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("palavra repetida. " * 200, encoding="utf-8")
    docs = load_documents([f], collection="teste")
    assert docs and docs[0].metadata["collection"] == "teste"
    assert docs[0].metadata["path"].endswith("doc.md")
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(c.metadata["chunk"] >= 0 for c in chunks)


def test_load_documents_ignores_missing_path(tmp_path):
    assert load_documents([tmp_path / "nao_existe.txt"]) == []


def test_load_csv(tmp_path):
    f = tmp_path / "t.csv"
    f.write_text("a,b\n1,2\n", encoding="utf-8")
    docs = load_documents([f])
    assert docs and "1 | 2" in docs[0].page_content


def test_ingest_directory(tmp_path):
    (tmp_path / "a.txt").write_text("conteudo A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("conteudo B", encoding="utf-8")
    assert len(ingest([tmp_path])) == 2


# ------------------------------------------------------------------ extra bits
def test_llm_fallback_is_extractive():
    from pyrag.llm import ExtractiveFallbackLLM

    llm = ExtractiveFallbackLLM(max_tokens=300)
    out = llm.invoke("Contexto: o teste passou com sucesso. tudo certo.")
    assert "EXTRATIVA" in out.content


def test_hashing_embeddings_deterministic():
    from pyrag.llm import HashingEmbeddings

    emb = HashingEmbeddings()
    assert emb.embed_query("ola") == emb.embed_query("ola")
    assert len(emb.embed_query("ola")) == 384
    assert emb.embed_query("ola") != emb.embed_query("tchau")


def test_effective_provider_is_none_in_tests(settings):
    from pyrag.llm import effective_provider

    assert effective_provider() == "none"


def test_agent_card_json_serializable():
    from pyrag.a2a.card import build_agent_card

    payload = json.loads(json.dumps(build_agent_card("http://x").model_dump(mode="json")))
    assert payload["skills"][0]["id"]


def test_react_agent_handles_unknown_tool():
    from pyrag.tools import calculator, finish

    class UnknownToolLLM:
        def __init__(self):
            self.i = 0

        def invoke(self, _messages):
            from langchain_core.messages import AIMessage

            self.i += 1
            if self.i == 1:
                return AIMessage(content='ACTION: {"tool": "inexistente", "args": {}}')
            return AIMessage(content='ACTION: {"tool": "finish", "args": {"task_summary": "s", "result": "fim"}}')

    agent = ReactAgent(tools=[calculator, finish], max_iterations=4)
    agent.llm = UnknownToolLLM()
    out = agent.run("teste")
    assert any("inexistente" in str(s.get("observation", "")) for s in out["steps"])
    assert "TAREFA CONCLUIDA" in out["output"]


def test_react_agent_respects_iteration_limit():
    from pyrag.tools import finish

    agent = ReactAgent(tools=[finish], max_iterations=2)
    agent.llm = _scripted_llm(['ACTION: {"tool": "calculator", "args": {"expression": "1+1"}}'])
    out = agent.run("loop")
    assert out["iterations"] == 2
    assert "Limite de iteracoes" in out["output"]


def test_build_tool_agent_returns_object():
    agent = build_tool_agent()
    assert hasattr(agent, "invoke") or hasattr(agent, "run")


def test_react_agent_detects_identical_output_loop():
    from pyrag.tools import calculator, finish

    class RepeatingLLM:
        def invoke(self, _messages):
            from langchain_core.messages import AIMessage

            return AIMessage(content='ACTION: {"tool": "calculator", "args": {"expression": "2+2"}}')

    agent = ReactAgent(tools=[calculator, finish], max_iterations=6)
    agent.llm = RepeatingLLM()
    out = agent.run("loop de saida")
    assert out.get("forced") is True
    assert out["iterations"] <= 6
    assert "4" in out["output"]


def test_react_agent_detects_repeated_action():
    from pyrag.tools import calculator, finish

    class AlternatingLLM:
        def __init__(self):
            self.i = 0

        def invoke(self, _messages):
            from langchain_core.messages import AIMessage

            self.i += 1
            if self.i % 2:
                return AIMessage(content="pensando...")
            return AIMessage(content='ACTION: {"tool": "calculator", "args": {"expression": "2+2"}}')

    agent = ReactAgent(tools=[calculator, finish], max_iterations=6)
    agent.llm = AlternatingLLM()
    out = agent.run("loop de acao")
    assert out.get("forced") is True
    assert "calculator" in out["output"]


def test_react_agent_accepts_final_without_action():
    from pyrag.tools import finish

    class FinalOnlyLLM:
        def invoke(self, _messages):
            from langchain_core.messages import AIMessage

            return AIMessage(content="PENSE: ja sei\nFINAL: a resposta e 7")

    agent = ReactAgent(tools=[finish], max_iterations=3)
    agent.llm = FinalOnlyLLM()
    out = agent.run("responda")
    assert "7" in out["output"]


def test_parse_action_variants():
    assert ReactAgent._parse_action('ACTION: ```json\n{"tool": "x", "args": {"a": 1}}\n```') == {
        "tool": "x", "args": {"a": 1}}
    assert ReactAgent._parse_action("sem acao") is None