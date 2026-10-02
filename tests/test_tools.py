"""Testes das ferramentas agenticas (sem rede)."""

from __future__ import annotations

import sqlite3

from pyrag.tools import (a2a_list_agents, calculator, finish, get_tools, list_documents,
                         rag_search, read_file, sql_query)


def test_calculator_basic():
    assert calculator.invoke({"expression": "2*(3+4)"}) == "14"


def test_calculator_rejects_code_execution():
    out = calculator.invoke({"expression": "__import__('os').system('ls')"})
    assert out.startswith("ERRO")


def test_calculator_reports_error():
    assert calculator.invoke({"expression": "1/0"}).startswith("ERRO")


def test_read_file(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("conteudo do arquivo", encoding="utf-8")
    assert "conteudo" in read_file.invoke({"path": str(f)})
    assert "ERRO" in read_file.invoke({"path": str(tmp_path / "nao_existe.txt")})


def test_list_documents(tmp_path):
    (tmp_path / "b.txt").write_text("x", encoding="utf-8")
    assert "b.txt" in list_documents.invoke({"path": str(tmp_path)})


def test_sql_rejects_write_statements(settings):
    assert "ERRO" in sql_query.invoke({"sql": "DELETE FROM users"})
    assert "ERRO" in sql_query.invoke({"sql": "DROP TABLE users"})
    assert "ERRO" in sql_query.invoke({"sql": "SELECT 1; SELECT 2"})


def test_sql_read_only_query(settings):
    con = sqlite3.connect(settings.sql_db_path)
    con.execute("CREATE TABLE IF NOT EXISTS teste (id INTEGER, nome TEXT)")
    con.execute("INSERT INTO teste VALUES (1, 'ana'), (2, 'bruno')")
    con.commit()
    con.close()
    out = sql_query.invoke({"sql": "SELECT nome FROM teste ORDER BY id"})
    assert "ana" in out and "bruno" in out


def test_rag_search_on_empty_index_returns_message():
    out = rag_search.invoke({"query": "inexistente", "k": 2})
    assert isinstance(out, str)


def test_a2a_list_agents_without_peers():
    out = a2a_list_agents.invoke({})
    assert isinstance(out, str)


def test_finish_tool():
    assert "TAREFA CONCLUIDA" in finish.invoke({"task_summary": "ok", "result": "42"})


def test_get_tools_respects_config(settings):
    tools = {t.name for t in get_tools()}
    assert "calculator" in tools and "rag_search" in tools and "finish" in tools
    assert "web_search" not in tools  # PYRAG_ENABLE_WEB_SEARCH=false no conftest