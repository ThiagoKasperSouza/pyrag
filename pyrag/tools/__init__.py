"""Ferramentas agenticas locais (calculadora, RAG, web, SQL, arquivos, A2A)."""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.tools import tool

from ..config import get_settings

log = logging.getLogger(__name__)


@tool
def calculator(expression: str) -> str:
    """Avalia uma expressao aritmetica com seguranca (ex: '2*(3+4) - 10/5')."""
    import ast
    import operator

    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
           ast.FloorDiv: operator.floordiv}

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            left, right = _eval(node.left), _eval(node.right)
            if type(node.op) in (ast.Pow, ast.Div, ast.FloorDiv, ast.Mod) and abs(right) > 1e6:
                raise ValueError("operador com valor grande demais")
            return ops[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = _eval(node.operand)
            return v if isinstance(node.op, ast.UAdd) else -v
        raise ValueError(f"expressao nao permitida: {type(node).__name__}")

    try:
        return str(round(_eval(ast.parse(expression, mode="eval")), 10))
    except Exception as exc:
        return f"ERRO ao calcular: {exc}"


@tool
def rag_search(query: str, k: int = 5) -> str:
    """Busca semantica nos documentos indexados e devolve os melhores trechos com a fonte.

    Use sempre que a resposta estiver nos documentos internos.
    """
    from ..rag import RAGPipeline

    try:
        chunks = RAGPipeline().retrieve(query, k=k)
    except Exception as exc:
        return f"ERRO na busca: {exc}"
    if not chunks:
        return "Nenhum trecho relevante encontrado."
    return RAGPipeline.build_context(chunks)


@tool
def read_file(path: str, max_chars: int = 4000) -> str:
    """Le um arquivo de texto do disco e devolve seu conteudo (limitado)."""
    from pathlib import Path

    p = Path(path).expanduser()
    if not p.is_file():
        return f"ERRO: arquivo nao encontrado: {p}"
    try:
        return p.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception as exc:
        return f"ERRO ao ler arquivo: {exc}"


@tool
def list_documents(path: str = ".") -> str:
    """Lista arquivos de um diretorio, com tamanho em KB."""
    from pathlib import Path

    p = Path(path).expanduser()
    if not p.is_dir():
        return f"ERRO: diretorio nao encontrado: {p}"
    rows = [f"{f.name}\t{f.stat().st_size / 1024:.1f} KB" for f in sorted(p.iterdir())[:200]]
    return "\n".join(rows) or "diretorio vazio"


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Pesquisa na web (DuckDuckGo, sem API key) e devolve titulo, url e trecho."""
    if not get_settings().enable_web_search:
        return "Busca web desabilitada (PYRAG_ENABLE_WEB_SEARCH=false)."
    try:
        from duckduckgo_search import DDGS

        with DDGS(timeout=12) as ddgs:
            hits = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:
        return f"ERRO na busca web: {exc}"
    if not hits:
        return "Nenhum resultado encontrado."
    return "\n\n".join(
        f"{i}. {h.get('title', '')}\n   {h.get('href', '')}\n   {(h.get('body') or '')[:300]}"
        for i, h in enumerate(hits, 1)
    )


@tool
def fetch_url(url: str, max_chars: int = 6000) -> str:
    """Baixa uma pagina web e devolve o texto extraido (sem HTML)."""
    try:
        import httpx
        from bs4 import BeautifulSoup

        r = httpx.get(url, timeout=20, follow_redirects=True,
                      headers={"User-Agent": "Mozilla/5.0 pyrag/0.1"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return re.sub(r"\n{3,}", "\n\n", soup.get_text("\n", strip=True))[:max_chars]
    except Exception as exc:
        return f"ERRO ao baixar a pagina: {exc}"


@tool
def sql_query(sql: str, limit: int = 50) -> str:
    """Executa uma query SQL somente-leitura no banco SQLite local e devolve a tabela.

    Use para responder perguntas sobre dados estruturados.
    """
    s = get_settings()
    if not s.enable_sql:
        return "Ferramenta SQL desabilitada."
    forbidden = ("insert", "update", "delete", "drop", "alter", "attach", "pragma", "create")
    low = sql.strip().lower()
    if not (low.startswith("select") or low.startswith("with")):
        return "ERRO: apenas SELECT/WITH e permitido."
    if any(f" {w} " in low or low.endswith(f" {w}") for w in forbidden):
        return "ERRO: comando nao permitido (somente leitura)."
    if ";" in low.rstrip(";"):
        return "ERRO: multiplos statements nao permitidos."
    try:
        import sqlite3

        con = sqlite3.connect(s.sql_db_path)
        con.row_factory = sqlite3.Row
        cur = con.execute(sql)
        rows = cur.fetchmany(limit)
        cols = [d[0] for d in (cur.description or [])]
        con.close()
        if not cols:
            return "Query executada. Nenhuma linha retornada."
        body = "\n".join(" | ".join(str(r[c]) for c in cols) for r in rows)
        return f"{' | '.join(cols)}\n{'-' * 60}\n{body}\n({len(rows)} linhas)"
    except Exception as exc:
        return f"ERRO SQL: {exc}"


@tool
def a2a_delegate(agent: str, task: str) -> str:
    """Delega uma tarefa a um agente remoto via A2A (Agent2Agent) e devolve a resposta.

    Use `agent` com o nome ou a URL de um peer registrado (veja `pyrag peers list`).
    """
    from ..a2a.client import A2AClient
    from ..a2a.peers import resolve_peer

    try:
        peer = resolve_peer(agent)
        if peer is None:
            return f"ERRO: agente A2A '{agent}' nao encontrado. Use `pyrag peers list`."
        return A2AClient(peer.url).send_text(task)
    except Exception as exc:
        return f"ERRO na delegacao A2A: {exc}"


@tool
def a2a_list_agents() -> str:
    """Lista os agentes remotos A2A disponiveis (nome, url e skills)."""
    from ..a2a.peers import list_peers

    peers = list_peers()
    if not peers:
        return "Nenhum agente remoto cadastrado. Adicione com `pyrag peers add <nome> <url>`."
    return "\n".join(f"- {p.name}: {p.url} (skills: {', '.join(p.skills) or 'n/d'})" for p in peers)


@tool
def finish(task_summary: str, result: str) -> str:
    """Encerra a tarefa: informe o resumo do que foi feito e o resultado final.

    Chame esta tool quando a tarefa estiver concluida.
    """
    return f"TAREFA CONCLUIDA\nResumo: {task_summary}\nResultado: {result}"


BUILTIN_TOOLS: list[Any] = [
    calculator, rag_search, read_file, list_documents,
    web_search, fetch_url, sql_query, a2a_delegate, a2a_list_agents, finish,
]


def get_tools() -> list[Any]:
    """Devolve as tools habilitadas conforme a configuracao."""
    s = get_settings()
    out: list[Any] = []
    for t in BUILTIN_TOOLS:
        if t.name == "web_search" and not s.enable_web_search:
            continue
        if t.name == "sql_query" and not s.enable_sql:
            continue
        if t.name == "calculator" and not s.enable_calc:
            continue
        out.append(t)
    return out


__all__ = ["get_tools", "BUILTIN_TOOLS", "calculator", "rag_search", "read_file",
           "list_documents", "web_search", "fetch_url", "sql_query", "a2a_delegate",
           "a2a_list_agents", "finish"]