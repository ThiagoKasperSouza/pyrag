"""CLI do pyrag (Typer): indexar, perguntar, agentar, A2A e servir."""

from __future__ import annotations

import json
import logging
import sys

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .config import get_settings

app = typer.Typer(add_completion=False, help="pyrag: RAG + agentes + A2A com LLMs gratuitos")
rag_app = typer.Typer(help="Comandos de RAG")
peers_app = typer.Typer(help="Gerencia agentes remotos A2A")
a2a_app = typer.Typer(help="Operacoes A2A")
app.add_typer(rag_app, name="rag")
app.add_typer(peers_app, name="peers")
app.add_typer(a2a_app, name="a2a")

console = Console()


def answer_panel(result) -> Panel:
    body = result.answer
    if result.citations:
        refs = "\n".join(
            f"  [{c.index}] {c.source.path}" + (f" (chunk {c.source.chunk})" if c.source.chunk else "")
            for c in result.citations
        )
        body += f"\n\n{refs}"
    meta = (f"\n\n[dim]provedor={result.provider} modelo={result.model} "
            f"{result.elapsed_ms}ms chunks={len(result.chunks)}[/dim]")
    return Panel(Markdown(body + meta), title="Resposta", border_style="cyan")


# -------------------------------------------------------------------- indexacao
@app.command("index")
def index_cmd(
    paths: list[str] = typer.Argument(..., help="Arquivos, diretorios ou URLs a indexar."),
    collection: str = typer.Option("default", help="Nome da colecao."),
) -> None:
    """Indexa documentos no Chroma + BM25."""
    from .rag import RAGPipeline

    pipeline = RAGPipeline()
    n = pipeline.index_paths(paths, collection=collection)
    console.print(f"[green]{n}[/green] chunks indexados. Status: {pipeline.status()}")


@app.command("reset")
def reset_cmd(confirm: bool = typer.Option(False, "--yes", help="Nao pedir confirmacao.")) -> None:
    """Apaga o indice vetorial."""
    from .vectorstore import get_store

    if not confirm and not typer.confirm("Apagar todo o indice?"):
        raise typer.Abort()
    get_store().delete_collection()
    console.print("[yellow]Indice removido.[/yellow]")


@app.command("status")
def status_cmd() -> None:
    """Mostra status do indice e do provider."""
    from .a2a.peers import list_peers
    from .llm import effective_provider
    from .rag import RAGPipeline

    s, st = get_settings(), RAGPipeline().status()
    table = Table(title="pyrag")
    table.add_column("Campo"); table.add_column("Valor")
    for k, v in [("provider", effective_provider()), ("modelo chat", s.chat_model),
                 ("embeddings", s.embed_model), ("chunks", str(st["chunks"])),
                 ("bm25 docs", str(st["bm25_docs"])), ("chroma", st["chroma_dir"]),
                 ("a2a peers", str(len(list_peers()))), ("a2a url", s.a2a_public_url)]:
        table.add_row(k, str(v))
    console.print(table)


# --------------------------------------------------------------------------- RAG
@rag_app.command("ask")
def rag_ask(
    question: list[str] = typer.Argument(..., help="Pergunta."),
    session: str = typer.Option("default", help="Sessao de memoria."),
    k: int = typer.Option(0, help="Numero de chunks (0 = config)."),
) -> None:
    """Pergunta com RAG, exibindo fontes."""
    from .rag import RAGPipeline

    result = RAGPipeline().chat(" ".join(question), session=session, k=k or None)
    console.print(answer_panel(result))


@rag_app.command("search")
def rag_search_cmd(
    query: list[str] = typer.Argument(...),
    k: int = typer.Option(5),
    show_text: bool = typer.Option(False, "--text", help="Mostrar texto completo dos chunks."),
) -> None:
    """Busca hibrida sem chamar o LLM."""
    from .rag import RAGPipeline

    chunks = RAGPipeline().retrieve(" ".join(query), k=k)
    if not chunks:
        console.print("[yellow]Nenhum trecho encontrado.[/yellow]")
        return
    for i, c in enumerate(chunks, 1):
        console.print(f"[bold][{i}][/bold] {c.source.path} (chunk {c.source.chunk}) score={c.score:.3f}")
        if c.dense_score is not None:
            console.print(f"    [dim]denso={c.dense_score:.3f} lexical={c.lexical_score or 0:.3f}[/dim]")
        body = c.text if show_text else c.text[:400] + ("..." if len(c.text) > 400 else "")
        console.print(Markdown(body))
        console.print("")


@rag_app.command("chat")
def rag_chat(session: str = typer.Option("default")) -> None:
    """Sessao interativa de perguntas (com memoria)."""
    from .rag import RAGPipeline

    pipeline = RAGPipeline()
    console.print("[bold]pyrag chat[/bold] - digite 'sair' para encerrar.")
    while True:
        try:
            q = console.input("\n[cyan]voce>[/cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in ("sair", "exit", "quit"):
            break
        console.print(answer_panel(pipeline.chat(q, session=session)))

# --------------------------------------------------------------------------- A2A
@a2a_app.command("serve")
def a2a_serve(
    host: str = typer.Option("", help="Host (vazio = config)."),
    port: int = typer.Option(0, help="Porta (0 = config)."),
) -> None:
    """Sobe um servidor A2A dedicado."""
    from .a2a.server import run_a2a_server

    s = get_settings()
    console.print(f"[green]A2A em http://{host or s.a2a_bind_host}:{port or s.a2a_bind_port}[/green]")
    run_a2a_server(host or None, port or None)


@a2a_app.command("card")
def a2a_card(url: str = typer.Option("", help="Base URL publica (padrao = config).")) -> None:
    """Mostra o Agent Card local."""
    from .a2a.card import build_agent_card

    console.print_json(json.dumps(build_agent_card(url or None).model_dump(mode="json"), ensure_ascii=False))


@a2a_app.command("ask")
def a2a_ask(
    url: str = typer.Argument(..., help="URL base do agente remoto."),
    text: list[str] = typer.Argument(..., help="Mensagem para o agente remoto."),
    stream: bool = typer.Option(False, "--stream", help="Usar message/stream (SSE)."),
) -> None:
    """Envia uma mensagem para um agente remoto via A2A."""
    from .a2a.client import A2AClient
    from .models import A2AMessage

    client = A2AClient(url)
    msg = A2AMessage.text_message(" ".join(text))
    if stream:
        for chunk in client.stream_message(msg):
            console.print(chunk, end="")
        console.print()
    else:
        console.print(Markdown(client.send_text(msg.text())))


@peers_app.command("add")
def peers_add(name: str = typer.Argument(...), url: str = typer.Argument(...)) -> None:
    """Adiciona um agente remoto (descobre skills pelo Agent Card)."""
    from .a2a.peers import add_peer

    peer = add_peer(name, url)
    console.print(f"[green]{peer.name}[/green] -> {peer.url} skills={peer.skills}")


@peers_app.command("remove")
def peers_remove(name: str = typer.Argument(...)) -> None:
    """Remove um agente remoto."""
    from .a2a.peers import remove_peer

    console.print("[green]removido[/green]" if remove_peer(name) else "[yellow]nao encontrado[/yellow]")


@peers_app.command("list")
def peers_list() -> None:
    """Lista agentes remotos e testa conectividade."""
    from .a2a.client import A2AClient
    from .a2a.peers import list_peers

    peers = list_peers()
    if not peers:
        console.print("[yellow]Nenhum peer cadastrado.[/yellow]")
        return
    table = Table(title="A2A peers")
    table.add_column("nome"); table.add_column("url"); table.add_column("status")
    for p in peers:
        ping = A2AClient(p.url, timeout=5).ping()
        table.add_row(p.name, p.url, "online" if ping.get("online")
                      else f"offline ({str(ping.get('error'))[:40]})")
    console.print(table)


# --------------------------------------------------------------------- servidor
@app.command("serve")
def serve_cmd(
    host: str = typer.Option("", help="Host (vazio = config)."),
    port: int = typer.Option(0, help="Porta (0 = config)."),
    reload: bool = typer.Option(False, help="Recarregar ao mudar codigo."),
) -> None:
    """Sobe a API HTTP (REST + A2A)."""
    import uvicorn

    s = get_settings()
    console.print(f"[green]API em http://{host or s.api_host}:{port or s.api_port}[/green] | docs em /docs")
    uvicorn.run("pyrag.api:app", host=host or s.api_host, port=port or s.api_port,
                reload=reload, log_level=s.log_level.lower())


@app.command("demo")
def demo_cmd() -> None:
    """Demonstra as 3 camadas: RAG, ferramentas e A2A."""
    from .a2a.card import build_agent_card
    from .agent import run_agent
    from .rag import RAGPipeline

    console.rule("1. RAG (recuperacao com fontes)")
    for i, c in enumerate(RAGPipeline().retrieve("como o sistema funciona?", k=2), 1):
        console.print(f"[{i}] {c.source.path} score={c.score:.3f} :: {c.text[:200]}")

    console.rule("2. Agente com ferramentas")
    out = run_agent("Use a ferramenta calculator para calcular 17*23 e explique o resultado.")
    console.print(str(out.get("output", ""))[:1500])

    console.rule("3. A2A (Agent Card local)")
    card = build_agent_card().model_dump(mode="json")
    console.print(f"agente={card['name']} url={card['url']} skills={[s['id'] for s in card['skills']]}")


def main() -> None:  # pragma: no cover
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    app()


if __name__ == "__main__":  # pragma: no cover
    main()

# ----------------------------------------------------------------------- agente
@app.command("agent")
def agent_cmd(
    task: list[str] = typer.Argument(..., help="Tarefa para o agente executar."),
    reasoner: bool = typer.Option(False, "--reason", help="Usar modelo de raciocinio."),
    steps: bool = typer.Option(False, "--steps", help="Exibir trace de passos."),
) -> None:
    """Executa uma tarefa autonoma com ferramentas (ReAct)."""
    from .agent import run_agent

    with console.status("[cyan]agente pensando...[/cyan]"):
        result = run_agent(" ".join(task), reasoner=reasoner)
    output = result.get("output", "")
    console.print(Markdown(output) if isinstance(output, str) else str(output))
    if steps:
        console.rule("trace")
        console.print_json(json.dumps(result.get("steps", []), ensure_ascii=False))