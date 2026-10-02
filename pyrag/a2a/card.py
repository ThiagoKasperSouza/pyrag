"""Agent Card A2A: descricao de skills para descoberta por outros agentes."""

from __future__ import annotations

from ..config import get_settings
from ..models import AgentCapabilities, AgentCard, AgentSkill


def skill_definitions() -> list[AgentSkill]:
    return [
        AgentSkill(
            id="rag-qa",
            name="Resposta sobre documentos",
            description="Responde perguntas com base nos documentos indexados, citando as fontes.",
            tags=["rag", "documentos", "interno"],
            examples=["Qual o prazo de entrega do contrato?", "Resuma a politica de报销 de custos."],
        ),
        AgentSkill(
            id="research-web",
            name="Pesquisa web",
            description="Pesquisa na web e sintetiza informacoes recentes com links.",
            tags=["web", "pesquisa"],
            examples=["Quais as novidades do LangChain em 2026?"],
        ),
        AgentSkill(
            id="data-sql",
            name="Analise de dados",
            description="Consulta o banco SQLite local em modo somente-leitura e agrega os resultados.",
            tags=["sql", "dados"],
            examples=["Quantas vendas houve por mes?"],
        ),
        AgentSkill(
            id="agentic-task",
            name="Tarefa autonoma com ferramentas",
            description="Planeja e executa tarefas multiplas usando calculadora, arquivos e delegacao A2A.",
            tags=["agente", "tools"],
            examples=["Compare o custo anual de 3 planos e recomende um."],
        ),
        AgentSkill(
            id="orchestration",
            name="Orquestracao A2A",
            description="Recebe subtarefas de outros agentes e delega specialists quando necessario.",
            tags=["a2a", "coordenacao"],
            examples=["Delega a analise financeira ao agente contábil."],
        ),
    ]


def build_agent_card(base_url: str | None = None) -> AgentCard:
    s = get_settings()
    return AgentCard(
        protocol_version="0.3.0",
        name=s.agent_name,
        description=s.agent_description,
        version=s.agent_version,
        url=(base_url or s.a2a_public_url).rstrip("/"),
        capabilities=AgentCapabilities(streaming=True, push_notifications=False,
                                      state_transition_history=True),
        skills=skill_definitions(),
        provider={"organization": "pyrag", "url": (base_url or s.a2a_public_url).rstrip("/")},
    )