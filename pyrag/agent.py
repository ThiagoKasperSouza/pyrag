"""Agente ReAct com ferramentas: decide usar RAG, calculadora, web, SQL ou A2A."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from .config import get_settings
from .llm import effective_provider, get_llm_or_fallback
from .tools import get_tools

log = logging.getLogger(__name__)

AGENT_SYSTEM = """Voce e um agente autonomo que resolve tarefas usando ferramentas.

Ferramentas disponiveis:
- rag_search: busca nos documentos internos (fonte de verdade do projeto).
- calculator: calculos numericos exatos.
- web_search / fetch_url: pesquisa e leitura de paginas web.
- sql_query: consultas somente-leitura em um banco SQLite local.
- read_file / list_documents: leitura de arquivos locais.
- a2a_delegate: delega a um agente remoto via A2A (use quando for outro dominio).
- a2a_list_agents: lista agentes remotos disponiveis.
- finish: encerra a tarefa com resumo e resultado.

Regras:
1. Planeje em silencio; chame ferramentas na ordem correta.
2. NUNCA invente resultado de ferramenta. Use apenas os valores retornados.
3. Se faltar contexto, chame rag_search antes de responder.
4. Delegue via a2a_delegate apenas quando outro agente for mesmo especialista no assunto.
5. Ao terminar, chame finish com o resultado final em Portuguese do Brasil."""


def build_tool_agent(reasoner: bool = False, max_iterations: int | None = None):
    """create_agent (tool-calling nativo) quando disponivel; senao ReAct manual."""
    s = get_settings()
    tools = get_tools()
    max_iterations = max_iterations or s.agent_max_iterations

    try:
        from langchain.agents import create_agent

        return create_agent(get_llm_or_fallback(reasoner=reasoner), tools, prompt=AGENT_SYSTEM)
    except Exception as exc:
        log.info("create_agent indisponivel (%s); usando ReAct manual.", exc)
        return ReactAgent(tools=tools, reasoner=reasoner, max_iterations=max_iterations)


class ReactAgent:
    """ReAct simples e previsivel: funciona em qualquer provider, mesmo sem tool-calling."""

    name = "react-agent"

    def __init__(self, tools: list[Any], reasoner: bool = False, max_iterations: int | None = None):
        self.tools = {t.name: t for t in tools}
        self.tool_list = tools
        self.reasoner = reasoner
        self.max_iterations = max_iterations or get_settings().agent_max_iterations
        self._llm: BaseChatModel | None = None

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = get_llm_or_fallback(reasoner=self.reasoner)
        return self._llm

    @llm.setter
    def llm(self, value: BaseChatModel) -> None:
        self._llm = value

    def _format_tools(self) -> str:
        rows = []
        for t in self.tool_list:
            args = ", ".join("%s: %s" % (k, v.get("type", "string")) for k, v in t.args.items())
            rows.append("- %s(%s) :: %s" % (t.name, args, t.description))
        return "\n".join(rows)

    @staticmethod
    def _parse_action(text: str) -> dict | None:
        start, end = text.find("ACTION:"), text.rfind("ACTION:")
        if start == -1:
            return None
        blob = text[start + len("ACTION:"):]
        if end > start:
            blob = text[start + len("ACTION:"):end]
        blob = blob.replace("```json", "").replace("```", "").split("\nFINAL")[0].strip()
        candidates = [blob] + [l.strip() for l in blob.splitlines() if l.strip().startswith("{")]
        for cand in candidates:
            try:
                data = json.loads(cand)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and "tool" in data:
                return data
        return None

    def _run_tool(self, action: dict) -> str:
        name = action["tool"]
        tool = self.tools.get(name)
        if tool is None:
            return f"ERRO: ferramenta '{name}' inexistente. Disponiveis: {list(self.tools)}"
        args = action.get("args") or action.get("arguments") or {}
        if not isinstance(args, dict):
            args = {"input": args}
        try:
            return str(tool.invoke(args))[:6000]
        except Exception as exc:
            return f"ERRO executando {name}: {exc}"

    def run(self, task: str, history: list[BaseMessage] | None = None) -> dict[str, Any]:
        prompt = (
            f"{AGENT_SYSTEM}\n\nFerramentas:\n{self._format_tools()}\n\n"
            "Formato de resposta (use exatamente):\n"
            'PENSE: <raciocinio curto>\nACTION: {"tool": "nome", "args": {...}}\n'
            "ou, quando concluir:\nFINAL: <resposta final>\n\n"
            f"Tarefa: {task}"
        )
        messages: list[BaseMessage] = [SystemMessage(content=prompt)]
        if history:
            messages += history[-4:]
        transcript: list[dict[str, str]] = []
        seen: dict[str, int] = {}          # detecta loops de modelos pequenos
        last_text = ""
        no_action_streak = 0

        for step in range(self.max_iterations):
            text = str(getattr(self.llm.invoke(messages), "content", ""))
            transcript.append({"step": str(step), "thought": text[:1200]})
            log.info("Agente passo %d: %s", step, text[:180].replace("\n", " "))

            # Saida identica a anterior: provavelmente travado, forcamos conclusao.
            if text.strip() and text.strip() == last_text.strip():
                no_action_streak += 1
                messages.append(HumanMessage(
                    content="Voce repetiu a mesma resposta. Encerre agora chamando a ferramenta "
                            "finish com o resultado que ja tem nas observacoes."))
                messages.append(AIMessage(content=text))
                if no_action_streak >= 2:
                    prev = _last_observation(transcript)
                    return _forced_finish(
                        transcript, step + 1,
                        "Agente travado em repeticao; encerrado. "
                        f"Ultimo resultado disponivel: {prev[:400] or 'n/d'}")
                last_text = text
                continue
            last_text = text
            no_action_streak = 0
            messages.append(AIMessage(content=text))

            action = self._parse_action(text)

            if action is None:
                final = self._parse_final(text)
                if final:
                    return _forced_finish(transcript, step + 1, final)
                no_action_streak += 1
                messages.append(HumanMessage(
                    content="Responda no formato definido: PENSE + ACTION: {...} ou FINAL: <resposta>. "
                            "Use rag_search se precisar de contexto dos documentos."))
                if no_action_streak >= 3:
                    return _forced_finish(transcript, step + 1,
                                          "Agente nao produziu acoes validas; encerrado.")
                continue

            # Loop de mesma tool com mesmos argumentos?
            key = json.dumps([action["tool"], action.get("args", action.get("arguments"))],
                             sort_keys=True, default=str)
            seen[key] = seen.get(key, 0) + 1
            if seen[key] >= 2:
                prev = _last_observation(transcript)
                return _forced_finish(
                    transcript, step + 1,
                    f"Acao repetida ({action['tool']}) sem avanco. "
                    f"Ultimo resultado da ferramenta: {prev[:300] or 'n/d'}")

            obs = self._run_tool(action)
            transcript.append({"step": str(step), "tool": action["tool"], "observation": obs[:2000]})
            if action["tool"] == "finish":
                return {"output": obs, "steps": transcript, "provider": effective_provider(),
                        "iterations": step + 1}
            messages.append(HumanMessage(
                content=f"OBSERVACAO da ferramenta {action['tool']}:\n{obs}\n\nContinue o raciocinio."))

        return {"output": "Limite de iteracoes atingido. Ultima observacao:\n"
                         + _last_observation(transcript),
                "steps": transcript, "provider": effective_provider(),
                "iterations": self.max_iterations}

    @staticmethod
    def _parse_final(text: str) -> str | None:
        """Extrai a linha FINAL: quando o modelo ja responde mas esqueceu o ACTION."""
        if "FINAL:" not in text:
            return None
        after = text.split("FINAL:")[-1].strip()
        return after or None


def _last_observation(transcript: list[dict[str, str]]) -> str:
    for entry in reversed(transcript):
        if "observation" in entry:
            return entry["observation"]
    return ""


def _forced_finish(transcript: list[dict[str, str]], iterations: int, output: str) -> dict[str, Any]:
    transcript.append({"step": str(iterations), "observation": output[:2000]})
    return {"output": output, "steps": transcript, "provider": effective_provider(),
            "iterations": iterations, "forced": True}


def run_agent(task: str, reasoner: bool = False,
              history: list[BaseMessage] | None = None) -> dict[str, Any]:
    """Atalho: cria o agente e executa uma tarefa."""
    return build_tool_agent(reasoner=reasoner).run(task, history=history)