#!/usr/bin/env python3
"""Valida o agente ReAct com ferramentas reais (calculator, rag_search, sql)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pyrag.agent import ReactAgent, build_tool_agent  # noqa: E402
from pyrag.llm import effective_provider  # noqa: E402
from pyrag.tools import calculator, finish, list_documents, rag_search, sql_query  # noqa: E402

provider = effective_provider()
print(f"provider = {provider}\n")

print("=== 1. Agente ReAct com LLM real ===")
print(f"build_tool_agent -> {type(build_tool_agent()).__name__}")

agent = ReactAgent(tools=[calculator, rag_search, sql_query, list_documents, finish],
                   max_iterations=5)
res = agent.run("Use a ferramenta calculator para calcular 17*23. Depois chame finish "
                "com o resultado da conta.")
print(f"iteracoes={res['iterations']} provider={res['provider']}")
print("--- trace ---")
for step in res["steps"]:
    if step.get("tool"):
        print(f"  step {step['step']}: TOOL {step['tool']} -> "
              f"{step['observation'][:110]}".replace("\n", " "))
    else:
        print(f"  step {step['step']}: {step.get('thought', '')[:110]}".replace("\n", " "))
print("--- saida ---")
print(str(res["output"])[:400])

print("\n=== 2. Agente usando rag_search nos documentos ===")
agent2 = ReactAgent(tools=[rag_search, finish], max_iterations=5)
res2 = agent2.run("Pesquise nos documentos o que e a comunicacao A2A usando rag_search "
                  "e depois chame finish.")
print(f"iteracoes={res2['iterations']}")
for step in res2["steps"]:
    if step.get("tool"):
        print(f"  TOOL {step['tool']} -> {step['observation'][:130]}".replace("\n", " "))
print(str(res2["output"])[:400])

print("\n=== 3. Ferramenta a2a_delegate com peer registered ===")
from pyrag.a2a.peers import list_peers  # noqa: E402
from pyrag.tools import a2a_delegate, a2a_list_agents  # noqa: E402

print("peers:", [p.name for p in list_peers()])
print("a2a_list_agents ->", a2a_list_agents.invoke({})[:150])
print("a2a_delegate(inexistente) ->", a2a_delegate.invoke({"agent": "nao-existe", "task": "oi"})[:150])

print("\nOK")