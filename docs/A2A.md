# A2A (Agent2Agent)

Subconjunto da spec A2A: **Agent Card** para descoberta + **JSON-RPC 2.0** para tarefas.

## 1. Expor um agente

```bash
pyrag a2a serve                      # porta 9100
# ou junto com a API REST
pyrag serve                          # porta 8000 (inclui /a2a)
```

Agent Card (descoberta):

```bash
curl http://localhost:9100/.well-known/agent.json
```

```json
{
  "protocolVersion": "0.3.0",
  "name": "pyrag-orchestrator",
  "url": "http://localhost:9100",
  "skills": [{"id": "rag-qa", "name": "Resposta sobre documentos", "...": "..."}]
}
```

## 2. Registrar um peer

```bash
pyrag peers add agente-financeiro http://localhost:9200
pyrag peers list          # testa conectividade
```

O `add` lê o Agent Card remoto e guarda as skills automaticamente.

## 3. Delegar uma tarefa

```bash
pyrag a2a ask http://localhost:9200 "gere o relatorio mensal"
pyrag a2a ask http://localhost:9200 "…" --stream      # SSE
```

Via API do agente (JSON-RPC):

```bash
curl -X POST http://localhost:9100/a2a \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"message/send",
       "params":{"message":{"role":"user","parts":[{"kind":"text","text":"ola"}]}}}'
```

## 4. Delegação pelo agente

O agente ReAct tem a tool `a2a_delegate(agent, task)`. Com LLM capaz de tool-calling
(ou via prompt ReAct), ele delega sozinho:

```bash
pyrag agent "delegue ao agente-financeiro a analise de custo do Q3"
```

## Métodos suportados

| Método | Descrição |
|--------|-----------|
| `message/send` | Envia mensagem, retorna Task completa |
| `message/stream` | Mesma coisa em SSE (submitted → working → completed) |
| `tasks/get` | Consulta task por id |
| `tasks/cancel` | Cancela task |
| `tasks/list` | Lista as tasks do agente |
| `agent/getAuthenticatedExtendedCard` | Agent Card |

## Ciclo de vida da task

```
submitted ──► working ──► completed
                 │
                 └──────► failed | canceled | input-required
```

## Segurança

Defina `PYRAG_A2A_API_KEY` para exigir `Authorization: Bearer <key>` nos requests A2A
(client e servidor leem a mesma variavel).