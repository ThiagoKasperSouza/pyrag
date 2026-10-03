# pyrag — visão geral do sistema

<img width="842" height="191" alt="image" src="https://github.com/user-attachments/assets/4ef1866a-5c62-4392-a795-52f5ff5c4ec3" />


pyrag e um sistema RAG completo, 100% local por padrao, com LLMs gratuitos.

## Camadas

1. **Ingestao** — PDF, Markdown, TXT, HTML, CSV, JSON e URLs -> chunks
2. **Indexacao hibrida** — Chroma (vetorial/denso) + BM25 (lexical) com reranking
3. **Geracao com fontes** — LLM local (Ollama) ou free tier (Groq/Gemini/OpenRouter)
4. **Ferramentas agenticas** — calculadora, RAG, web, SQL, arquivos, A2A
5. **Agente ReAct** — planeja e executa tarefas multiplas com ferramentas
6. **A2A** — Agent Card + JSON-RPC 2.0 para comunicacao entre agentes

## Estado validado

| Validacao | Resultado |
|-----------|-----------|
| Testes unitarios | 61 passando (sem LLM, sem rede) |
| Smoke E2E | 7/7 (inclui 2 agentes A2A via HTTP real) |
| API REST + A2A | 12/12 checks |
| LLM local (Ollama) | embeddings + chat + agente ReAct OK |

## LLMs gratuitos suportados

| Provedor | Custo | Requisitos |
|----------|-------|------------|
| Ollama | 100% local | `ollama serve` + modelos baixados |
| Groq | free tier | `GROQ_API_KEY` |
| Gemini | free tier | `GEMINI_API_KEY` |
| OpenRouter | modelos `:free` | `OPENROUTER_API_KEY` |

Sem nenhuma chave, o sistema roda em **modo extrativo** (retorna os trechos relevantes).

## Uso rapido

```bash
source .venv/bin/activate
pyrag index ./docs
pyrag rag ask "o que o sistema faz?"
pyrag rag chat
pyrag agent "use as ferramentas para comparar 3 planos de custo"
pyrag serve          # REST + A2A em :8000
pyrag a2a serve      # somente A2A em :9100
```

## Endpoints A2A

- `GET /.well-known/agent.json` — Agent Card (descoberta)
- `POST /a2a` — JSON-RPC 2.0: `message/send`, `message/stream`, `tasks/get`, `tasks/cancel`
- `GET /a2a/health`

Docs: `docs/GUIA_RAPIDO.md` (uso), `docs/ARQUITETURA.md` (design), `docs/A2A.md` (protocolo).
