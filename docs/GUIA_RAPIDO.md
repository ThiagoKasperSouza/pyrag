# Guia rapido

## 0. Antes de tudo: ativar o ambiente

O comando `pyrag` vive **dentro do venv**. Se ele nao esta ativo, o bash responde
`pyrag: command not found`. Ha tres formas de resolver:

**A) Ativar manualmente (qualquer diretorio):**

```bash
source /home/thiag/pyrag/.venv/bin/activate
```

O prompt passa a mostrar `(.venv)`. Para desativar: `deactivate`.

**B) Automatica (ja configurada no seu `~/.bashrc`):**

Ao entrar em `/home/thiag/pyrag`, o venv e ativado sozinho. Basta:

```bash
cd /home/thiag/pyrag
pyrag rag ask "sua pergunta"
```

O `(.venv)` no prompt confirma que deu certo. As linhas ficam no final do `.bashrc`,
entre comentarios `# pyrag: ativa o venv ...` — basta apagar esse bloco para
desativar o comportamento.

**C) Sem ativar nada (um comando pontual):**

```bash
/home/thiag/pyrag/.venv/bin/pyrag rag ask "sua pergunta"
# ou, de dentro do projeto:
./.venv/bin/pyrag rag ask "sua pergunta"
```

**D) Como modulo Python** (funciona sempre, sem venv ativado):

```bash
cd /home/thiag/pyrag
.venv/bin/python -m pyrag rag ask "sua pergunta"
```

Diagnostico rapido: `which pyrag` deve retornar
`/home/thiag/pyrag/.venv/bin/pyrag`. Se retornar vazio, o venv nao esta ativo.

## 1. Requisitos

- Python 3.10+ (testado em 3.12)
- [Ollama](https://ollama.com) para LLM local (opcional)

## 2. Instalacao

```bash
cd /home/thiag/pyrag
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env      # ajuste se quiser
```

## 3. LLM

**Opcao A — Ollama (local, ilimitado, gratis):**

```bash
ollama serve &
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull embeddinggemma:300m
```

Deixe `PYRAG_LLM_PROVIDER=ollama` no `.env`.

**Opcao B — Groq (free tier, bem mais rapido):**

1. Crie a key em <https://console.groq.com/keys>
2. No `.env`: `GROQ_API_KEY=gsk_...` e `PYRAG_LLM_PROVIDER=groq`

**Opcao C — Gemini / OpenRouter:** igual, com `GEMINI_API_KEY` ou `OPENROUTER_API_KEY`.

**Opcao D — sem nada:** funciona em modo extrativo (retorna os trechos com fonte).

## 4. Indexar e perguntar

```bash
pyrag index ./docs ./data/samples
pyrag status
pyrag rag ask "como a recuperacao hibrida funciona?"
pyrag rag search "BM25" --text
pyrag rag chat          # sessao com memoria
```

## 5. Agente com ferramentas

```bash
pyrag agent "use as ferramentas: calcule 15% de 3200 e depois some 500" --steps
pyrag agent "delegue ao agente-financeiro a analise do Q3"   # requer peer A2A
```

## 6. Servir

```bash
pyrag serve     # REST + A2A em :8000  (docs em /docs)
pyrag a2a serve # so A2A em :9100
```

## 7. Multiplos agentes (A2A)

```bash
# terminal 1
PYRAG_AGENT_NAME=agente-vendas PYRAG_A2A_BIND_PORT=9100 pyrag a2a serve
# terminal 2
PYRAG_AGENT_NAME=agente-rh    PYRAG_A2A_BIND_PORT=9200 pyrag a2a serve

# registra e delega
pyrag peers add vendas http://localhost:9100
pyrag peers list
pyrag a2a ask http://localhost:9100 "resuma o trimestre"
```

## 8. Testes e validacao

```bash
pytest -q                      # 61 testes, sem LLM e sem rede
bash scripts/smoke_test.sh     # 7 verificacoes E2E (inclui 2 agentes A2A via HTTP)
```

Com o servidor rodando:

```bash
pyrag serve &
.venv/bin/python scripts/api_check.py http://localhost:8000   # 12 checks REST + A2A
.venv/bin/python scripts/llm_check.py     # pipeline com LLM real + embeddings reais
.venv/bin/python scripts/agent_check.py   # agente ReAct com ferramentas
```

## 9. Notas praticas

**Modelos locais dependem de inferencia em CPU pura.** O padrao `qwen2.5:0.5b` roda em
qualquer maquina (util para testes) mas responde em ~20-30s por pergunta. Para uso real:

```bash
ollama pull llama3.1:8b     # ~4.7 GB, bom equilibrio PT-BR
ollama pull qwen2.5:7b      # ~4.7 GB, melhor em raciocinio/instrucao
```

Depois ajuste `PYRAG_CHAT_MODEL` e `PYRAG_REASONING_MODEL` no `.env`.

**Embeddings:** `embeddinggemma:300m` (300 dim, multilíngue). Para PT-BR,
`ollama pull bge-m3` funciona bem e basta trocar `PYRAG_EMBED_MODEL`.

**Agentes ReAct:** modelos muito pequenos as vezes repetem a mesma acao. O agente detecta
o loop (mesma tool + mesmos args, ou saida identica) e encerra retornando o ultimo
resultado valido — nunca trava ate o limite de iteracoes.

## 10. Performance (referencia: WSL2, CPU, sem GPU)

| Operacao | Tempo |
|----------|-------|
| Indexar 13 chunks | ~2s |
| Busca hibrida (sem LLM) | ~30ms |
| Resposta com LLM 0.5b | ~26s |
| Resposta com LLM 7-8B | ~3-6s |
| `message/send` A2A | ~40ms |

## Estrutura

```
pyrag/
├── pyrag/
│   ├── config.py       # Settings (pydantic-settings + .env)
│   ├── llm.py          # fabricas de LLM/embeddings + fallback extrativo
│   ├── models.py       # modelos RAG + modelos A2A (camelCase)
│   ├── ingest.py       # loaders (PDF/MD/HTML/CSV/JSON/URL) + chunking
│   ├── vectorstore.py  # Chroma + BM25 + rerank híbrido
│   ├── rag.py          # pipeline RAG (rewrite, contexto, citações)
│   ├── memory.py       # histórico de sessão em JSONL
│   ├── agent.py        # ReactAgent + create_agent
│   ├── tools/          # 10 ferramentas agenticas
│   ├── a2a/            # peers, card, client, server (JSON-RPC + SSE)
│   ├── api.py          # FastAPI: REST + A2A
│   └── cli.py          # Typer
├── tests/              # 61 testes
├── scripts/            # setup, smoke test, checks
└── docs/               # arquitetura, A2A, guia rápido
```